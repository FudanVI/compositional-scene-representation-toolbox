from functools import partial

import einops as E
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_func


def get_net_list(net, activation, norm='none', dim=64):
    activation = 'linear' if activation is None else activation
    nn.init.xavier_uniform_(net.weight)
    if net.bias is not None:
        nn.init.zeros_(net.bias)
    net_list = [net]
    if norm != 'none':
        if norm == 'bn':
            net_list.append(nn.BatchNorm2d(dim))
        elif norm == 'gn':
            net_list.append(nn.GroupNorm(8, dim))
        else:
            raise NotImplementedError
    if activation == 'relu':
        net_list.append(nn.ReLU(inplace=True))
    elif activation == 'tanh':
        net_list.append(nn.Tanh())
    elif activation != 'linear':
        raise ValueError
    return net_list


def get_grid(shape):
    assert len(shape) == 2
    row_list = torch.linspace(-1, 1, shape[0])
    col_list = torch.linspace(-1, 1, shape[1])
    row_grid, col_grid = torch.meshgrid(row_list, col_list)
    grid = torch.stack([col_grid, row_grid])[None]
    return grid


def get_grid_dual(shape):
    grid = get_grid(shape)
    grid = torch.cat([1 + grid, 1 - grid], dim=1) * 0.5
    return grid


class Interpolate(nn.Module):

    def __init__(self, in_size, out_size, mode):
        super(Interpolate, self).__init__()
        assert len(in_size) == len(out_size) == 2
        self.in_size = in_size
        self.out_size = out_size
        self.mode = mode
        if mode == 'nearest':
            self.fn_interpolate = partial(nn_func.interpolate, size=out_size, mode=mode)
        elif mode == 'bilinear':
            self.fn_interpolate = partial(nn_func.interpolate, size=out_size, mode=mode, align_corners=False)
        else:
            raise ValueError

    def forward(self, x):
        if self.in_size != self.out_size:
            x = self.fn_interpolate(x)
        return x

    def extra_repr(self):
        if self.in_size == self.out_size:
            return 'identity'
        else:
            return 'in_size={in_size}, out_size={out_size}, mode={mode}'.format(**self.__dict__)


def get_decoder(in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, feature_list_rev,
                activation, norm, mode, spatial_broadcast):
    if spatial_broadcast:
        dec = DecoderPos(in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, activation, norm,
                         mode)
    else:
        dec = DecoderDense(in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, feature_list_rev,
                           activation, norm, mode)
    return dec


class LinearLayer(nn.Sequential):

    def __init__(self, in_features, out_features, activation, bias=True):
        net = nn.Linear(in_features, out_features, bias=bias)
        net_list = get_net_list(net, activation)
        super(LinearLayer, self).__init__(*net_list)


class LinearBlock(nn.Sequential):

    def __init__(self, in_features, feature_list, act_inner, act_out, bias=True):
        net_list = []
        for idx, num_features in enumerate(feature_list):
            activation = act_inner if idx < len(feature_list) - 1 else act_out
            net = LinearLayer(
                in_features=in_features,
                out_features=num_features,
                activation=activation,
                bias=bias
            )
            net_list.append(net)
            in_features = num_features
        self.out_features = in_features
        super(LinearBlock, self).__init__(*net_list)


class ConvTLayer(nn.Sequential):

    def __init__(self, in_channels, out_shape, kernel_size, stride, activation, norm):
        net = self.get_net(in_channels, out_shape, kernel_size, stride)
        net_list = get_net_list(net, activation, norm, dim=out_shape[0])
        super(ConvTLayer, self).__init__(*net_list)

    @staticmethod
    def get_net(in_channels, out_shape, kernel_size, stride):
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        output_padding = [(n - 1) % stride for n in out_shape[1:]]
        net = nn.ConvTranspose2d(
            in_channels, out_shape[0], kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        return net


class UpConvLayer(nn.Sequential):

    def __init__(self, in_shape, out_shape, kernel_size, activation, norm, mode):
        net = self.get_net(in_shape, out_shape, kernel_size)
        net_list = get_net_list(net, activation, norm=norm, dim=out_shape[0])
        net_list = [Interpolate(in_shape[1:], out_shape[1:], mode)] + net_list
        super(UpConvLayer, self).__init__(*net_list)

    @staticmethod
    def get_net(in_shape, out_shape, kernel_size):
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        net = nn.Conv2d(in_shape[0], out_shape[0], kernel_size, stride=1, padding=padding)
        return net


class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, norm='none'):
        net = self.get_net(in_channels, out_channels, kernel_size, stride)
        net_list = get_net_list(net, activation, norm=norm, dim=out_channels)
        super(ConvLayer, self).__init__(*net_list)

    @staticmethod
    def get_net(in_channels, out_channels, kernel_size, stride):
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        net = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        return net


class ConvTBlock(nn.Sequential):

    def __init__(self, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, act_inner, norm_inner, act_out):
        assert len(channel_list_rev) == len(kernel_list_rev) == len(stride_list_rev)
        net_list_rev = []
        for idx, (num_channels, kernel_size, stride) in enumerate(
                zip(channel_list_rev, kernel_list_rev, stride_list_rev)):
            activation = act_out if idx == 0 else act_inner
            norm = 'none' if idx == 0 else norm_inner
            net = ConvTLayer(
                in_channels=num_channels,
                out_shape=out_shape,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm
            )
            net_list_rev.append(net)
            out_shape = [num_channels] + [(val - 1) // stride + 1 for val in out_shape[1:]]
        self.in_shape = out_shape
        super(ConvTBlock, self).__init__(*reversed(net_list_rev))


class UpConvBlock(nn.Sequential):

    def __init__(self, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, act_inner, norm_inner, act_out,
                 mode):
        assert len(channel_list_rev) == len(kernel_list_rev) == len(stride_list_rev)
        net_list_rev = []
        for idx, (num_channels, kernel_size, stride) in enumerate(
                zip(channel_list_rev, kernel_list_rev, stride_list_rev)):
            activation = act_out if idx == 0 else act_inner
            norm = 'none' if idx == 0 else norm_inner
            in_shape = [num_channels] + [(val - 1) // stride + 1 for val in out_shape[1:]]
            net = UpConvLayer(
                in_shape=in_shape,
                out_shape=out_shape,
                kernel_size=kernel_size,
                activation=activation,
                norm=norm,
                mode=mode,
            )
            net_list_rev.append(net)
            out_shape = in_shape
        self.in_shape = out_shape
        super(UpConvBlock, self).__init__(*reversed(net_list_rev))


class DecoderPos(nn.Module):

    def __init__(self, in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, activation, norm,
                 mode):
        super(DecoderPos, self).__init__()
        channel_list_rev = channel_list_rev + [in_features]
        kernel_list_rev = kernel_list_rev
        stride_list_rev = stride_list_rev
        if mode == 'conv_t':
            self.net_image = ConvTBlock(
                out_shape=out_shape,
                channel_list_rev=channel_list_rev,
                kernel_list_rev=kernel_list_rev,
                stride_list_rev=stride_list_rev,
                act_inner=activation,
                norm_inner=norm,
                act_out=None,
            )
        else:
            self.net_image = UpConvBlock(
                out_shape=out_shape,
                channel_list_rev=channel_list_rev,
                kernel_list_rev=kernel_list_rev,
                stride_list_rev=stride_list_rev,
                act_inner=activation,
                norm_inner=norm,
                act_out=None,
                mode=mode,
            )
        self.register_buffer('grid', get_grid_dual(self.net_image.in_shape[1:]))
        self.net_grid = ConvLayer(
            in_channels=4,
            out_channels=self.net_image.in_shape[0],
            kernel_size=1,
            stride=1,
            activation=None,
            norm='none'
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x[..., None, None] + self.net_grid(self.grid)
        else:
            x = x + self.net_grid(self.grid)
        x = self.net_image(x)
        return x


class DecoderDense(nn.Module):

    def __init__(self, in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, feature_list_rev,
                 activation, norm, mode):
        super(DecoderDense, self).__init__()
        if mode == 'conv_t':
            self.conv = ConvTBlock(
                out_shape=out_shape,
                channel_list_rev=channel_list_rev,
                kernel_list_rev=kernel_list_rev,
                stride_list_rev=stride_list_rev,
                act_inner=activation,
                norm_inner=norm,
                act_out=None,
            )
        else:
            self.conv = UpConvBlock(
                out_shape=out_shape,
                channel_list_rev=channel_list_rev,
                kernel_list_rev=kernel_list_rev,
                stride_list_rev=stride_list_rev,
                act_inner=activation,
                norm_inner=norm,
                act_out=None,
                mode=mode,
            )
        self.linear = LinearBlock(
            in_features=in_features,
            feature_list=[*reversed(feature_list_rev)] + [np.prod(self.conv.in_shape)],
            act_inner=activation,
            act_out=None if len(channel_list_rev) == 0 else activation,
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], *self.conv.in_shape)
        x = self.conv(x)
        return x


class ConvBlock(nn.Sequential):

    def __init__(self, in_shape, channel_list, kernel_list, stride_list, act_inner, norm_inner, act_out):
        assert len(channel_list) == len(kernel_list) == len(stride_list)
        net_list = []
        in_ch, in_ht, in_wd = in_shape
        for idx, (num_channels, kernel_size, stride) in enumerate(zip(channel_list, kernel_list, stride_list)):
            activation = act_inner if idx < len(channel_list) - 1 else act_out
            norm = norm_inner if idx < len(channel_list) - 1 else 'none'
            net = ConvLayer(
                in_channels=in_ch,
                out_channels=num_channels,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm
            )
            net_list.append(net)
            in_ch = num_channels
            in_ht = (in_ht - 1) // stride + 1
            in_wd = (in_wd - 1) // stride + 1
        self.out_shape = [in_ch, in_ht, in_wd]
        super(ConvBlock, self).__init__(*net_list)


class EncoderDense(nn.Module):
    def __init__(self, in_shape, channel_list, kernel_list, stride_list, act_inner, norm_inner, act_out,
                 feature_list, out_features, linear_act_inner='relu', linear_act_out=None):
        super(EncoderDense, self).__init__()
        self.conv = ConvBlock(in_shape, channel_list, kernel_list, stride_list, act_inner, norm_inner, act_out)
        self.linear = LinearBlock(in_features=np.prod(self.conv.out_shape),
                                  feature_list=feature_list + [out_features],
                                  act_inner=linear_act_inner,
                                  act_out=linear_act_out)
        self.out_features = out_features

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, np.prod(self.conv.out_shape))
        x = self.linear(x)
        return x


class EncoderPos(nn.Module):

    def __init__(self, in_shape, channel_list, kernel_list, stride_list, activation, norm):
        super(EncoderPos, self).__init__()
        self.net_image = ConvBlock(
            in_shape=in_shape,
            channel_list=channel_list,
            kernel_list=kernel_list,
            stride_list=stride_list,
            act_inner=activation,
            norm_inner=norm,
            act_out=activation,
        )
        self.register_buffer('grid', get_grid_dual(self.net_image.out_shape[1:]))
        self.net_grid = ConvLayer(
            in_channels=4,
            out_channels=self.net_image.out_shape[0],
            kernel_size=1,
            stride=1,
            activation=None,
            norm=norm
        )

    def forward(self, x):
        x = self.net_image(x)
        x = x + self.net_grid(self.grid)
        return x


class EncoderLSTM(nn.Module):

    def __init__(self, in_shape, channel_list, kernel_list, stride_list, act_inner, norm_inner, act_out,
                 feature_list, out_features, state_size):
        super(EncoderLSTM, self).__init__()
        self.enc = EncoderDense(
            in_shape=in_shape,
            channel_list=channel_list,
            kernel_list=kernel_list,
            stride_list=stride_list,
            act_inner=act_inner,
            norm_inner=norm_inner,
            act_out=act_out,
            feature_list=feature_list,
            out_features=out_features,
        )
        self.lstm = nn.LSTMCell(self.enc.out_features, state_size)

    def forward(self, inputs, states=None):
        x = self.enc(inputs)
        states = self.lstm(x, states)
        return states


class PositionalEmbedding3D(nn.Module):
    def __init__(self, dim):
        super(PositionalEmbedding3D, self).__init__()
        self.channels = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2) / self.channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        device = tensor.device
        assert tensor.ndim == 5, "3D positional embedding only works for 5D Tensor"
        assert (
                tensor.shape[-3] <= self.channels
        ), "Channel dimension must be smaller than the maximum channels"
        b, t, c, h, w = tensor.shape
        t_inp = torch.einsum("i,j -> ij", torch.arange(t).to(device), self.inv_freq)
        h_inp = torch.einsum("i,j -> ij", torch.arange(h).to(device), self.inv_freq)
        w_inp = torch.einsum("i,j -> ij", torch.arange(w).to(device), self.inv_freq)

        emb_t = torch.cat((torch.sin(t_inp), torch.cos(t_inp)), dim=-1)
        emb_h = torch.cat((torch.sin(h_inp), torch.cos(h_inp)), dim=-1)
        emb_w = torch.cat((torch.sin(w_inp), torch.cos(w_inp)), dim=-1)

        emb_t = E.repeat(emb_t, "t c -> b t h w c", b=b, h=h, w=w)
        emb_h = E.repeat(emb_h, "h c -> b t h w c", b=b, t=t, w=w)
        emb_w = E.repeat(emb_w, "w c -> b t h w c", b=b, t=t, h=h)

        return torch.cat((emb_t, emb_h, emb_w), dim=-1)[..., :c] + tensor.permute(0, 1, 3, 4, 2).contiguous()


class DynamicPositionalEmbedding(nn.Module):
    def __init__(self, dim, pos_concat, dynamic_timestep):
        super(DynamicPositionalEmbedding, self).__init__()
        self.channels = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2) / self.channels))
        self.register_buffer("inv_freq", inv_freq)
        self.concat = pos_concat
        if self.concat:
            self.net_grid = ConvLayer(
                in_channels=3 * dim,
                out_channels=dim,
                kernel_size=1,
                stride=1,
                activation=None
            )
        self.dynamic_timestep = dynamic_timestep

    def forward(self, tensor, timesteps=None):
        device = tensor.device
        assert tensor.ndim == 5, "3D positional embedding only works for 5D Tensor"
        assert (
                tensor.shape[-3] <= self.channels
        ), "Channel dimension must be smaller than the maximum channels"
        b, t, c, h, w = tensor.shape
        if self.dynamic_timestep:
            assert timesteps is not None
            t_inp = torch.einsum("bi,j -> bij", timesteps.to(device), self.inv_freq)
            emb_t = torch.cat((torch.sin(t_inp), torch.cos(t_inp)), dim=-1)
            emb_t = E.repeat(emb_t, "b t c -> b t h w c", h=h, w=w)
        else:
            t_inp = torch.einsum("i,j -> ij", torch.arange(t).to(device), self.inv_freq)
            emb_t = torch.cat((torch.sin(t_inp), torch.cos(t_inp)), dim=-1)
            emb_t = E.repeat(emb_t, "t c -> b t h w c", b=b, h=h, w=w)
        h_inp = torch.einsum("i,j -> ij", torch.arange(h).to(device), self.inv_freq)
        w_inp = torch.einsum("i,j -> ij", torch.arange(w).to(device), self.inv_freq)
        emb_h = torch.cat((torch.sin(h_inp), torch.cos(h_inp)), dim=-1)
        emb_w = torch.cat((torch.sin(w_inp), torch.cos(w_inp)), dim=-1)
        emb_h = E.repeat(emb_h, "h c -> b t h w c", b=b, t=t, w=w)
        emb_w = E.repeat(emb_w, "w c -> b t h w c", b=b, t=t, h=h)

        emb = torch.cat((emb_t, emb_h, emb_w), dim=-1)
        if self.concat:
            emb = E.rearrange(emb, "b t h w c -> (b t) c h w")
            emb = self.net_grid(emb)
            emb = E.rearrange(emb, "(b t) c h w -> b t h w c", b=b, t=t)
        tensor = E.rearrange(tensor, "b t c h w -> b t h w c")
        return emb[..., :c] + tensor


class GRULayer(nn.GRUCell):

    def __init__(self, input_size, hidden_size):
        super(GRULayer, self).__init__(input_size, hidden_size)
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)


if __name__ == '__main__':
    net = get_decoder(
        in_features=67,
        out_shape=[4, 64, 64],
        channel_list_rev=[32, 64, 128],
        kernel_list_rev=[3, 5, 5],
        stride_list_rev=[1, 2, 2, ],
        feature_list_rev=[],
        activation='relu',
        norm='gn',
        mode='nearest',
        spatial_broadcast=False,
    )
