import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from functools import partial


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


def get_net_list(net, activation):
    activation = 'linear' if activation is None else activation
    nn.init.xavier_uniform_(net.weight)
    if net.bias is not None:
        nn.init.zeros_(net.bias)
    net_list = [net]
    if activation == 'relu':
        net_list.append(nn.ReLU(inplace=True))
    elif activation != 'linear':
        raise ValueError
    return net_list


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


class GRULayer(nn.GRUCell):

    def __init__(self, input_size, hidden_size):
        super(GRULayer, self).__init__(input_size, hidden_size)
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)


class MALayer(nn.Module):

    def __init__(self, num_features, num_heads, activation):
        super(MALayer, self).__init__()
        self.net = nn.MultiheadAttention(num_features, num_heads)
        activation = 'linear' if activation is None else activation
        if activation == 'linear':
            self.act = None
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError

    def forward(self, x):
        x = x.transpose(0, 1).contiguous()
        x, _ = self.net(x, x, x)
        x = x.transpose(0, 1).contiguous()
        if self.act is not None:
            x = self.act(x)
        return x


class LinearLayer(nn.Sequential):

    def __init__(self, in_features, out_features, activation, bias=True):
        net = nn.Linear(in_features, out_features, bias=bias)
        net_list = get_net_list(net, activation)
        super(LinearLayer, self).__init__(*net_list)


class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation):
        net = self.get_net(in_channels, out_channels, kernel_size, stride)
        net_list = get_net_list(net, activation)
        super(ConvLayer, self).__init__(*net_list)

    @staticmethod
    def get_net(in_channels, out_channels, kernel_size, stride):
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        net = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        return net


class UpConvLayer(nn.Sequential):

    def __init__(self, in_shape, out_shape, kernel_size, activation, mode):
        net = self.get_net(in_shape, out_shape, kernel_size)
        net_list = get_net_list(net, activation)
        net_list = [Interpolate(in_shape[1:], out_shape[1:], mode)] + net_list
        super(UpConvLayer, self).__init__(*net_list)

    @staticmethod
    def get_net(in_shape, out_shape, kernel_size):
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        net = nn.Conv2d(in_shape[0], out_shape[0], kernel_size, stride=1, padding=padding)
        return net


class ConvTLayer(nn.Sequential):

    def __init__(self, in_channels, out_shape, kernel_size, stride, activation):
        net = self.get_net(in_channels, out_shape, kernel_size, stride)
        net_list = get_net_list(net, activation)
        super(ConvTLayer, self).__init__(*net_list)

    @staticmethod
    def get_net(in_channels, out_shape, kernel_size, stride):
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        output_padding = [(n - 1) % stride for n in out_shape[1:]]
        net = nn.ConvTranspose2d(
            in_channels, out_shape[0], kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        return net


class MABlock(nn.Sequential):

    def __init__(self, num_features, head_list, act_inner, act_out):
        net_list = []
        for idx, num_heads in enumerate(head_list):
            activation = act_inner if idx < len(head_list) - 1 else act_out
            net = MALayer(
                num_features=num_features,
                num_heads=num_heads,
                activation=activation,
            )
            net_list.append(net)
        super(MABlock, self).__init__(*net_list)


class LinearBlock(nn.Sequential):

    def __init__(self, in_features, feature_list, act_inner, act_out):
        net_list = []
        for idx, num_features in enumerate(feature_list):
            activation = act_inner if idx < len(feature_list) - 1 else act_out
            net = LinearLayer(
                in_features=in_features,
                out_features=num_features,
                activation=activation,
            )
            net_list.append(net)
            in_features = num_features
        self.out_features = in_features
        super(LinearBlock, self).__init__(*net_list)


class ConvBlock(nn.Sequential):

    def __init__(self, in_shape, channel_list, kernel_list, stride_list, act_inner, act_out):
        assert len(channel_list) == len(kernel_list) == len(stride_list)
        net_list = []
        in_ch, in_ht, in_wd = in_shape
        for idx, (num_channels, kernel_size, stride) in enumerate(zip(channel_list, kernel_list, stride_list)):
            activation = act_inner if idx < len(channel_list) - 1 else act_out
            net = ConvLayer(
                in_channels=in_ch,
                out_channels=num_channels,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
            )
            net_list.append(net)
            in_ch = num_channels
            in_ht = (in_ht - 1) // stride + 1
            in_wd = (in_wd - 1) // stride + 1
        self.out_shape = [in_ch, in_ht, in_wd]
        super(ConvBlock, self).__init__(*net_list)


class UpConvBlock(nn.Sequential):

    def __init__(self, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, act_inner, act_out, mode):
        assert len(channel_list_rev) == len(kernel_list_rev) == len(stride_list_rev)
        net_list_rev = []
        for idx, (num_channels, kernel_size, stride) in enumerate(
                zip(channel_list_rev, kernel_list_rev, stride_list_rev)):
            activation = act_out if idx == 0 else act_inner
            in_shape = [num_channels] + [(val - 1) // stride + 1 for val in out_shape[1:]]
            net = UpConvLayer(
                in_shape=in_shape,
                out_shape=out_shape,
                kernel_size=kernel_size,
                activation=activation,
                mode=mode,
            )
            net_list_rev.append(net)
            out_shape = in_shape
        self.in_shape = out_shape
        super(UpConvBlock, self).__init__(*reversed(net_list_rev))


class ConvTBlock(nn.Sequential):

    def __init__(self, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, act_inner, act_out):
        assert len(channel_list_rev) == len(kernel_list_rev) == len(stride_list_rev)
        net_list_rev = []
        for idx, (num_channels, kernel_size, stride) in enumerate(
                zip(channel_list_rev, kernel_list_rev, stride_list_rev)):
            activation = act_out if idx == 0 else act_inner
            net = ConvTLayer(
                in_channels=num_channels,
                out_shape=out_shape,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
            )
            net_list_rev.append(net)
            out_shape = [num_channels] + [(val - 1) // stride + 1 for val in out_shape[1:]]
        self.in_shape = out_shape
        super(ConvTBlock, self).__init__(*reversed(net_list_rev))


class EncoderPos(nn.Module):

    def __init__(self, in_shape, channel_list, kernel_list, stride_list, activation):
        super(EncoderPos, self).__init__()
        self.net_image = ConvBlock(
            in_shape=in_shape,
            channel_list=channel_list,
            kernel_list=kernel_list,
            stride_list=stride_list,
            act_inner=activation,
            act_out=activation,
        )
        self.register_buffer('grid', get_grid_dual(self.net_image.out_shape[1:]))
        self.net_grid = ConvLayer(
            in_channels=4,
            out_channels=self.net_image.out_shape[0],
            kernel_size=1,
            stride=1,
            activation=None,
        )

    def forward(self, x):
        x = self.net_image(x)
        x = x + self.net_grid(self.grid)
        return x


class DecoderPos(nn.Module):

    def __init__(self, in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, activation, mode):
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
                act_out=None,
            )
        else:
            self.net_image = UpConvBlock(
                out_shape=out_shape,
                channel_list_rev=channel_list_rev,
                kernel_list_rev=kernel_list_rev,
                stride_list_rev=stride_list_rev,
                act_inner=activation,
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
        )

    def forward(self, x):
        x = x[..., None, None] + self.net_grid(self.grid)
        x = self.net_image(x)
        return x


class DecoderDense(nn.Module):

    def __init__(self, in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, feature_list_rev,
                 activation, mode):
        super(DecoderDense, self).__init__()
        if mode == 'conv_t':
            self.conv = ConvTBlock(
                out_shape=out_shape,
                channel_list_rev=channel_list_rev,
                kernel_list_rev=kernel_list_rev,
                stride_list_rev=stride_list_rev,
                act_inner=activation,
                act_out=None,
            )
        else:
            self.conv = UpConvBlock(
                out_shape=out_shape,
                channel_list_rev=channel_list_rev,
                kernel_list_rev=kernel_list_rev,
                stride_list_rev=stride_list_rev,
                act_inner=activation,
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


def get_decoder(in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, feature_list_rev,
                activation, mode, spatial_broadcast):
    if spatial_broadcast:
        dec = DecoderPos(in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, activation, mode)
    else:
        dec = DecoderDense(in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, feature_list_rev,
                           activation, mode)
    return dec


def compute_variable_full(x_view, x_attr):
    x_view_expand = x_view[:, :, None].expand(-1, -1, x_attr.shape[1], -1)
    x_attr_expand = x_attr[:, None].expand(-1, x_view.shape[1], -1, -1)
    x = torch.cat([x_view_expand, x_attr_expand], dim=-1)
    return x


class SlotAttentionMulti(nn.Module):

    def __init__(self, num_steps, qry_size, slot_view_size, slot_attr_size, in_features, feature_res_list,
                 activation):
        super(SlotAttentionMulti, self).__init__()
        self.num_steps = num_steps
        self.slot_view_size = slot_view_size
        self.slot_attr_size = slot_attr_size
        slot_full_size = slot_view_size + slot_attr_size
        self.coef_qry = 1 / math.sqrt(qry_size)
        self.layer_norm_in = nn.LayerNorm(in_features)
        self.layer_norm_qry = nn.LayerNorm(slot_full_size)
        self.layer_norm_res = nn.LayerNorm(slot_full_size)
        self.view_loc = nn.Parameter(torch.zeros([slot_view_size]), requires_grad=True)
        self.view_log_scl = nn.Parameter(torch.zeros([slot_view_size]), requires_grad=True)
        self.attr_loc = nn.Parameter(torch.zeros([slot_attr_size]), requires_grad=True)
        self.attr_log_scl = nn.Parameter(torch.zeros([slot_attr_size]), requires_grad=True)
        self.net_qry = LinearLayer(
            in_features=slot_full_size,
            out_features=qry_size,
            activation=None,
            bias=False,
        )
        self.net_key = LinearLayer(
            in_features=in_features,
            out_features=qry_size,
            activation=None,
            bias=False,
        )
        self.net_val = LinearLayer(
            in_features=in_features,
            out_features=slot_full_size,
            activation=None,
            bias=False,
        )
        self.gru = GRULayer(slot_full_size, slot_full_size)
        self.net_res = LinearBlock(
            in_features=slot_full_size,
            feature_list=feature_res_list + [slot_full_size],
            act_inner=activation,
            act_out=None,
        )

    def forward(self, x, num_slots, slots_attr=None):
        batch_size, num_views = x.shape[:2]
        x = self.layer_norm_in(x)
        x_key = self.net_key(x)
        x_val = self.net_val(x)
        noise_view = torch.randn([batch_size, num_views, self.slot_view_size], device=x.device)
        slots_view = self.view_loc + torch.exp(self.view_log_scl) * noise_view
        if slots_attr is None:
            noise_attr = torch.randn([batch_size, num_slots, self.slot_attr_size], device=x.device)
            slots_attr = self.attr_loc + torch.exp(self.attr_log_scl) * noise_attr
            infer_attr = True
        else:
            infer_attr = False
        for _ in range(self.num_steps):
            slots_full = compute_variable_full(slots_view, slots_attr)
            x = self.layer_norm_qry(slots_full)
            x_qry = self.net_qry(x) * self.coef_qry
            logits_attn = torch.einsum('bvni,bvsi->bvns', x_key, x_qry)
            logits_attn = torch.log_softmax(logits_attn, dim=-1)
            attn = torch.softmax(logits_attn, dim=-2)
            updates = torch.einsum('bvns,bvni->bvsi', attn, x_val)
            updates = updates.reshape(-1, updates.shape[-1])
            slots_full = slots_full.reshape(-1, slots_full.shape[-1])
            slots_full_main = self.gru(updates, slots_full).reshape(batch_size, num_views, num_slots, -1)
            x = self.layer_norm_res(slots_full_main)
            slots_full_res = self.net_res(x)
            slots_full = slots_full_main + slots_full_res
            slots_view_raw, slots_attr_raw = slots_full.split([self.slot_view_size, self.slot_attr_size], dim=-1)
            slots_view = slots_view_raw.mean(2)
            if infer_attr:
                slots_attr = slots_attr_raw.mean(1)
        outputs = {'slots_view': slots_view, 'slots_attr': slots_attr}
        return outputs
