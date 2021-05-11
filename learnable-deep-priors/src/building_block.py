import torch
import torch.nn as nn
import torch.nn.functional as nn_func


def add_nonlinearity(layers, normalization, activation):
    activation = 'linear' if activation is None else activation
    if activation == 'linear':
        sub_layers = []
    else:
        sub_layers = [] if normalization is None else [normalization]
        if activation == 'relu':
            sub_layers.append(nn.ReLU(inplace=True))
        elif activation == 'elu':
            sub_layers.append(nn.ELU(inplace=True))
            activation = 'relu'
        else:
            raise AssertionError
    nn.init.kaiming_normal_(layers[-1].weight, mode='fan_in', nonlinearity=activation)
    nn.init.zeros_(layers[-1].bias)
    layers += sub_layers
    return layers


def get_linear_ln(in_features, out_features, activation=None):
    layers = [nn.Linear(in_features, out_features)]
    normalization = nn.LayerNorm(out_features)
    layers = add_nonlinearity(layers, normalization, activation)
    return layers


def get_enc_conv_ln(in_channels, out_channels, kernel_size, stride, activation=None):
    assert kernel_size % 2 == 1
    padding = kernel_size // 2
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)]
    normalization = LayerNormConv(out_channels)
    layers = add_nonlinearity(layers, normalization, activation)
    return layers


def get_dec_conv_ln(in_channels, out_channels, kernel_size, in_size, out_size, activation=None):
    layers = [Interpolate(in_size, out_size)]
    layers += get_enc_conv_ln(in_channels, out_channels, kernel_size, stride=1, activation=activation)
    return layers


class LayerNormConv(nn.Module):

    def __init__(self, num_channels):
        super(LayerNormConv, self).__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_channels, 1, 1), requires_grad=True)

    def forward(self, x):
        weight = self.weight.expand(-1, *x.shape[-2:])
        bias = self.bias.expand(-1, *x.shape[-2:])
        return nn_func.layer_norm(x, x.shape[-3:], weight, bias)

    def extra_repr(self):
        return '{num_channels}'.format(**self.__dict__)


class Interpolate(nn.Module):

    def __init__(self, in_size, out_size):
        super(Interpolate, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):
        if self.in_size != self.out_size:
            x = nn_func.interpolate(x, size=self.out_size, mode='nearest')
        return x

    def extra_repr(self):
        if self.in_size == self.out_size:
            return 'identity'
        else:
            return 'in_size={in_size}, out_size={out_size}'.format(**self.__dict__)


class LinearBlock(nn.Sequential):

    def __init__(self, hidden_list, in_features, out_features, activation):
        layers = []
        for num_features in hidden_list:
            layers += get_linear_ln(in_features, num_features, activation=activation)
            in_features = num_features
        if out_features is not None:
            layers += get_linear_ln(in_features, out_features)
            in_features = out_features
        self.out_features = in_features
        super(LinearBlock, self).__init__(*layers)


class EncoderBlock(nn.Module):

    def __init__(self, channel_list, kernel_list, stride_list, hidden_list, in_shape, out_features, activation='elu'):
        super(EncoderBlock, self).__init__()
        assert len(channel_list) == len(kernel_list)
        assert len(channel_list) == len(stride_list)
        layers = []
        in_ch, in_ht, in_wd = in_shape
        for num_channels, kernel_size, stride in zip(channel_list, kernel_list, stride_list):
            layers += get_enc_conv_ln(in_ch, num_channels, kernel_size, stride, activation=activation)
            in_ch = num_channels
            in_ht = (in_ht - 1) // stride + 1
            in_wd = (in_wd - 1) // stride + 1
        self.conv = nn.Sequential(*layers)
        self.linear = LinearBlock(
            hidden_list=hidden_list,
            in_features=in_ch * in_ht * in_wd,
            out_features=out_features,
            activation=activation,
        )
        self.out_features = self.linear.out_features

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, channel_list_rev, kernel_list_rev, stride_list_rev, hidden_list_rev, in_features, out_shape,
                 activation='relu'):
        super(DecoderBlock, self).__init__()
        assert len(channel_list_rev) == len(kernel_list_rev)
        assert len(channel_list_rev) == len(stride_list_rev)
        layers = []
        out_ch, out_ht, out_wd = out_shape
        layer_act = None
        for num_channels, kernel_size, stride in zip(channel_list_rev, kernel_list_rev, stride_list_rev):
            in_ht = (out_ht - 1) // stride + 1
            in_wd = (out_wd - 1) // stride + 1
            sub_layers = get_dec_conv_ln(num_channels, out_ch, kernel_size, in_size=[in_ht, in_wd],
                                         out_size=[out_ht, out_wd], activation=layer_act)
            layers = sub_layers + layers
            out_ch = num_channels
            out_ht = in_ht
            out_wd = in_wd
            layer_act = activation
        hidden_list = [*reversed(hidden_list_rev)]
        out_features = out_ch * out_ht * out_wd
        if layer_act is not None:
            hidden_list.append(out_features)
            out_features = None
        self.linear = LinearBlock(
            hidden_list=hidden_list,
            in_features=in_features,
            out_features=out_features,
            activation=activation,
        )
        self.conv = nn.Sequential(*layers)
        self.in_shape = [out_ch, out_ht, out_wd]

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], *self.in_shape)
        x = self.conv(x)
        return x
