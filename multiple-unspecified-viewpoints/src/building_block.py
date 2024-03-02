import math
from typing import List, Tuple, Union

import torch

compile_mode = 'default'
compile_disable = False


def get_grid(ht: int, wd: int) -> torch.Tensor:
    row_list = torch.linspace(-1, 1, ht)
    col_list = torch.linspace(-1, 1, wd)
    row_grid, col_grid = torch.meshgrid(row_list, col_list, indexing='ij')
    grid = torch.stack([col_grid, row_grid], dim=-1)[None]  # [1, H, W, 2]
    return grid


def get_activation(activation: str) -> torch.nn.Module:
    if activation == 'relu':
        net = torch.nn.ReLU(inplace=True)
    elif activation == 'silu':
        net = torch.nn.SiLU(inplace=True)
    else:
        raise ValueError
    return net


def get_net_list(
    net: torch.nn.Module,
    activation: Union[str, None],
) -> List[torch.nn.Module]:
    torch.nn.init.xavier_uniform_(net.weight)
    if net.bias is not None:
        torch.nn.init.zeros_(net.bias)
    net_list = [net]
    if activation is not None:
        net_list.append(get_activation(activation))
    return net_list


class Permute(torch.nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x):
        x = x.permute(self.dims)
        return x

    def extra_repr(self):
        return '{dims}'.format(**self.__dict__)


class LinearPosEmbedLayer(torch.nn.Module):
    def __init__(self, ht: int, wd: int, out_features: int) -> None:
        super().__init__()
        self.register_buffer('pos_grid', get_grid(ht, wd), persistent=False)
        self.net = torch.nn.Linear(self.pos_grid.shape[-1], out_features)
        torch.nn.init.xavier_uniform_(self.net.weight)
        torch.nn.init.zeros_(self.net.bias)

    @torch.compile(mode=compile_mode, disable=compile_disable)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.net(self.pos_grid).permute(0, 3, 1, 2)
        return x


class SinusoidPosEmbedLayer(torch.nn.Module):
    def __init__(self, ht: int, wd: int, out_features: int) -> None:
        super().__init__()
        self.register_buffer('pos_grid', get_grid(ht, wd), persistent=False)
        self.net = torch.nn.Linear(self.pos_grid.shape[-1], out_features)
        init_scale = math.sqrt(6 / self.pos_grid.shape[-1])
        torch.nn.init.uniform_(self.net.weight, -init_scale, init_scale)

    @torch.compile(mode=compile_mode, disable=compile_disable)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.sin(self.net(self.pos_grid)).permute(0, 3, 1, 2)
        return x


class GRULayer(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.net = torch.nn.GRUCell(input_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.net.weight_ih)
        torch.nn.init.orthogonal_(self.net.weight_hh)
        torch.nn.init.zeros_(self.net.bias_ih)
        torch.nn.init.zeros_(self.net.bias_hh)

    @torch.compile(mode=compile_mode, disable=compile_disable)
    def forward(
        self,
        x_input: torch.Tensor,
        x_hidden: torch.Tensor,
    ) -> torch.Tensor:
        x = self.net(x_input, x_hidden)
        return x


class LinearLayer(torch.nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Union[str, None],
        bias: bool = True,
    ) -> None:
        net = torch.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        net_list = get_net_list(net, activation)
        super().__init__(*net_list)


class ConvLayer(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: Union[str, None],
    ) -> None:
        assert (kernel_size - stride) % 2 == 0
        padding = (kernel_size - stride) // 2
        net = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        net_list = get_net_list(net, activation)
        super().__init__(*net_list)


class ConvTLayer(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: Union[str, None],
    ) -> None:
        assert (kernel_size - stride) % 2 == 0
        padding = (kernel_size - stride) // 2
        net = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        net_list = get_net_list(net, activation)
        super().__init__(*net_list)


class LinearBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_features: int,
        feature_list: List[int],
        act_inner: str,
        act_out: Union[str, None],
    ) -> None:
        net_list = []
        for idx, num_features in enumerate(feature_list):
            activation = act_inner if idx < len(feature_list) - 1 else act_out
            net = torch.compile(
                LinearLayer(
                    in_features=in_features,
                    out_features=num_features,
                    activation=activation,
                ),
                mode=compile_mode,
                disable=compile_disable,
            )
            net_list.append(net)
            in_features = num_features
        self.out_features = in_features
        super().__init__(*net_list)


class ConvBlock(torch.nn.Sequential):
    def __init__(
        self,
        in_shape: Tuple[int, Union[int, None], Union[int, None]],
        channel_list: List[int],
        kernel_list: List[int],
        stride_list: List[int],
        act_inner: str,
        act_out: Union[str, None],
    ) -> None:
        assert len(channel_list) == \
            len(kernel_list) == \
            len(stride_list)
        net_list = []
        in_ch, in_ht, in_wd = in_shape
        for idx, (num_channels, kernel_size, stride) in enumerate(
                zip(channel_list, kernel_list, stride_list)):
            activation = act_inner if idx < len(channel_list) - 1 else act_out
            net = ConvLayer(
                in_channels=in_ch,
                out_channels=num_channels,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
            )
            if stride == 1:
                net = torch.compile(
                    net,
                    mode=compile_mode,
                    disable=compile_disable,
                )
            net_list.append(net)
            in_ch = num_channels
            if in_ht is not None:
                in_ht = (in_ht - 1) // stride + 1
            if in_wd is not None:
                in_wd = (in_wd - 1) // stride + 1
        self.out_shape = (in_ch, in_ht, in_wd)
        super().__init__(*net_list)


class ConvTBlock(torch.nn.Sequential):
    def __init__(
        self,
        out_shape: Tuple[int, Union[int, None], Union[int, None]],
        channel_list_rev: List[int],
        kernel_list_rev: List[int],
        stride_list_rev: List[int],
        act_inner: str,
        act_out: Union[str, None],
    ) -> None:
        assert len(channel_list_rev) == \
            len(kernel_list_rev) == \
            len(stride_list_rev)
        net_list_rev = []
        out_ch, out_ht, out_wd = out_shape
        for idx, (num_channels, kernel_size, stride) in enumerate(
                zip(channel_list_rev, kernel_list_rev, stride_list_rev)):
            activation = act_out if idx == 0 else act_inner
            net = ConvTLayer(
                in_channels=num_channels,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
            )
            if stride == 1:
                net = torch.compile(
                    net,
                    mode=compile_mode,
                    disable=compile_disable,
                )
            net_list_rev.append(net)
            out_ch = num_channels
            if out_ht is not None:
                out_ht = (out_ht - 1) // stride + 1
            if out_wd is not None:
                out_wd = (out_wd - 1) // stride + 1
        self.in_shape = (out_ch, out_ht, out_wd)
        super().__init__(*reversed(net_list_rev))
