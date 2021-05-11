import torch
import torch.nn as nn
from building_block import LinearBlock, EncoderBlock, DecoderBlock


class EncoderLSTM(nn.Module):

    def __init__(self, channel_list, kernel_list, stride_list, hidden_list, in_shape, state_size):
        super(EncoderLSTM, self).__init__()
        self.enc = EncoderBlock(
            channel_list=channel_list,
            kernel_list=kernel_list,
            stride_list=stride_list,
            hidden_list=hidden_list,
            in_shape=in_shape,
            out_features=None,
        )
        self.lstm = nn.LSTMCell(self.enc.out_features, state_size)

    def forward(self, inputs, states=None):
        x = inputs * 2 - 1
        x = self.enc(x)
        states = self.lstm(x, states)
        return states


class InitializerBck(EncoderLSTM):

    def __init__(self, config):
        super(InitializerBck, self).__init__(
            channel_list=config['init_bck_channel'],
            kernel_list=config['init_bck_kernel'],
            stride_list=config['init_bck_stride'],
            hidden_list=config['init_bck_hidden'],
            in_shape=config['image_shape'],
            state_size=config['state_bck_size'],
        )


class InitializerObj(nn.Module):

    def __init__(self, config):
        super(InitializerObj, self).__init__()
        self.upd = EncoderLSTM(
            channel_list=config['init_upd_channel'],
            kernel_list=config['init_upd_kernel'],
            stride_list=config['init_upd_stride'],
            hidden_list=config['init_upd_hidden'],
            in_shape=[config['image_shape'][0] * 2 + 1, *config['image_shape'][1:]],
            state_size=config['init_upd_state'],
        )
        self.enc = LinearBlock(
            hidden_list=config['init_obj_hidden'],
            in_features=config['init_upd_state'],
            out_features=None,
            activation='elu',
        )
        self.lstm = nn.LSTMCell(self.enc.out_features, config['state_obj_size'])

    def forward(self, inputs, states_upd):
        states_upd = self.upd(inputs, states_upd)
        x = self.enc(states_upd[0])
        states_obj = self.lstm(x)
        return states_obj, states_upd


class UpdaterBck(nn.Module):

    def __init__(self, config):
        super(UpdaterBck, self).__init__()
        self.enc = EncoderBlock(
            channel_list=config['dec_bck_channel_rev'],
            kernel_list=config['dec_bck_kernel_rev'],
            stride_list=config['dec_bck_stride_rev'],
            hidden_list=config['dec_bck_hidden_rev'],
            in_shape=config['image_shape'],
            out_features=None,
        )
        self.lstm = nn.LSTMCell(self.enc.out_features, config['state_bck_size'])

    def forward(self, inputs, states):
        x = self.enc(inputs)
        states = self.lstm(x, states)
        return states


class UpdaterObj(nn.Module):

    def __init__(self, config):
        super(UpdaterObj, self).__init__()
        self.enc_apc = EncoderBlock(
            channel_list=config['dec_apc_channel_rev'],
            kernel_list=config['dec_apc_kernel_rev'],
            stride_list=config['dec_apc_stride_rev'],
            hidden_list=config['dec_apc_hidden_rev'],
            in_shape=config['image_shape'],
            out_features=None,
        )
        self.enc_shp = EncoderBlock(
            channel_list=config['dec_shp_channel_rev'],
            kernel_list=config['dec_shp_kernel_rev'],
            stride_list=config['dec_shp_stride_rev'],
            hidden_list=config['dec_shp_hidden_rev'],
            in_shape=[1, *config['image_shape'][1:]],
            out_features=None,
        )
        self.lstm = nn.LSTMCell(self.enc_apc.out_features + self.enc_shp.out_features, config['state_obj_size'])

    def forward(self, inputs_apc, inputs_shp, states):
        inputs_apc = inputs_apc.reshape(-1, *inputs_apc.shape[2:])
        inputs_shp = inputs_shp.reshape(-1, *inputs_shp.shape[2:])
        x_apc = self.enc_apc(inputs_apc)
        x_shp = self.enc_shp(inputs_shp)
        x = torch.cat([x_apc, x_shp], dim=-1)
        states = self.lstm(x, states)
        return states


class DecoderBck(nn.Module):

    def __init__(self, config):
        super(DecoderBck, self).__init__()
        self.dec = DecoderBlock(
            channel_list_rev=config['dec_bck_channel_rev'],
            kernel_list_rev=config['dec_bck_kernel_rev'],
            stride_list_rev=config['dec_bck_stride_rev'],
            hidden_list_rev=config['dec_bck_hidden_rev'],
            in_features=config['state_bck_size'],
            out_shape=config['image_shape'],
        )

    def forward(self, x):
        unbiased_bck = self.dec(x)
        bck = (unbiased_bck + 1) * 0.5
        return {'bck': bck}


class DecoderObj(nn.Module):

    def __init__(self, config):
        super(DecoderObj, self).__init__()
        self.dec_apc = DecoderBlock(
            channel_list_rev=config['dec_apc_channel_rev'],
            kernel_list_rev=config['dec_apc_kernel_rev'],
            stride_list_rev=config['dec_apc_stride_rev'],
            hidden_list_rev=config['dec_apc_hidden_rev'],
            in_features=config['state_obj_size'],
            out_shape=config['image_shape'],
        )
        self.dec_shp = DecoderBlock(
            channel_list_rev=config['dec_shp_channel_rev'],
            kernel_list_rev=config['dec_shp_kernel_rev'],
            stride_list_rev=config['dec_shp_stride_rev'],
            hidden_list_rev=config['dec_shp_hidden_rev'],
            in_features=config['state_obj_size'],
            out_shape=[1, *config['image_shape'][1:]],
        )

    def forward(self, x, obj_slots):
        unbiased_apc = self.dec_apc(x)
        apc = (unbiased_apc + 1) * 0.5
        logits_shp = self.dec_shp(x)
        shp = torch.sigmoid(logits_shp)
        result = {'apc': apc, 'shp': shp, 'logits_shp': logits_shp}
        result = {key: val.reshape(obj_slots, -1, *val.shape[1:]) for key, val in result.items()}
        return result
