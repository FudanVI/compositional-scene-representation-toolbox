import torch.nn as nn
from building_block import EncoderBlock, DecoderBlock, RNN


class Updater(nn.Module):

    def __init__(self, config):
        super(Updater, self).__init__()
        self.state_size = config['state_size']
        self.layer_norm = None
        if config['ln_inputs']:
            self.layer_norm = nn.LayerNorm(config['image_shape'], elementwise_affine=False)
        self.enc = EncoderBlock(
            channel_list=config['dec_channel_list_rev'],
            kernel_list=config['dec_kernel_list_rev'],
            stride_list=config['dec_stride_list_rev'],
            hidden_list=config['dec_hidden_list_rev'],
            in_shape=config['image_shape'],
            out_features=None,
        )
        self.rnn = RNN(self.enc.out_features, self.state_size)

    def forward(self, inputs, states):
        inputs = inputs.reshape(-1, *inputs.shape[2:])
        if self.layer_norm is not None:
            inputs = self.layer_norm(inputs)
        x = self.enc(inputs)
        outputs, states = self.rnn(x, states)
        return outputs, states


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.dec = DecoderBlock(
            channel_list_rev=config['dec_channel_list_rev'],
            kernel_list_rev=config['dec_kernel_list_rev'],
            stride_list_rev=config['dec_stride_list_rev'],
            hidden_list_rev=config['dec_hidden_list_rev'],
            in_features=config['state_size'],
            out_shape=config['image_shape'],
        )

    def forward(self, x, num_slots):
        x = self.dec(x)
        x = x.reshape(num_slots, -1, *x.shape[1:])
        return x
