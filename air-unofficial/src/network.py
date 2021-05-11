import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as nn_func
from building_block import LinearBlock, EncoderBlock, DecoderBlock


def reparameterize_normal(mu, logvar):
    std = torch.exp(0.5 * logvar)
    noise = torch.randn_like(std)
    return mu + std * noise


def compute_kld_normal(mu, logvar, prior_mu, prior_logvar):
    prior_invvar = torch.exp(-prior_logvar)
    kld = 0.5 * (prior_logvar - logvar + prior_invvar * ((mu - prior_mu).pow(2) + logvar.exp()) - 1)
    return kld.sum(-1)


class Updater(nn.Module):

    def __init__(self, config):
        super(Updater, self).__init__()
        self.enc = EncoderBlock(
            channel_list=config['upd_channel'],
            kernel_list=config['upd_kernel'],
            stride_list=config['upd_stride'],
            hidden_list=config['upd_hidden'],
            in_shape=[config['image_shape'][0] * 2, *config['image_shape'][1:]],
            out_features=None,
        )
        self.lstm = nn.LSTMCell(self.enc.out_features, config['state_size'])

    def forward(self, inputs, states):
        x = inputs * 2 - 1
        x = self.enc(x)
        states = self.lstm(x, states)
        return states


class DecoderResidual(nn.Module):

    def __init__(self, avg_hidden_list_rev, res_channel_list_rev, res_kernel_list_rev, res_stride_list_rev,
                 res_hidden_list_rev, in_features, image_shape):
        super(DecoderResidual, self).__init__()
        self.dec_avg = LinearBlock(
            hidden_list=reversed(avg_hidden_list_rev),
            in_features=in_features,
            out_features=image_shape[0],
        )
        self.dec_res = DecoderBlock(
            channel_list_rev=res_channel_list_rev,
            kernel_list_rev=res_kernel_list_rev,
            stride_list_rev=res_stride_list_rev,
            hidden_list_rev=res_hidden_list_rev,
            in_features=in_features,
            out_shape=image_shape,
        )

    def forward(self, x):
        x_avg = self.dec_avg(x)[..., None, None]
        x_res = self.dec_res(x)
        x = (x_avg + x_res + 1) * 0.5
        return x


class NetworkBack(nn.Module):

    def __init__(self, config):
        super(NetworkBack, self).__init__()
        latent_size = config['latent_back_size']
        self.register_buffer('prior_mu', torch.zeros([latent_size]))
        self.register_buffer('prior_logvar', torch.zeros([latent_size]))
        self.enc = EncoderBlock(
            channel_list=config['enc_back_channel'],
            kernel_list=config['enc_back_kernel'],
            stride_list=config['enc_back_stride'],
            hidden_list=config['enc_back_hidden'],
            in_shape=config['image_shape'],
            out_features=latent_size * 2,
        )
        self.dec = DecoderResidual(
            avg_hidden_list_rev=config['dec_back_avg_hidden_rev'],
            res_channel_list_rev=config['dec_back_res_channel_rev'],
            res_kernel_list_rev=config['dec_back_res_kernel_rev'],
            res_stride_list_rev=config['dec_back_res_stride_rev'],
            res_hidden_list_rev=config['dec_back_res_hidden_rev'],
            in_features=latent_size,
            image_shape=config['image_shape'],
        )

    def encode(self, x):
        x = x * 2 - 1
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, mu, logvar):
        sample = reparameterize_normal(mu, logvar)
        back = self.dec(sample)
        return back

    def forward(self, x):
        mu, logvar = self.encode(x)
        back = self.decode(mu, logvar)
        kld = compute_kld_normal(mu, logvar, self.prior_mu, self.prior_logvar)
        result = {'back': back, 'back_kld': kld}
        return result


class NetworkPres(nn.Module):

    def __init__(self, config):
        super(NetworkPres, self).__init__()
        self.enc = LinearBlock(
            hidden_list=config['enc_pres_hidden'],
            in_features=config['state_size'],
            out_features=1,
        )

    def forward(self, x, temp, hard):
        logits_cond_zeta = self.enc(x)
        if hard:
            dist_cond_pres = torch_dist.bernoulli.Bernoulli(logits=logits_cond_zeta)
            cond_pres = dist_cond_pres.sample()
        else:
            dist_cond_pres = torch_dist.relaxed_bernoulli.RelaxedBernoulli(temp, logits=logits_cond_zeta)
            cond_pres = dist_cond_pres.rsample()
        result = {'cond_pres': cond_pres, 'logits_cond_zeta': logits_cond_zeta}
        return result


class NetworkWhere(nn.Module):

    def __init__(self, config):
        super(NetworkWhere, self).__init__()
        self.register_buffer('prior_mu', torch.tensor(config['prior_where_mu']))
        self.register_buffer('prior_logvar', 2 * torch.tensor(config['prior_where_std']).log())
        self.enc = LinearBlock(
            hidden_list=config['enc_where_hidden'],
            in_features=config['state_size'],
            out_features=8,
        )

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def decode(mu, logvar):
        sample = reparameterize_normal(mu, logvar)
        scl = torch.sigmoid(sample[..., :2])
        trs = torch.tanh(sample[..., 2:])
        return scl, trs, sample

    def forward(self, x):
        mu, logvar = self.encode(x)
        scl, trs, sample = self.decode(mu, logvar)
        kld = compute_kld_normal(mu, logvar, self.prior_mu, self.prior_logvar)
        result = {'scl': scl, 'trs': trs, 'where_sample': sample, 'where_kld': kld}
        return result


class NetworkWhat(nn.Module):

    def __init__(self, config):
        super(NetworkWhat, self).__init__()
        latent_size = config['latent_what_size']
        self.register_buffer('prior_mu', torch.zeros([latent_size]))
        self.register_buffer('prior_logvar', torch.zeros([latent_size]))
        self.enc = EncoderBlock(
            channel_list=config['enc_what_channel'],
            kernel_list=config['enc_what_kernel'],
            stride_list=config['enc_what_stride'],
            hidden_list=config['enc_what_hidden'],
            in_shape=[config['crop_shape'][0] * 2, *config['crop_shape'][1:]],
            out_features=latent_size * 2,
        )
        self.dec = DecoderBlock(
            channel_list_rev=config['dec_what_channel_rev'],
            kernel_list_rev=config['dec_what_kernel_rev'],
            stride_list_rev=config['dec_what_stride_rev'],
            hidden_list_rev=config['dec_what_hidden_rev'],
            in_features=latent_size,
            out_shape=config['crop_shape'],
        )

    def encode(self, x, grid_crop):
        x = nn_func.grid_sample(x, grid_crop, align_corners=False)
        x = x * 2 - 1
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, mu, logvar, grid_full):
        sample = reparameterize_normal(mu, logvar)
        x = self.dec(sample)
        x = nn_func.grid_sample(x, grid_full, align_corners=False)
        return x, sample

    def forward(self, x, grid_crop, grid_full):
        mu, logvar = self.encode(x, grid_crop)
        what, sample = self.decode(mu, logvar, grid_full)
        kld = compute_kld_normal(mu, logvar, self.prior_mu, self.prior_logvar)
        result = {'what': what, 'what_sample': sample, 'what_kld': kld}
        return result
