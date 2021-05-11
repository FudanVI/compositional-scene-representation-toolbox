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


class InitializerBack(EncoderLSTM):

    def __init__(self, config):
        super(InitializerBack, self).__init__(
            channel_list=config['init_back_channel'],
            kernel_list=config['init_back_kernel'],
            stride_list=config['init_back_stride'],
            hidden_list=config['init_back_hidden'],
            in_shape=config['image_shape'],
            state_size=config['state_back_size'],
        )


class InitializerFull(nn.Module):

    def __init__(self, config):
        super(InitializerFull, self).__init__()
        self.upd = EncoderLSTM(
            channel_list=config['init_main_channel'],
            kernel_list=config['init_main_kernel'],
            stride_list=config['init_main_stride'],
            hidden_list=config['init_main_hidden'],
            in_shape=[config['image_shape'][0] * 2 + 1, *config['image_shape'][1:]],
            state_size=config['init_main_state'],
        )
        self.enc = LinearBlock(
            hidden_list=config['init_full_hidden'],
            in_features=config['init_main_state'],
            out_features=None,
        )
        self.lstm = nn.LSTMCell(self.enc.out_features, config['state_full_size'])

    def forward(self, inputs, states_main):
        states_main = self.upd(inputs, states_main)
        x = self.enc(states_main[0])
        states_full = self.lstm(x)
        return states_full, states_main


class InitializerCrop(EncoderLSTM):

    def __init__(self, config):
        super(InitializerCrop, self).__init__(
            channel_list=config['init_crop_channel'],
            kernel_list=config['init_crop_kernel'],
            stride_list=config['init_crop_stride'],
            hidden_list=config['init_crop_hidden'],
            in_shape=[config['crop_shape'][0] * 2 + 1, *config['crop_shape'][1:]],
            state_size=config['state_crop_size'],
        )


class UpdaterBack(EncoderLSTM):

    def __init__(self, config):
        super(UpdaterBack, self).__init__(
            channel_list=config['upd_back_channel'],
            kernel_list=config['upd_back_kernel'],
            stride_list=config['upd_back_stride'],
            hidden_list=config['upd_back_hidden'],
            in_shape=[config['image_shape'][0] * 3 + 1, *config['image_shape'][1:]],
            state_size=config['state_back_size'],
        )


class UpdaterFull(EncoderLSTM):

    def __init__(self, config):
        super(UpdaterFull, self).__init__(
            channel_list=config['upd_full_channel'],
            kernel_list=config['upd_full_kernel'],
            stride_list=config['upd_full_stride'],
            hidden_list=config['upd_full_hidden'],
            in_shape=[config['image_shape'][0] * 3 + 3, *config['image_shape'][1:]],
            state_size=config['state_full_size'],
        )


class UpdaterCrop(EncoderLSTM):

    def __init__(self, config):
        super(UpdaterCrop, self).__init__(
            channel_list=config['upd_crop_channel'],
            kernel_list=config['upd_crop_kernel'],
            stride_list=config['upd_crop_stride'],
            hidden_list=config['upd_crop_hidden'],
            in_shape=[config['crop_shape'][0] * 3 + 3, *config['crop_shape'][1:]],
            state_size=config['state_crop_size'],
        )


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
        latent_size = config['latent_bck_size']
        self.register_buffer('prior_mu', torch.zeros([latent_size]))
        self.register_buffer('prior_logvar', torch.zeros([latent_size]))
        self.enc = LinearBlock(
            hidden_list=config['enc_bck_hidden'],
            in_features=config['state_back_size'],
            out_features=latent_size * 2,
        )
        self.dec = DecoderResidual(
            avg_hidden_list_rev=config['dec_bck_avg_hidden_rev'],
            res_channel_list_rev=config['dec_bck_res_channel_rev'],
            res_kernel_list_rev=config['dec_bck_res_kernel_rev'],
            res_stride_list_rev=config['dec_bck_res_stride_rev'],
            res_hidden_list_rev=config['dec_bck_res_hidden_rev'],
            in_features=latent_size,
            image_shape=config['image_shape'],
        )

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, mu, logvar):
        sample = reparameterize_normal(mu, logvar)
        bck = self.dec(sample)
        return bck

    def forward(self, x):
        mu, logvar = self.encode(x)
        bck = self.decode(mu, logvar)
        kld = compute_kld_normal(mu, logvar, self.prior_mu, self.prior_logvar)
        result = {'bck': bck, 'bck_kld': kld}
        return result


class NetworkFull(nn.Module):

    def __init__(self, config):
        super(NetworkFull, self).__init__()
        self.register_buffer('prior_mu', torch.tensor(config['prior_stn_mu']))
        self.register_buffer('prior_logvar', 2 * torch.tensor(config['prior_stn_std']).log())
        self.enc = LinearBlock(
            hidden_list=config['enc_stn_hidden'],
            in_features=config['state_full_size'],
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
        return scl, trs

    def forward(self, x):
        mu, logvar = self.encode(x)
        scl, trs = self.decode(mu, logvar)
        kld = compute_kld_normal(mu, logvar, self.prior_mu, self.prior_logvar)
        result = {'scl': scl, 'trs': trs, 'stn_kld': kld}
        return result


class NetworkCrop(nn.Module):

    def __init__(self, config):
        super(NetworkCrop, self).__init__()
        latent_size = config['latent_obj_size']
        self.register_buffer('prior_mu', torch.zeros([latent_size]))
        self.register_buffer('prior_logvar', torch.zeros([latent_size]))
        self.enc_pres = LinearBlock(
            hidden_list=config['enc_pres_hidden'],
            in_features=config['state_full_size'] + config['state_crop_size'],
            out_features=3,
        )
        self.enc_obj = LinearBlock(
            hidden_list=config['enc_obj_hidden'],
            in_features=config['state_crop_size'],
            out_features=latent_size * 2,
        )
        self.dec_apc = DecoderResidual(
            avg_hidden_list_rev=config['dec_apc_avg_hidden_rev'],
            res_channel_list_rev=config['dec_apc_res_channel_rev'],
            res_kernel_list_rev=config['dec_apc_res_kernel_rev'],
            res_stride_list_rev=config['dec_apc_res_stride_rev'],
            res_hidden_list_rev=config['dec_apc_res_hidden_rev'],
            in_features=latent_size,
            image_shape=config['crop_shape'],
        )
        self.dec_shp = DecoderBlock(
            channel_list_rev=config['dec_shp_channel_rev'],
            kernel_list_rev=config['dec_shp_kernel_rev'],
            stride_list_rev=config['dec_shp_stride_rev'],
            hidden_list_rev=config['dec_shp_hidden_rev'],
            in_features=latent_size,
            out_shape=[1, *config['crop_shape'][1:]],
        )

    def encode(self, x_full, x_crop):
        x_both = torch.cat([x_full, x_crop], dim=-1)
        logits_tau1, logits_tau2, logits_zeta = self.enc_pres(x_both).chunk(3, dim=-1)
        tau1 = nn_func.softplus(logits_tau1)
        tau2 = nn_func.softplus(logits_tau2)
        mu, logvar = self.enc_obj(x_crop).chunk(2, dim=-1)
        return tau1, tau2, logits_zeta, mu, logvar

    def decode(self, logits_zeta, mu, logvar, grid_full, temp, hard):
        if hard:
            dist_pres = torch_dist.bernoulli.Bernoulli(logits=logits_zeta)
            pres = dist_pres.sample()
        else:
            dist_pres = torch_dist.relaxed_bernoulli.RelaxedBernoulli(temp, logits=logits_zeta)
            pres = dist_pres.rsample()
        sample = reparameterize_normal(mu, logvar)
        apc_crop = self.dec_apc(sample)
        logits_shp_crop = self.dec_shp(sample)
        shp_crop = torch.sigmoid(logits_shp_crop)
        apc = nn_func.grid_sample(apc_crop, grid_full, align_corners=False)
        shp = nn_func.grid_sample(shp_crop, grid_full, align_corners=False)
        return pres, apc, shp, apc_crop, shp_crop

    def forward(self, x_full, x_crop, grid_full, temp, hard):
        tau1, tau2, logits_zeta, mu, logvar = self.encode(x_full, x_crop)
        pres, apc, shp, apc_crop, shp_crop = self.decode(logits_zeta, mu, logvar, grid_full, temp, hard)
        kld = compute_kld_normal(mu, logvar, self.prior_mu, self.prior_logvar)
        result = {
            'pres': pres, 'apc': apc, 'shp': shp, 'apc_crop': apc_crop, 'shp_crop': shp_crop,
            'tau1': tau1, 'tau2': tau2, 'logits_zeta': logits_zeta, 'obj_kld': kld,
        }
        return result
