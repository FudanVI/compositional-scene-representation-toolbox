import math
import numpy as np
import torch
import torch.distributions as torch_dist
import torch.nn as nn
import torch.nn.functional as nn_func
from building_block import get_grid, get_decoder, compute_variable_full, LinearBlock, ConvBlock, EncoderPos, \
    SlotAttentionMulti


def select_by_index(x, index):
    x_ndim = x.ndim
    index_ndim = index.ndim
    index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
    index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
    x = torch.gather(x, index_ndim - 1, index)
    return x


def reparameterize_normal(mu, logvar):
    std = torch.exp(0.5 * logvar)
    noise = torch.randn_like(std)
    return mu + std * noise


def compute_kld_normal(mu, logvar, prior_mu, prior_logvar):
    prior_invvar = torch.exp(-prior_logvar)
    kld = 0.5 * (prior_logvar - logvar + prior_invvar * ((mu - prior_mu).square() + logvar.exp()) - 1)
    return kld.sum(-1)


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        enc_feat_act = config['enc_feat_act']
        self.conv = EncoderPos(
            in_shape=config['image_shape'],
            channel_list=config['enc_feat_channel'],
            kernel_list=config['enc_feat_kernel'],
            stride_list=config['enc_feat_stride'],
            activation=enc_feat_act,
        )
        conv_size = self.conv.net_image.out_shape[0]
        self.layer_norm = nn.LayerNorm(conv_size)
        self.linear = LinearBlock(
            in_features=conv_size,
            feature_list=config['enc_feat_feature'],
            act_inner=enc_feat_act,
            act_out=None,
        )
        self.slot_attn = SlotAttentionMulti(
            num_steps=config['slot_steps'],
            qry_size=config['slot_qry_size'],
            slot_view_size=config['slot_view_size'],
            slot_attr_size=config['slot_attr_size'],
            in_features=self.linear.out_features,
            feature_res_list=config['enc_slot_feature_res'],
            activation=config['enc_slot_act'],
        )

    def forward(self, x, num_slots, outputs_prev=None):
        batch_size, num_views = x.shape[:2]
        x = x.reshape(-1, *x.shape[2:])
        x = x * 2 - 1
        x = self.conv(x)
        x = x.reshape(batch_size, num_views, *x.shape[1:-2], -1).transpose(-1, -2).contiguous()
        x = self.layer_norm(x)
        x = self.linear(x)
        outputs = self.slot_attn(x, num_slots, None if outputs_prev is None else outputs_prev['slots_attr_new'])
        return outputs


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        image_shape = config['image_shape']
        slot_view_size = config['slot_view_size']
        slot_attr_size = config['slot_attr_size']
        latent_view_size = config['latent_view_size']
        latent_attr_obj_size = config['latent_attr_obj_size']
        latent_attr_bck_size = config['latent_attr_bck_size']
        latent_full_obj_size = latent_view_size + latent_attr_obj_size
        latent_full_bck_size = latent_view_size + latent_attr_bck_size
        self.register_buffer('prior_view_mu', torch.zeros([latent_view_size]))
        self.register_buffer('prior_view_logvar', torch.zeros([latent_view_size]))
        self.register_buffer('prior_attr_obj_mu', torch.zeros([latent_attr_obj_size]))
        self.register_buffer('prior_attr_obj_logvar', torch.zeros([latent_attr_obj_size]))
        self.register_buffer('prior_attr_bck_mu', torch.zeros([latent_attr_bck_size]))
        self.register_buffer('prior_attr_bck_logvar', torch.zeros([latent_attr_bck_size]))
        self.prior_alpha = config['pres_alpha']
        self.use_shadow = config['use_shadow']
        self.enc_sel = LinearBlock(
            in_features=slot_attr_size,
            feature_list=config['enc_sel_feature'] + [1],
            act_inner=config['enc_sel_act'],
            act_out=None,
        )
        self.enc_view = LinearBlock(
            in_features=slot_view_size,
            feature_list=config['enc_view_feature'] + [latent_view_size * 2],
            act_inner=config['enc_view_act'],
            act_out=None,
        )
        self.split_attr_obj = [latent_attr_obj_size] * 2 + [1] * 3
        self.enc_attr_obj = LinearBlock(
            in_features=slot_attr_size,
            feature_list=config['enc_attr_obj_feature'] + [np.sum(self.split_attr_obj)],
            act_inner=config['enc_attr_obj_act'],
            act_out=None,
        )
        self.enc_attr_bck = LinearBlock(
            in_features=slot_attr_size,
            feature_list=config['enc_attr_bck_feature'] + [latent_attr_bck_size * 2],
            act_inner=config['enc_attr_bck_act'],
            act_out=None,
        )
        self.dec_ord = LinearBlock(
            in_features=latent_full_obj_size,
            feature_list=config['dec_ord_feature'] + [3],
            act_inner=config['dec_ord_act'],
            act_out=None,
        )
        self.dec_obj = get_decoder(
            in_features=latent_full_obj_size,
            out_shape=[image_shape[0] + 1, *image_shape[1:]],
            channel_list_rev=config['dec_obj_channel_rev'],
            kernel_list_rev=config['dec_obj_kernel_rev'],
            stride_list_rev=config['dec_obj_stride_rev'],
            feature_list_rev=config['dec_obj_feature_rev'],
            activation=config['dec_obj_act'],
            mode=config['dec_obj_mode'],
            spatial_broadcast=config['dec_obj_sbd'],
        )
        self.dec_bck = get_decoder(
            in_features=latent_full_bck_size,
            out_shape=image_shape,
            channel_list_rev=config['dec_bck_channel_rev'],
            kernel_list_rev=config['dec_bck_kernel_rev'],
            stride_list_rev=config['dec_bck_stride_rev'],
            feature_list_rev=config['dec_bck_feature_rev'],
            activation=config['dec_bck_act'],
            mode=config['dec_bck_mode'],
            spatial_broadcast=config['dec_bck_sbd'],
        )
        if self.use_shadow:
            self.sdw_max = config['sdw_max']
            self.dec_sdw = get_decoder(
                in_features=latent_full_obj_size,
                out_shape=[1, *image_shape[1:]],
                channel_list_rev=config['dec_sdw_channel_rev'],
                kernel_list_rev=config['dec_sdw_kernel_rev'],
                stride_list_rev=config['dec_sdw_stride_rev'],
                feature_list_rev=config['dec_sdw_feature_rev'],
                activation=config['dec_sdw_act'],
                mode=config['dec_sdw_mode'],
                spatial_broadcast=config['dec_sdw_sbd'],
            )
        else:
            self.sdw_max = None
            self.dec_sdw = None
        self.register_buffer('grid', get_grid(image_shape[1:]))

    def forward(self, inputs, temp_pres, temp_ord, temp_shp, noise_scale_1, noise_scale_2, outputs_prev=None):
        slots_view = inputs['slots_view']
        slots_attr = inputs['slots_attr']
        batch_size, num_slots = slots_attr.shape[:2]
        view_mu, view_logvar = self.enc_view(slots_view).chunk(2, dim=-1)
        if outputs_prev is None:
            index_base = np.empty([num_slots, num_slots - 1], dtype=np.int)
            for idx in range(num_slots):
                index_base[idx, :idx] = np.arange(0, idx)
                index_base[idx, idx:] = np.arange(idx + 1, num_slots)
            index_base = torch.tensor(index_base, device=slots_attr.device)[None].expand(batch_size, -1, -1)
            logits_sel = self.enc_sel(slots_attr).squeeze(-1)
            sel_log_prob = torch.log_softmax(logits_sel, dim=-1)
            sel_oh = torch_dist.OneHotCategorical(logits=logits_sel).sample()
            sel_index = torch.argmax(sel_oh, dim=-1, keepdim=True)
            sel_log_prob = select_by_index(sel_log_prob, sel_index).squeeze(1)
            obj_index = select_by_index(index_base, sel_index).squeeze(1)
            slots_attr_obj = select_by_index(slots_attr, obj_index)
            slots_attr_bck = select_by_index(slots_attr, sel_index)
            attr_obj_mu, attr_obj_logvar, logits_tau1, logits_tau2, logits_zeta = \
                self.enc_attr_obj(slots_attr_obj).split(self.split_attr_obj, dim=-1)
            slots_attr_new = torch.cat([slots_attr_obj, slots_attr_bck], dim=1)
            logits_pres = torch_dist.relaxed_bernoulli.LogitRelaxedBernoulli(temp_pres, logits=logits_zeta).rsample()
        else:
            sel_log_prob = outputs_prev['sel_log_prob']
            slots_attr_new = outputs_prev['slots_attr_new']
            logits_pres = outputs_prev['logits_pres']
            slots_attr_obj, slots_attr_bck = slots_attr_new[:, :-1], slots_attr_new[:, -1:]
            attr_obj_mu, attr_obj_logvar, logits_tau1, logits_tau2, logits_zeta = \
                self.enc_attr_obj(slots_attr_obj).split(self.split_attr_obj, dim=-1)
        tau1 = nn_func.softplus(logits_tau1)
        tau2 = nn_func.softplus(logits_tau2)
        zeta = torch.sigmoid(logits_zeta)
        pres = torch.sigmoid(logits_pres)
        attr_bck_mu, attr_bck_logvar = self.enc_attr_bck(slots_attr_bck).chunk(2, dim=-1)
        view_latent = reparameterize_normal(view_mu, view_logvar)
        attr_obj_latent = reparameterize_normal(attr_obj_mu, attr_obj_logvar)
        attr_bck_latent = reparameterize_normal(attr_bck_mu, attr_bck_logvar)
        kld_pres = self.compute_kld_pres(tau1, tau2, logits_zeta, num_slots)
        kld_view = compute_kld_normal(view_mu, view_logvar, self.prior_view_mu, self.prior_view_logvar).sum(-1)
        kld_attr_obj = compute_kld_normal(
            attr_obj_mu, attr_obj_logvar, self.prior_attr_obj_mu, self.prior_attr_obj_logvar).sum(-1)
        kld_attr_bck = compute_kld_normal(
            attr_bck_mu, attr_bck_logvar, self.prior_attr_bck_mu, self.prior_attr_bck_logvar).sum(-1)
        outputs = self.decode(view_latent, attr_obj_latent, attr_bck_latent, pres, logits_pres, temp_ord, temp_shp,
                              noise_scale_1, noise_scale_2)
        pres = torch.ge(pres.squeeze(-1), 0.5).to(torch.float)
        pres_all = torch.cat([pres, torch.ones([pres.shape[0], 1], device=pres.device)], dim=1)
        outputs.update({
            'pres': pres_all, 'logits_pres': logits_pres, 'zeta': zeta, 'logits_zeta': logits_zeta,
            'view_latent': view_latent, 'attr_obj_latent': attr_obj_latent, 'attr_bck_latent': attr_bck_latent,
            'kld_pres': kld_pres, 'kld_view': kld_view, 'kld_attr_obj': kld_attr_obj, 'kld_attr_bck': kld_attr_bck,
            'sel_log_prob': sel_log_prob, 'slots_attr_new': slots_attr_new,
        })
        return outputs

    @staticmethod
    def compute_mask(shp, logits_shp, pres, logits_pres, log_ord):
        pres = pres[:, None, ..., None, None]
        logits_pres = logits_pres[:, None, ..., None, None]
        log_ord = log_ord[..., None, None]
        mask_bck = (1 - shp * pres).prod(-4, keepdim=True)
        logits_mask_obj_rel = nn_func.logsigmoid(logits_shp) + nn_func.logsigmoid(logits_pres) + log_ord
        mask_obj_rel = torch.softmax(logits_mask_obj_rel, dim=-4)
        mask_obj = (1 - mask_bck) * mask_obj_rel
        mask = torch.cat([mask_obj, mask_bck], dim=-4)
        return mask

    def compute_kld_pres(self, tau1, tau2, logits_zeta, num_slots):
        tau1 = tau1.squeeze(-1)
        tau2 = tau2.squeeze(-1)
        logits_zeta = logits_zeta.squeeze(-1)
        coef_alpha = self.prior_alpha / (num_slots - 1)
        psi1 = torch.digamma(tau1)
        psi2 = torch.digamma(tau2)
        psi12 = torch.digamma(tau1 + tau2)
        kld_1 = torch.lgamma(tau1 + tau2) - torch.lgamma(tau1) - torch.lgamma(tau2) - math.log(coef_alpha)
        kld_2 = (tau1 - coef_alpha) * psi1 + (tau2 - 1) * psi2 - (tau1 + tau2 - coef_alpha - 1) * psi12
        zeta = torch.sigmoid(logits_zeta)
        log_zeta = nn_func.logsigmoid(logits_zeta)
        log1m_zeta = log_zeta - logits_zeta
        kld_3 = zeta * (log_zeta - psi1) + (1 - zeta) * (log1m_zeta - psi2) + psi12
        kld = kld_1 + kld_2 + kld_3
        return kld.sum(-1)

    def decode(self, view_latent, attr_obj_latent, attr_bck_latent, pres, logits_pres, temp_ord, temp_shp,
               noise_scale_1, noise_scale_2):
        full_obj_latent = compute_variable_full(view_latent, attr_obj_latent)
        full_bck_latent = compute_variable_full(view_latent, attr_bck_latent)
        batch_size, num_views, obj_slots = full_obj_latent.shape[:3]
        log_ord, logits_trs = self.dec_ord(full_obj_latent).split([1, 2], dim=-1)
        log_ord = log_ord / temp_ord
        trs = torch.tanh(logits_trs)
        grid = self.grid[None, None].expand(*trs.shape[:3], -1, -1, -1)
        sq_diff = (grid - trs[..., None, None]).square().sum(-3, keepdims=True)
        noise_coef = noise_scale_2 * (1 - torch.exp(-noise_scale_1 * sq_diff))
        full_obj_latent_reshape = full_obj_latent.reshape(-1, *full_obj_latent.shape[3:])
        full_bck_latent_reshape = full_bck_latent.reshape(-1, *full_bck_latent.shape[3:])
        x_obj = self.dec_obj(full_obj_latent_reshape)
        x_obj = x_obj.reshape(batch_size, num_views, obj_slots, *x_obj.shape[1:])
        apc = (x_obj[..., :-1, :, :] + 1) * 0.5
        noisy_apc = apc + noise_coef * torch.randn_like(apc)
        logits_shp_soft = x_obj[..., -1:, :, :]
        logits_shp = torch_dist.relaxed_bernoulli.LogitRelaxedBernoulli(temp_shp, logits=logits_shp_soft).rsample()
        shp_soft = torch.sigmoid(logits_shp_soft)
        shp = torch.sigmoid(logits_shp)
        x_bck = self.dec_bck(full_bck_latent_reshape)
        x_bck = x_bck.reshape(batch_size, num_views, 1, *x_bck.shape[1:])
        bck = (x_bck + 1) * 0.5
        if self.use_shadow:
            logits_sdw = self.dec_sdw(full_obj_latent_reshape) - 3
            logits_sdw = logits_sdw.reshape(batch_size, num_views, obj_slots, *logits_sdw.shape[1:])
            sdw = torch.sigmoid(logits_sdw / self.sdw_max) * self.sdw_max
            mask_sdw = (1 - sdw * pres.detach()[:, None, :, :, None, None]).prod(2, keepdim=True)
            bck_sdw = bck * mask_sdw
        else:
            sdw = None
            bck_sdw = bck
        mask = self.compute_mask(shp, logits_shp, pres, logits_pres, log_ord)
        mask_soft = self.compute_mask(shp_soft, logits_shp_soft, pres, logits_pres, log_ord)
        apc_all = torch.cat([apc, bck], dim=2)
        noisy_apc_all = torch.cat([noisy_apc, bck], dim=2)
        apc_all_sdw = torch.cat([apc, bck_sdw], dim=2)
        noisy_apc_all_sdw = torch.cat([noisy_apc, bck_sdw], dim=2)
        recon = (mask * apc_all_sdw).sum(2)
        noisy_recon = (mask * noisy_apc_all_sdw).sum(2)
        recon_soft = (mask_soft * apc_all_sdw).sum(2)
        ones = torch.ones(*shp.shape[:2], 1, *shp.shape[3:], device=shp.device)
        shp_all = torch.cat([shp, ones], dim=2)
        shp_soft_all = torch.cat([shp_soft, ones], dim=2)
        outputs = {
            'recon': recon, 'noisy_recon': noisy_recon, 'recon_soft': recon_soft, 'mask': mask, 'mask_soft': mask_soft,
            'apc': apc_all, 'noisy_apc': noisy_apc_all, 'bck_sdw': bck_sdw, 'shp': shp_all, 'shp_soft': shp_soft_all,
            'log_ord': log_ord, 'trs': trs,
        }
        if self.use_shadow:
            zeros = torch.zeros(*sdw.shape[:2], 1, *sdw.shape[3:], device=sdw.device)
            sdw_all = torch.cat([sdw, zeros], dim=2)
            outputs['sdw'] = sdw_all
        return outputs


class NetworkBaseline(nn.Module):

    def __init__(self, config):
        super(NetworkBaseline, self).__init__()
        activation = config['net_bl_act']
        self.conv = ConvBlock(
            in_shape=config['image_shape'],
            channel_list=config['net_bl_channel'],
            kernel_list=config['net_bl_kernel'],
            stride_list=config['net_bl_stride'],
            act_inner=activation,
            act_out=activation,
        )
        self.linear = LinearBlock(
            in_features=np.prod(self.conv.out_shape),
            feature_list=config['net_bl_feature'] + [1],
            act_inner=activation,
            act_out=None,
        )

    def forward(self, x):
        x = x * 2 - 1
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x).squeeze(-1)
        return x
