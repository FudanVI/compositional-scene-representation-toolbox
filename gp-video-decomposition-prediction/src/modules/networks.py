import math
import pdb

import einops as E
import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F

from .building_block import LinearBlock, ConvBlock, DynamicPositionalEmbedding, EncoderPos, LinearLayer, GRULayer, \
    get_decoder, get_grid
from .utils import compute_variable_full, reparameterize_normal


class CondEmbedding(nn.Module):
    def __init__(self, config):
        super(CondEmbedding, self).__init__()
        self.t_dim = config['t_dim']
        self.use_info = config['use_info']
        self.rule_dim = config['rule_dim']
        if self.use_info:
            assert 'info_dim' in config, 'info_dim is not found, error!'
            self.info_dim = config['info_dim']
        else:
            self.info_dim = 0
        full_dim = self.t_dim + self.info_dim
        self.net = LinearBlock(
            in_features=full_dim,
            feature_list=[self.rule_dim],
            act_inner=config['cond_act'],
            act_out=None,
            bias=False
        )

    def forward(self, timestep, y=None):
        embedding = timestep
        if y is not None:
            assert self.use_info is not None
            embedding = torch.cat((embedding, y), dim=-1)
        return self.net(embedding)


class TransformerBlock(nn.Module):
    def __init__(self, config, conv_dim):
        super(TransformerBlock, self).__init__()
        self.nhead = config['nhead']
        self.attention = nn.MultiheadAttention(conv_dim, config['nhead'])
        self.layer_norm_in = nn.LayerNorm(conv_dim)
        self.linear = LinearBlock(
            in_features=conv_dim,
            feature_list=config['trans_feat_list'] + [conv_dim],
            act_inner=config['trans_feat_act'],
            act_out=None,
            bias=False
        )
        self.layer_norm_out = nn.LayerNorm(conv_dim)

    def forward(self, x):
        x = E.rearrange(x, "b l c -> l b c")
        x_res, _ = self.attention(x, x, x)
        x = x + x_res
        x = E.rearrange(x, "l b c -> b l c")
        x = self.layer_norm_in(x)
        x_res = self.linear(x)
        x = self.layer_norm_out(x + x_res)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config, conv_dim):
        super(TransformerEncoder, self).__init__()
        self.nlayer = config['nlayer']
        transformer = nn.ModuleList()
        for i in range(self.nlayer):
            transformer.append(TransformerBlock(config, conv_dim))
        self.transformer = transformer

    def forward(self, x):
        for i in range(self.nlayer):
            x = self.transformer[i](x)
        return x


class ViewEncoder(nn.Module):
    def __init__(self, config, cond_dim):
        super(ViewEncoder, self).__init__()
        self.pos_concat = config['pos_concat']
        self.dynamic_timestep = config['dynamic_timestep']
        self.latent_view_size = config['latent_view_size']
        self.rule_dim = config['rule_dim']
        self.lambda_param = config['lambda_param']
        self.lambda_dim = config['latent_view_size'] * config['rule_dim'] * config['lambda_param']
        self.slot_view_size = config['slot_view_size']
        self.slot_attr_size = config['slot_attr_size']
        self.use_view_slot = config['use_view_slot']
        enc_feat_act = config['enc_feat_act']
        enc_feat_norm = config['enc_feat_norm']
        if config['lambda_param'] == 1:
            self.register_buffer("lambda_sigma",
                                 config['lambda_sigma'] * torch.ones([config['latent_view_size'], config['rule_dim']]))
        self.net_conv = ConvBlock(
            in_shape=config['image_shape'],
            channel_list=config['enc_channel_list'],
            kernel_list=config['enc_kernel_list'],
            stride_list=config['enc_stride_list'],
            act_inner=enc_feat_act,
            norm_inner=enc_feat_norm,
            act_out=None
        )
        conv_channel = self.net_conv.out_shape[0]
        self.pos_encoder = DynamicPositionalEmbedding(conv_channel, self.pos_concat, self.dynamic_timestep)
        self.transformer_in = TransformerEncoder(config, conv_channel)
        self.transformer_out = TransformerEncoder(config, conv_channel)

        self.lambda_net = LinearBlock(
            in_features=conv_channel + cond_dim,
            feature_list=config['lambda_feat_list'] + [self.lambda_dim],
            act_inner=config['lambda_feat_act'],
            act_out=None
        )
        self.trans_share = config['trans_share']
        if not self.trans_share:
            self.slot_net = EncoderPos(
                in_shape=config['image_shape'],
                channel_list=config['slot_channel_list'],
                kernel_list=config['slot_kernel_list'],
                stride_list=config['slot_stride_list'],
                activation=enc_feat_act,
                norm=enc_feat_norm
            )
            slot_conv_size = self.slot_net.net_image.out_shape[0]
        else:
            slot_conv_size = conv_channel
        self.layer_norm = nn.LayerNorm(slot_conv_size)
        self.slot_linear = LinearBlock(
            in_features=slot_conv_size,
            feature_list=config['slot_feature_list'],
            act_inner=enc_feat_act,
            act_out=None
        )

        self.slot_attention = SlotAttentionMulti(
            num_steps=config['slot_steps'],
            qry_size=config['slot_qry_size'],
            slot_view_size=config['slot_view_size'],
            slot_attr_size=config['slot_attr_size'],
            in_features=self.slot_linear.out_features,
            feature_res_list=config['enc_slot_feature_res'],
            activation=config['enc_slot_act']
        )

        self.enc_view = LinearBlock(
            in_features=self.slot_view_size,
            feature_list=config['enc_view_feature'] + [self.latent_view_size * 2],
            act_inner=config['enc_view_act'],
            act_out=None
        )
        self.view_linear = LinearLayer(
            in_features=conv_channel,
            out_features=config['slot_view_size'],
            activation=None,
            bias=False
        )

    def forward(self, image, emb, num_slots, timestep=None, stage='one', test_latent=False, output_prev=None):
        x = image
        t_emb, q_emb = emb
        b, t, c, h, w = x.shape
        x = E.rearrange(x, "b t c h w -> (b t) c h w")
        slot_input = x
        z = self.net_conv(x)
        z = E.rearrange(z, "(b t) c h w -> b t c h w", t=t)
        _, _, z_c, z_h, z_w = z.shape
        z = self.pos_encoder(z, timesteps=timestep)
        z = E.rearrange(z, "b t h w c -> b (t h w) c")
        z = self.transformer_in(z)
        z = E.rearrange(z, "b (t h w) c -> b t h w c", t=t, h=z_h, w=z_w)
        z_flatten = E.rearrange(z, "b t h w c -> b t (h w) c")
        zv_mean = E.reduce(z_flatten, "b t k c -> b t c", "mean")
        slots_view = self.view_linear(zv_mean)
        if self.trans_share:
            slot_input = E.rearrange(z, "b t h w c -> (b t) c h w")

        if stage == 'two' or test_latent:
            z = E.reduce(z, "b t (i h) (j w) c -> b t i j c", "sum", h=2, w=2)
            z = z / 2

            # transformer 2
            z = E.rearrange(z, "b t i j c -> b (t i j) c", t=t)
            z = self.transformer_out(z)
            k = z.shape[1] // t
            z = E.rearrange(z, "b (t k) c -> b t k c", t=t, k=k)

            view_feature = E.reduce(z, "b t k c -> b t c", "mean")

            if t_emb is not None:
                view_feature = torch.cat((view_feature, t_emb), dim=-1)
            if self.lambda_param == 1:
                t_lambda_mu = self.lambda_net(view_feature).view(b, t, self.latent_view_size, -1)
                t_lambda = t_lambda_mu + self.lambda_sigma * torch.randn_like(t_lambda_mu)
            else:
                raise NotImplementedError

            t_emb_T = E.rearrange(t_emb, "b t w -> b w t")
            matrix = torch.einsum("bmt,btn->bmn", t_emb_T, t_emb)
            trans_t_lambda_mu = E.rearrange(t_lambda_mu, "b t v r -> b v t r")
            result = torch.einsum("bwt,bvtr->bvwr", t_emb_T, trans_t_lambda_mu)
            matrix = matrix[:, None].expand(-1, self.latent_view_size, -1, -1)

            optimal_weight = E.rearrange(torch.solve(result, matrix)[0], "b v w r -> b v r w")

            if q_emb is not None:
                q_lambda_mu = torch.einsum("bvrw,btw->btvr", optimal_weight, q_emb)
                if self.lambda_param == 1:
                    q_lambda = q_lambda_mu + self.lambda_sigma * torch.randn_like(q_lambda_mu)
                else:
                    raise NotImplementedError
        # slot attention
        x = slot_input if self.trans_share else self.slot_net(slot_input)
        x = E.rearrange(x, "(b t) c h w -> b t (h w) c", b=b, t=t)
        x = self.layer_norm(x)
        x = self.slot_linear(x)
        kwargs = {}
        if self.use_view_slot:
            kwargs.update({"slots_view": slots_view})
        if output_prev is not None:
            kwargs = {key: val for key, val in output_prev.items() if key in ['slots_view', 'slots_attr']}
        slots_view, slots_attr = self.slot_attention(x, num_slots, **kwargs)
        t_view_mu, t_view_logvar = self.enc_view(slots_view).chunk(2, dim=-1)
        t_view_latent = reparameterize_normal(t_view_mu, t_view_logvar)
        if stage == 'one':
            output = {
                'T': {'view_mu': t_view_mu, 'view_logvar': t_view_logvar,
                      'view_latent': t_view_latent, 'slots_view': slots_view, 'slots_attr': slots_attr}
            }
            if test_latent:
                output['T'].update({'lambda': t_lambda, 'matrix': optimal_weight})
        else:
            output = {
                'T': {'lambda': t_lambda, 'lambda_mu': t_lambda_mu, 'view_mu': t_view_mu, 'view_logvar': t_view_logvar,
                      'view_latent': t_view_latent, 'slots_view': slots_view, 'slots_attr': slots_attr},
                'Q': {'lambda': q_lambda, 'lambda_mu': q_lambda_mu}}
        return output


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
        self.attr_obj_loc = nn.Parameter(torch.zeros([slot_attr_size]), requires_grad=True)
        self.attr_obj_scl = nn.Parameter(torch.zeros([slot_attr_size]), requires_grad=True)
        self.attr_bck_loc = nn.Parameter(torch.zeros([slot_attr_size]), requires_grad=True)
        self.attr_bck_scl = nn.Parameter(torch.zeros([slot_attr_size]), requires_grad=True)
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

    def forward(self, x, num_slots, slots_attr=None, slots_view=None):
        batch_size, num_views = x.shape[:2]
        x = self.layer_norm_in(x)
        x_key = self.net_key(x)
        x_val = self.net_val(x)
        if slots_view is None:
            noise_view = torch.randn([batch_size, num_views, self.slot_view_size], device=x.device)
            slots_view = self.view_loc + torch.exp(self.view_log_scl) * noise_view
            infer_view = True
        else:
            infer_view = False
        if slots_attr is None:
            noise_obj_attr = torch.randn([batch_size, num_slots - 1, self.slot_attr_size], device=x.device)
            noise_bck_attr = torch.randn([batch_size, 1, self.slot_attr_size], device=x.device)
            slots_obj_attr = self.attr_obj_loc + torch.exp(self.attr_obj_scl) * noise_obj_attr
            slots_bck_attr = self.attr_bck_loc + torch.exp(self.attr_bck_scl) * noise_bck_attr
            slots_attr = torch.cat((slots_obj_attr, slots_bck_attr), dim=1)
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
            if infer_view:
                slots_view = slots_view_raw.mean(2)
            if infer_attr:
                slots_attr = slots_attr_raw.mean(1)

        return slots_view, slots_attr


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        image_shape = config['image_shape']
        slot_view_size = config['slot_view_size']
        slot_attr_size = config['slot_attr_size']
        latent_view_size = config['latent_view_size']
        latent_attr_obj_size = config['latent_attr_obj_size']
        latent_attr_bck_size = config['latent_attr_bck_size']
        self.pixel_bound = config['pixel_bound']
        latent_full_obj_size = latent_attr_obj_size + latent_view_size
        latent_full_bck_size = latent_attr_bck_size + latent_view_size
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
            norm=config['dec_obj_norm']
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
            norm=config['dec_bck_norm']
        )
        self.register_buffer('grid', get_grid(image_shape[1:]))

    def forward(self, inputs, stage, temp_pres, temp_ord, noise_scale_1,
                noise_scale_2, output_prev=None):
        t_view_latent = inputs['T']['view_latent']
        if stage == 'two':
            q_view_latent = inputs['Q']['view_latent']
            view_latent = torch.cat((t_view_latent, q_view_latent), dim=1)
        else:
            view_latent = t_view_latent
        slots_attr = inputs['T']['slots_attr']
        slots_obj, slots_bck = slots_attr[:, :-1, :], slots_attr[:, -1:, :]
        attr_obj_mu, attr_obj_logvar, logits_tau1, logits_tau2, logit_zeta = self.enc_attr_obj(slots_obj).split(
            self.split_attr_obj, dim=-1)
        logits_pres = distributions.relaxed_bernoulli.LogitRelaxedBernoulli(temp_pres, logits=logit_zeta).rsample()
        pres = torch.sigmoid(logits_pres)
        zeta = torch.sigmoid(logit_zeta)
        tau1 = F.softplus(logits_tau1)
        tau2 = F.softplus(logits_tau2)
        attr_bck_mu, attr_bck_logvar = self.enc_attr_bck(slots_bck).chunk(2, dim=-1)
        attr_obj_latent = reparameterize_normal(attr_obj_mu, attr_obj_logvar)
        attr_bck_latent = reparameterize_normal(attr_bck_mu, attr_bck_logvar)
        outputs = self.decode(view_latent, attr_obj_latent, attr_bck_latent, pres, logits_pres, temp_ord, noise_scale_1,
                              noise_scale_2)
        pres = torch.ge(pres.squeeze(-1), 0.5).float()
        pres_all = torch.cat([pres, torch.ones([pres.shape[0], 1]).to(pres)], dim=1)
        attribute_dict = {
            'pres': pres_all, 'logits_pres': logits_pres, 'zeta': zeta, 'logits_zeta': logit_zeta,
            'attr_obj_latent': attr_obj_latent, 'attr_bck_latent': attr_bck_latent, 'attr_obj_mu': attr_obj_mu,
            'attr_obj_logvar': attr_obj_logvar, 'attr_bck_mu': attr_bck_mu, 'attr_bck_logvar': attr_bck_logvar,
            'tau1': tau1, 'tau2': tau2
        }
        return outputs, attribute_dict

    def decode(self, view_latent, attr_obj_latent, attr_bck_latent, pres, logits_pres, temp_ord, noise_scale_1,
               noise_scale_2):

        full_obj_latent = compute_variable_full(view_latent, attr_obj_latent)
        full_bck_latent = compute_variable_full(view_latent, attr_bck_latent)
        batch_size, num_views, obj_slots = full_obj_latent.shape[:3]
        log_ord, logits_trs = self.dec_ord(full_obj_latent).split([1, 2], dim=-1)
        log_ord = log_ord / temp_ord
        order = torch.exp(log_ord)
        trs = torch.tanh(logits_trs)
        grid = self.grid[None, None].expand(*trs.shape[:3], -1, -1, -1)
        sq_diff = (grid - trs[..., None, None]).square().sum(-3, keepdims=True)
        noise_coef = noise_scale_2 * (1 - torch.exp(-noise_scale_1 * sq_diff))
        full_obj_latent_reshape = full_obj_latent.reshape(-1, *full_obj_latent.shape[3:])
        full_bck_latent_reshape = full_bck_latent.reshape(-1, *full_bck_latent.shape[3:])
        x_obj = self.dec_obj(full_obj_latent_reshape)
        x_obj = x_obj.reshape(batch_size, num_views, obj_slots, *x_obj.shape[1:])
        apc = x_obj[..., :-1, :, :]
        noisy_apc = apc + noise_coef * torch.randn_like(apc)
        logits_shp = x_obj[..., -1:, :, :]
        shp = torch.sigmoid(logits_shp)
        x_bck = self.dec_bck(full_bck_latent_reshape)
        bck = x_bck.reshape(batch_size, num_views, 1, *x_bck.shape[1:])
        if self.pixel_bound:
            apc = torch.sigmoid(apc)
            bck = torch.sigmoid(bck)
        mask = self.compute_mask(shp, logits_shp, pres, logits_pres, log_ord)
        apc_all = torch.cat((apc, bck), dim=2)
        recon = (mask * apc_all).sum(2)
        noisy_apc_all = torch.cat([noisy_apc, bck], dim=2)
        noisy_recon = (mask * noisy_apc_all).sum(2)
        ones = torch.ones(*shp.shape[:2], 1, *shp.shape[3:], device=shp.device)
        shp_all = torch.cat([shp, ones], dim=2)
        output = {
            'recon': recon, 'noisy_recon': noisy_recon, 'mask': mask, 'apc': apc_all, 'noisy_apc': noisy_apc_all,
            'shp': shp_all, 'ord': order.squeeze(-1), 'log_ord': log_ord.squeeze(-1), 'trs': trs
        }
        return output

    @staticmethod
    def compute_mask(shp, logits_shp, pres, logits_pres, log_ord):
        pres = pres[:, None, ..., None, None]
        logits_pres = logits_pres[:, None, ..., None, None]
        log_ord = log_ord[..., None, None]
        mask_bck = (1 - shp * pres).prod(-4, keepdim=True)
        logits_mask_obj_rel = F.logsigmoid(logits_shp) + F.logsigmoid(logits_pres) + log_ord
        mask_obj_rel = torch.softmax(logits_mask_obj_rel, dim=-4)
        mask_obj = (1 - mask_bck) * mask_obj_rel
        mask = torch.cat([mask_obj, mask_bck], dim=-4)
        return mask
