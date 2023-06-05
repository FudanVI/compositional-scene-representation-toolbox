import math
import pdb

import einops as E
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.GP import DGP
from modules.building_block import LinearLayer
from modules.networks import ViewEncoder, Decoder
from modules.utils import select_by_index, compute_kld_normal


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.frequency = config['frequency']
        # self.cond_net = CondEmbedding(config)
        self.normal_invvar = 1 / pow(config['normal_scale'], 2)
        self.normal_const = math.log(2 * math.pi / self.normal_invvar)
        self.prior_alpha = config['prior_alpha']
        self.register_buffer("prior_attr_obj_mu", torch.zeros([config['latent_attr_obj_size']]))
        self.register_buffer("prior_attr_obj_logvar", torch.zeros([config['latent_attr_obj_size']]))
        self.register_buffer("prior_attr_bck_mu", torch.zeros([config['latent_attr_bck_size']]))
        self.register_buffer("prior_attr_bck_logvar", torch.zeros([config['latent_attr_bck_size']]))
        self.register_buffer("prior_lambda_logvar", torch.log(
            config['lambda_sigma'] ** 2 * torch.ones([config['latent_view_size'] * config['rule_dim']])))
        self.register_buffer("prior_view_mu", torch.zeros([config['latent_view_size']]))
        self.register_buffer("prior_view_logvar", torch.zeros([config['latent_view_size']]))
        self.latent_view_size = config['latent_view_size']
        self.seg_overlap = config['seg_overlap']
        self.mean_nll = config['mean_nll']
        embed_fns = []
        periodic_fns = [torch.cos, torch.sin]
        period_dim = 0
        freq_bands = 2. ** torch.arange(0, self.frequency)
        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
            period_dim += len(periodic_fns)
        self.embed_fns = embed_fns
        self.period_dim = period_dim
        self.info_dim = 0
        self.use_info = config['use_info']
        if self.use_info:
            self.info_dim = config['info_dim']
        self.view_encoder = ViewEncoder(config, self.period_dim + self.info_dim)
        self.gp = DGP(config)
        self.decoder = Decoder(config)
        linear = nn.ModuleList()
        for i in range(config['latent_view_size']):
            linear.append(LinearLayer(
                in_features=period_dim,
                out_features=config['rule_dim'],
                activation=None,
                bias=False
            ))
        self.linear = linear

    def forward(self, data, phase_param, loss_coef, single_view=False, y=None, require_results=True, test_data=False,
                test_latent=False, stage='one', determine_data=False, continuous=False, **kwargs):
        num_views = 1 if single_view else phase_param['num_views']
        observed_views = 1 if single_view else phase_param['observed_views']
        num_slots = phase_param['num_slots']
        temp_pres = loss_coef['temp_pres']
        temp_ord = loss_coef['temp_ord']
        noise_scale_1 = loss_coef['noise_scale_1']
        noise_scale_2 = loss_coef['noise_scale_2']
        data = self.convert_data(data, num_views, y, stage, determine_data=determine_data, test_data=test_data)
        data = self.sample_view_config(data, num_views, observed_views, test_data, **kwargs)
        t_timestep = data['T']['timestep']
        if continuous:
            t_timestep = torch.rand((t_timestep.shape[0], 1), device=t_timestep.device) + t_timestep
        t_embedding = torch.stack([fn(t_timestep) for fn in self.embed_fns],
                                  dim=-1)
        flag = (stage == 'one' or single_view)
        if flag:
            q_embedding = None
        else:
            q_timestep = data['Q']['timestep']
            q_timestep = torch.rand((q_timestep.shape[0], 1), device=q_timestep.device) + q_timestep
            q_embedding = torch.stack([fn(q_timestep) for fn in self.embed_fns],
                                      dim=-1)
        if y is not None:
            t_info = torch.cat([data['T'][key] for key in y])
            t_embedding = torch.cat((t_embedding, t_info), dim=-1)
            q_info = torch.cat([data['Q'][key] for key in y])
            q_embedding = torch.cat((q_embedding, q_info), dim=-1)
        output = self.view_encoder(data['T']['image'], (t_embedding, q_embedding), num_slots,
                                   timestep=data['T']['timestep'] if flag else None, stage='one' if flag else 'two',
                                   test_latent=test_latent)
        if not flag:
            output['Q'].update(
                self.gp(output['T']['view_latent'], output['T']['lambda'], output['Q']['lambda'], test_data=test_data))

        dec_output1, dec_output2 = self.decoder(output, stage='one' if flag else 'two',
                                                temp_pres=temp_pres, temp_ord=temp_ord,
                                                noise_scale_1=noise_scale_1, noise_scale_2=noise_scale_2)
        output['T'].update({key: val[:, :observed_views] for key, val in dec_output1.items()})
        output['T'].update({key: val for key, val in dec_output2.items()})
        output['T'].update({'embedding': t_embedding, 'image': data['T']['image']})
        if not flag:
            output['Q'].update({key: val[:, observed_views:] for key, val in dec_output1.items()})
            output['Q'].update({'embedding': q_embedding, 'image': data['Q']['image']})

        losses = self.compute_loss(output, data, stage='one' if flag else 'two') if not test_data else {}
        metrics = self.compute_metrics(output, data, stage='one' if flag else 'two')
        results = self.convert_results(output, stage='one' if flag else 'two') if require_results else {}
        if len(results) > 0:
            results['T'].update({'index': data['T']['timestep'].to(torch.uint8)})
            if 'Q' in results:
                results['Q'].update({'index': data['Q']['timestep'].to(torch.uint8)})

        return {'loss': losses, 'metric': metrics, 'result': results}

    @staticmethod
    def convert_data(data, num_views, y, stage, determine_data=False, test_data=False):
        batch_size, data_views = data['image'].shape[:2]
        device = data['image'].device
        if num_views == data_views or test_data:
            index = torch.arange(num_views)[None].expand(batch_size, -1)
        else:
            if not determine_data:
                noise = torch.rand([batch_size, data_views], device=device)
                index = torch.argsort(noise)
                index = index[:, :num_views]
                index, _ = torch.sort(index, dim=1)
            else:
                index_base = torch.empty([data_views - num_views + 1, num_views], dtype=torch.int64)
                for idx in range(data_views - num_views + 1):
                    index_base[idx] = torch.arange(idx, idx + num_views)
                index_base = index_base[None].expand(batch_size, -1, -1)
                start_index = torch.randint(0, data_views - num_views + 1, (batch_size, 1))
                index = select_by_index(index_base, start_index).squeeze(1)
        data = {key: select_by_index(val, index).cuda(non_blocking=True) for key, val in data.items()}
        image = data['image'].float() / 255
        segment_base = data['segment'][:, :, None, None].long()
        scatter_shape = [*segment_base.shape[:2], segment_base.max() + 1, *segment_base.shape[3:]]
        segment = torch.zeros(scatter_shape, device=segment_base.device).scatter_(2, segment_base, 1)
        overlap = torch.gt(data['overlap'][:, :, None, None], 1).float()
        if num_views != data_views and determine_data:
            index = torch.arange(num_views)[None].expand(batch_size, -1)
        dict = {'image': image, 'segment': segment, 'overlap': overlap, 'timestep': index.to(image)}
        if y is not None:
            label = {key: select_by_index(val, index).cuda(non_blocking=True) for key, val in data.items() if
                     key in y}
            dict.update(label)

        return dict

    def sample_view_config(self, data, num_views, observed_views, test_data, **kwargs):
        device = data['image'].device
        batch_size = data['image'].shape[0]
        if len(kwargs) > 0:
            t_index = torch.tensor(kwargs['t_index'], dtype=torch.int64).to(device)[None].expand(batch_size, -1)
            q_index = torch.tensor(kwargs['q_index'], dtype=torch.int64).to(device)[None].expand(batch_size, -1)
            dict = {'T': {key: select_by_index(val, t_index) for key, val in data.items()},
                    'Q': {key: select_by_index(val, q_index) for key, val in data.items()}}
        elif observed_views == num_views:
            dict = {'T': {key: val for key, val in data.items()}}
        else:
            noise = torch.rand([batch_size, num_views], device=device)
            index = torch.argsort(noise)
            t_index = index[:, :observed_views]
            q_index = index[:, observed_views:]
            t_index, _ = torch.sort(t_index, dim=1)
            q_index, _ = torch.sort(q_index, dim=1)
            dict = {'T': {key: select_by_index(val, t_index) for key, val in data.items()},
                    'Q': {key: select_by_index(val, q_index) for key, val in data.items()}}
        return dict

    @staticmethod
    def convert_results(results, stage):
        variable_share_key_list = ['attr_obj_latent', 'attr_bck_latent', 'logits_pres', 'trs']
        share_key_list = ['pres']
        disc_key_list = ['image', 'recon', 'mask', 'apc', 'shp', 'noisy_apc', 'noisy_recon']
        view_key_list = ['view_latent', 'lambda', 'matrix']
        ord_key_list = ['ord', 'log_ord']
        output = {'T': {}}
        if stage == 'two':
            output.update({'Q': {}})
        for mode in output.keys():
            output[mode].update({key: results['T'][key] for key in share_key_list})
            output[mode].update({key: results[mode][key] for key in disc_key_list})
            output[mode].update({key: results['T'][key] for key in variable_share_key_list})
            output[mode].update({key: results[mode][key] for key in view_key_list if key in results[mode]})
            output[mode].update({key: results[mode][key] for key in ord_key_list})

        pixel_key_list = share_key_list + disc_key_list
        for mode in output.keys():
            for key, val in output[mode].items():
                if key in pixel_key_list:
                    output[mode][key] = (val.clamp(0, 1) * 255).to(torch.uint8)

        return output

    def compute_loss(self, results, data, stage):
        t_image = data['T']['image']
        observed_nums = t_image.shape[1]
        t_bck = results['T']['apc'][:, :, -1]
        t_recon = results['T']['noisy_recon']
        if stage == 'two':
            q_recon = results['Q']['noisy_recon']
            q_image = data['Q']['image']
            query_nums = q_image.shape[1]
        t_nll = 0.5 * self.normal_invvar * (t_recon - t_image).square().sum([*range(1, t_image.ndim)])
        if self.mean_nll:
            t_nll = t_nll / observed_nums
        num_slots = results['T']['apc'].shape[2]
        kl_pres = self.compute_kld_pres(results['T']['tau1'], results['T']['tau2'], results['T']['logits_zeta'],
                                        num_slots)
        kl_bck = compute_kld_normal(results['T']['attr_bck_mu'], results['T']['attr_bck_logvar'],
                                    self.prior_attr_bck_mu, self.prior_attr_bck_logvar).sum(-1)
        kl_obj = compute_kld_normal(results['T']['attr_obj_mu'], results['T']['attr_obj_logvar'],
                                    self.prior_attr_obj_mu, self.prior_attr_obj_logvar).sum(-1)
        if stage == 'two':
            prior_view_logvar = torch.log(E.rearrange(results['Q']['diag'], "b d t -> b t d"))
            t_prior_view_logvar, q_prior_view_logvar = prior_view_logvar[:, :observed_nums], prior_view_logvar[:,
                                                                                             observed_nums:]
        else:
            t_prior_view_logvar = self.prior_view_logvar

        # compute view attributes losses
        t_kl_view = compute_kld_normal(results['T']['view_mu'], results['T']['view_logvar'], self.prior_view_mu,
                                       t_prior_view_logvar).sum(-1)
        loss_reg_bck = 0.5 * self.normal_invvar * (t_bck - t_image).square().sum([*range(1, t_image.ndim)])

        if stage == 'two':

            q_nll = 0.5 * self.normal_invvar * (q_recon - q_image).square().sum([*range(1, q_image.ndim)])
            if self.mean_nll:
                q_nll = q_nll / query_nums
            q_kl_view = compute_kld_normal(results['Q']['view_mu'], results['Q']['view_logvar'], self.prior_view_mu,
                                           q_prior_view_logvar).sum(-1)

            prior_lambda_mu_list = []
            embedding = torch.cat((results['T']['embedding'], results['Q']['embedding']), dim=1)
            for i in range(self.latent_view_size):
                prior_lambda_mu_list.append(self.linear[i](embedding))
            t_prior_lambda_mu, q_prior_lambda_mu = torch.stack(prior_lambda_mu_list, dim=2).split(
                [observed_nums, query_nums], dim=1)
            t_prior_lambda_mu, q_prior_lambda_mu = E.rearrange(t_prior_lambda_mu, "b t d r -> b t (d r)"), E.rearrange(
                q_prior_lambda_mu, "b t d r-> b t (d r)")
            t_lambda_mu, q_lambda_mu = E.rearrange(results['T']['lambda_mu'], "b t d r -> b t (d r)"), E.rearrange(
                results['Q']['lambda_mu'], "b t d r -> b t (d r)")
            t_kl_lambda = compute_kld_normal(t_lambda_mu, self.prior_lambda_logvar, t_prior_lambda_mu,
                                             self.prior_lambda_logvar).sum(-1)
            q_kl_lambda = compute_kld_normal(q_lambda_mu, self.prior_lambda_logvar, q_prior_lambda_mu,
                                             self.prior_lambda_logvar).sum(-1)
            loss = {'T': {'nll': t_nll, 'kld_pres': kl_pres, 'kld_attr_bck': kl_bck, 'kld_attr_obj': kl_obj,
                          'kld_view': t_kl_view,
                          'kld_lam': t_kl_lambda, 'reg_bck': loss_reg_bck},
                    'Q': {'nll': q_nll, 'kld_view': q_kl_view, 'kld_lam': q_kl_lambda}}
        else:
            loss = {'T': {'nll': t_nll, 'kld_pres': kl_pres, 'kld_attr_bck': kl_bck, 'kld_attr_obj': kl_obj,
                          'kld_view': t_kl_view, 'reg_bck': loss_reg_bck}}
        return loss

    @staticmethod
    def compute_ari(mask_true, mask_pred):
        def comb2(x):
            x = x * (x - 1)
            if x.ndim > 1:
                x = x.sum([*range(1, x.ndim)])
            return x

        mask_true = mask_true.reshape(*mask_true.shape[:2], -1)
        mask_pred = mask_pred.reshape(*mask_pred.shape[:2], -1)
        num_pixels = mask_true.sum([*range(1, mask_true.ndim)])
        mask_true = mask_true.reshape([mask_true.shape[0], mask_true.shape[1], 1, mask_true.shape[-1]])
        mask_pred = mask_pred.reshape([mask_pred.shape[0], 1, mask_pred.shape[1], mask_pred.shape[-1]])
        mat = (mask_true * mask_pred).sum(-1)
        sum_row = mat.sum(1)
        sum_col = mat.sum(2)
        comb_mat = comb2(mat)
        comb_row = comb2(sum_row)
        comb_col = comb2(sum_col)
        comb_num = comb2(num_pixels)
        comb_prod = (comb_row * comb_col) / comb_num
        comb_mean = 0.5 * (comb_row + comb_col)
        diff = comb_mean - comb_prod
        score = (comb_mat - comb_prod) / diff
        invalid = ((comb_num == 0) + (diff == 0)) > 0
        score = torch.where(invalid, torch.ones_like(score), score)
        return score

    def compute_metrics(self, results, data, stage):
        def compute_ari_values(mask_true, mask):
            mask_true_s = mask_true.reshape(-1, *mask_true.shape[2:])
            mask_true_m = mask_true.transpose(1, 2).contiguous()
            mask_oh = torch.argmax(mask, dim=2, keepdim=True)
            mask_oh = torch.zeros_like(mask).scatter_(2, mask_oh, 1)
            mask_oh_s = mask_oh.reshape(-1, *mask_oh.shape[2:])
            mask_oh_m = mask_oh.transpose(1, 2).contiguous()
            ari_all_s = self.compute_ari(mask_true_s, mask_oh_s).reshape(batch_size, num_views).mean(1)
            ari_all_m = self.compute_ari(mask_true_m, mask_oh_m)
            return ari_all_s, ari_all_m

        output = {'T': {}}
        if stage == 'two':
            output.update({'Q': {}})
        count = 0
        for mode in output.keys():
            image, segment, overlap = data[mode]['image'], data[mode]['segment'], data[mode]['overlap']
            batch_size, num_views = image.shape[:2]
            segment_obj = segment[:, :, :-1]
            # ARI
            segment_all_sel = segment if self.seg_overlap else segment * (1 - overlap)
            segment_obj_sel = segment_obj if self.seg_overlap else segment_obj * (1 - overlap)
            mask = results[mode]['mask']
            mask_obj = mask[:, :, :-1]
            ari_all_s, ari_all_m = compute_ari_values(segment_all_sel, mask)
            ari_obj_s, ari_obj_m = compute_ari_values(segment_obj_sel, mask_obj)
            # MSE
            recon = results[mode]['recon']
            mse = (recon - image).square().mean([*range(1, image.ndim)])

            # Count
            if mode == 'T':
                count_true = segment_obj.reshape(*segment_obj.shape[:-3], -1).max(-1).values.max(-2).values.sum(-1)
                count_pred = results['T']['pres'][:, :-1].sum(1)
                count_acc = torch.eq(count_true, count_pred).to(dtype=torch.float)
                count = count_acc

            metrics = {
                'ari_all_s': ari_all_s, 'ari_all_m': ari_all_m,
                'ari_obj_s': ari_obj_s, 'ari_obj_m': ari_obj_m,
                'mse': mse, 'count': count
            }
            output[mode].update(metrics)

        return output

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
        log_zeta = F.logsigmoid(logits_zeta)
        log1m_zeta = log_zeta - logits_zeta
        kld_3 = zeta * (log_zeta - psi1) + (1 - zeta) * (log1m_zeta - psi2) + psi12
        kld = kld_1 + kld_2 + kld_3
        return kld.sum(-1)


def get_model(config):
    net = Model(config).cuda()
    return net
