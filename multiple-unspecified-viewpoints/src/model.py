import math
import torch
import torch.nn as nn
from network import select_by_index, Encoder, Decoder, NetworkBaseline


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # Hyperparameters
        self.normal_invvar = 1 / pow(config['normal_scale'], 2)
        self.normal_const = math.log(2 * math.pi / self.normal_invvar)
        self.register_buffer('score_mean', torch.zeros([]))
        self.register_buffer('score_var', torch.zeros([]))
        self.bl_momentum = config['bl_momentum']
        self.seg_overlap = config['seg_overlap']
        # Neural networks
        self.enc = Encoder(config)
        self.dec = Decoder(config)
        self.net_bl = NetworkBaseline(config)

    def forward(self, data, phase_param, loss_coef, single_view=False, infer_extra=False, require_results=True,
                require_results_gen=True, deterministic_data=False):
        data = {key: val.cuda(non_blocking=True) for key, val in data.items()}
        num_views = 1 if single_view else phase_param['num_views']
        num_slots = phase_param['num_slots']
        outputs, results = self.compute_outputs(
            data, num_views, num_slots, loss_coef, require_results, deterministic_data)
        if infer_extra:
            num_views_ext = 1 if single_view else phase_param['num_views_ext']
            outputs_ext_1, _ = self.compute_outputs(
                data, num_views_ext, num_slots, loss_coef, require_results, deterministic_data, results)
            outputs_ext_1 = {'{}_1'.format(key): val for key, val in outputs_ext_1.items()}
            outputs_ext_2, _ = self.compute_outputs(
                data, -4, num_slots, loss_coef, require_results, deterministic_data, results)
            outputs_ext_2 = {'{}_2'.format(key): val for key, val in outputs_ext_2.items()}
            outputs_ext = {**outputs_ext_1, **outputs_ext_2}
        else:
            outputs_ext = None
        if require_results_gen:
            with torch.set_grad_enabled(False):
                outputs_view, outputs_attr = self.compute_outputs_gen(results, loss_coef)
        else:
            outputs_view = None
            outputs_attr = None
        return outputs, outputs_ext, outputs_view, outputs_attr

    def compute_outputs(self, data, num_views, num_slots, loss_coef, require_results, deterministic_data,
                        results_prev=None):
        temp_pres = loss_coef['temp_pres']
        temp_ord = loss_coef['temp_ord']
        temp_shp = loss_coef['temp_shp']
        noise_scale_1 = loss_coef['noise_scale_1']
        noise_scale_2 = loss_coef['noise_scale_2']
        image, segment, overlap = self.convert_data(data, num_views, deterministic_data)
        results_enc = self.enc(image, num_slots, results_prev)
        results_dec = self.dec(results_enc, temp_pres, temp_ord, temp_shp, noise_scale_1, noise_scale_2, results_prev)
        results_raw = {'image': image, 'segment': segment, 'overlap': overlap, **results_enc, **results_dec}
        losses, loss_baseline = self.compute_losses(results_raw)
        metrics = self.compute_metrics(results_raw, loss_baseline)
        results = self.convert_results(results_raw) if require_results else {}
        outputs = {'result': results, 'metric': metrics, 'loss': losses}
        return outputs, results_raw

    def compute_outputs_gen(self, results, loss_coef):
        temp_ord = loss_coef['temp_ord']
        temp_shp = loss_coef['temp_shp']
        noise_scale_1 = loss_coef['noise_scale_1']
        noise_scale_2 = loss_coef['noise_scale_2']
        view_latent = results['view_latent']
        attr_obj_latent = results['attr_obj_latent']
        attr_bck_latent = results['attr_bck_latent']
        rand_view_latent = torch.randn_like(view_latent)
        rand_attr_obj_latent = torch.randn_like(attr_obj_latent)
        rand_attr_bck_latent = torch.randn_like(attr_bck_latent)
        logits_pres = results['logits_pres']
        pres = torch.sigmoid(logits_pres)
        results_view = self.dec.decode(rand_view_latent, attr_obj_latent, attr_bck_latent, pres, logits_pres, temp_ord,
                                       temp_shp, noise_scale_1, noise_scale_2)
        results_attr = self.dec.decode(view_latent, rand_attr_obj_latent, rand_attr_bck_latent, pres, logits_pres,
                                       temp_ord, temp_shp, noise_scale_1, noise_scale_2)
        for key, val in results.items():
            if key not in results_view:
                results_view[key] = val
            if key not in results_attr:
                results_attr[key] = val
        outputs_view = {'result': self.convert_results(results_view)}
        outputs_attr = {'result': self.convert_results(results_attr)}
        return outputs_view, outputs_attr

    @staticmethod
    def convert_data(data, num_views, deterministic_data):
        batch_size, data_views = data['image'].shape[:2]
        device = data['image'].device
        if deterministic_data:
            index = torch.arange(data_views, device=device)[None].expand(batch_size, -1)
        else:
            noise = torch.rand([batch_size, data_views], device=device)
            index = torch.argsort(noise, dim=1)
        if num_views >= 0:
            index = index[:, :num_views]
        else:
            index = index[:, num_views:]
        data = {key: select_by_index(val, index) for key, val in data.items()}
        image = data['image'].float() / 255
        segment_base = data['segment'][:, :, None, None].long()
        scatter_shape = [*segment_base.shape[:2], segment_base.max() + 1, *segment_base.shape[3:]]
        segment = torch.zeros(scatter_shape, device=segment_base.device).scatter_(2, segment_base, 1)
        overlap = torch.gt(data['overlap'][:, :, None, None], 1).float()
        return image, segment, overlap

    @staticmethod
    def convert_results(results):
        disc_key_list = [
            'image', 'recon', 'noisy_recon', 'recon_soft', 'mask', 'mask_soft',
            'apc', 'noisy_apc', 'shp', 'shp_soft', 'pres',
        ]
        cont_key_list = [
            'view_latent', 'attr_obj_latent', 'attr_bck_latent', 'logits_pres', 'zeta', 'logits_zeta',
            'log_ord', 'trs',
        ]
        full_key_list = disc_key_list + cont_key_list
        results = {key: val for key, val in results.items() if key in full_key_list}
        for key, val in results.items():
            if key in disc_key_list:
                results[key] = (val.clamp(0, 1) * 255).to(torch.uint8)
        return results

    def compute_losses(self, results):
        image = results['image']
        noisy_recon = results['noisy_recon']
        loss_nll = 0.5 * self.normal_invvar * (noisy_recon - image).square().sum([*range(1, image.ndim)])
        image_reshape = image.reshape(-1, *image.shape[2:])
        baseline_reshape = self.net_bl(image_reshape)
        baseline = baseline_reshape.reshape(*image.shape[:2]).sum(1)
        score = (-loss_nll - baseline).detach()
        sub_score_var, sub_score_mean = torch.var_mean(score, dim=0)
        num_views = image.shape[1]
        sq_num_views = pow(num_views, 2)
        self.score_mean = self.bl_momentum * self.score_mean + (1 - self.bl_momentum) * sub_score_mean / num_views
        self.score_var = self.bl_momentum * self.score_var + (1 - self.bl_momentum) * sub_score_var / sq_num_views
        score = (score - self.score_mean * num_views) / (self.score_var * sq_num_views).sqrt().clamp_min(1)
        sel_log_prob = results['sel_log_prob']
        loss_discrete = -score * sel_log_prob
        loss_baseline = -score * baseline
        loss_nll = loss_nll + loss_discrete - loss_discrete.detach() + loss_baseline - loss_baseline.detach()
        losses = {'nll': loss_nll}
        losses.update({key: val for key, val in results.items() if key.split('_')[0] in ['kld', 'reg']})
        return losses, loss_baseline

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

    def compute_metrics(self, results, loss_baseline):
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
        image = results['image']
        segment = results['segment']
        overlap = results['overlap']
        batch_size, num_views = image.shape[:2]
        segment_obj = segment[:, :, :-1]
        # ARI
        segment_all_sel = segment if self.seg_overlap else segment * (1 - overlap)
        segment_obj_sel = segment_obj if self.seg_overlap else segment_obj * (1 - overlap)
        mask_hard_all = results['mask']
        mask_hard_obj = mask_hard_all[:, :, :-1]
        mask_soft_all = results['mask_soft']
        mask_soft_obj = mask_soft_all[:, :, :-1]
        ari_hard_all_s, ari_hard_all_m = compute_ari_values(segment_all_sel, mask_hard_all)
        ari_hard_obj_s, ari_hard_obj_m = compute_ari_values(segment_obj_sel, mask_hard_obj)
        ari_soft_all_s, ari_soft_all_m = compute_ari_values(segment_all_sel, mask_soft_all)
        ari_soft_obj_s, ari_soft_obj_m = compute_ari_values(segment_obj_sel, mask_soft_obj)
        # MSE
        recon_hard = results['recon']
        recon_soft = results['recon_soft']
        mse_hard = (recon_hard - image).square().mean([*range(1, image.ndim)])
        mse_soft = (recon_soft - image).square().mean([*range(1, image.ndim)])
        # Count
        count_true = segment_obj.reshape(*segment_obj.shape[:-3], -1).max(-1).values.max(-2).values.sum(-1)
        count_pred = results['pres'][:, :-1].sum(1)
        count_acc = torch.eq(count_true, count_pred).to(dtype=torch.float)
        metrics = {
            'ari_hard_all_s': ari_hard_all_s, 'ari_hard_all_m': ari_hard_all_m,
            'ari_hard_obj_s': ari_hard_obj_s, 'ari_hard_obj_m': ari_hard_obj_m,
            'ari_soft_all_s': ari_soft_all_s, 'ari_soft_all_m': ari_soft_all_m,
            'ari_soft_obj_s': ari_soft_obj_s, 'ari_soft_obj_m': ari_soft_obj_m,
            'mse_hard': mse_hard, 'mse_soft': mse_soft, 'count': count_acc, 'baseline': loss_baseline,
        }
        return metrics


def get_model(config):
    net = Model(config).cuda()
    return nn.DataParallel(net)
