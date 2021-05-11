import math
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from network import Updater, NetworkBack, NetworkPres, NetworkWhere, NetworkWhat


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # Hyperparameters
        self.register_buffer('prior_log_alpha', torch.tensor([config['prior_pres_alpha']]).log())
        self.register_buffer('prior_log1m_alpha', torch.tensor([1 - config['prior_pres_alpha']]).log())
        self.image_shape = config['image_shape']
        self.crop_shape = config['crop_shape']
        self.normal_invvar = 1 / pow(config['normal_scale'], 2)
        self.normal_const = math.log(2 * math.pi / self.normal_invvar)
        self.seg_overlap = config['seg_overlap']
        # Neural networks
        self.upd = Updater(config)
        self.net_back = NetworkBack(config)
        self.net_pres = NetworkPres(config)
        self.net_where = NetworkWhere(config)
        self.net_what = NetworkWhat(config)

    def forward(self, data, phase_param, temp=None, hard=True, require_extra=True):
        num_slots = phase_param['num_slots']
        data = {key: val.cuda(non_blocking=True) for key, val in data.items()}
        images = data['image'].float() / 255
        segments = data['segment'][:, None, None].long()
        scatter_shape = [segments.shape[0], segments.max() + 1, *segments.shape[2:]]
        segments = torch.zeros(scatter_shape, device=segments.device).scatter_(1, segments, 1)
        overlaps = torch.gt(data['overlap'][:, None, None], 1).float()
        # Background
        result_bck = self.net_back(images)
        canvas = result_bck['back'].clone().detach()
        # Objects
        obj_slots = num_slots - 1
        result_obj = {}
        pres = torch.ones([images.shape[0], 1], device=images.device)
        states = None
        for _ in range(obj_slots):
            # States
            inputs_full = torch.cat([images, canvas], dim=1)
            states = self.upd(inputs_full, states)
            # Presence
            result_pres = self.net_pres(states[0], temp, hard)
            pres = pres * result_pres['cond_pres']
            # Where
            result_where = self.net_where(states[0])
            # What
            grid_crop, grid_full = self.compute_grid(result_where['scl'], result_where['trs'])
            result_what = self.net_what(inputs_full, grid_crop, grid_full)
            # Update canvas
            canvas += (result_what['what'] * pres[..., None, None]).detach()
            # Update storage
            update_dict = {**result_pres, **result_where, **result_what, 'pres': pres}
            for key, val in update_dict.items():
                if key in result_obj:
                    result_obj[key].append(val)
                else:
                    result_obj[key] = [val]
        result_obj = {key: torch.stack(val, dim=1) for key, val in result_obj.items()}
        # Outputs
        losses, kld, recon = self.compute_loss_values(images, result_bck, result_obj)
        losses['compare'] = losses['nll'] + kld
        apc_pres = result_bck['back'][:, None] + result_obj['what'] * result_obj['pres'][..., None, None]
        apc_pres_all = torch.cat([apc_pres, result_bck['back'][:, None]], dim=1)
        apc = result_bck['back'][:, None] + result_obj['what']
        apc_all = torch.cat([apc, result_bck['back'][:, None]], dim=1)
        pres = torch.ge(result_obj['pres'], 0.5).type(torch.float).squeeze(-1)
        pres_all = torch.cat([pres, torch.ones([pres.shape[0], 1], device=images.device)], dim=1)
        logits_mask = -0.5 * self.normal_invvar * (apc_pres_all - images[:, None]).pow(2).sum(-3, keepdim=True)
        mask = nn_func.softmax(logits_mask, dim=1)
        segment_all = torch.argmax(mask, dim=1, keepdim=True)
        segment_obj = torch.argmax(mask[:, :-1], dim=1, keepdim=True)
        mask_oh_all = torch.zeros_like(mask).scatter_(1, segment_all, 1)
        mask_oh_obj = torch.zeros_like(mask[:, :-1]).scatter_(1, segment_obj, 1)
        if require_extra:
            results = {'apc': apc_all, 'mask': mask, 'pres': pres_all, 'recon': recon}
            results = {key: (val.clamp(0, 1) * 255).to(torch.uint8) for key, val in results.items()}
            results.update({key: result_obj[key] for key in ['scl', 'trs']})
        else:
            results = {}
        metrics = self.compute_metrics(images, segments, overlaps, pres, mask_oh_all, mask_oh_obj, recon, losses['nll'])
        return results, metrics, losses

    def compute_grid(self, scl, trs):
        batch_size = scl.shape[0]
        zeros = torch.zeros_like(scl[:, 0])
        theta_crop = torch.stack([
            torch.stack([scl[:, 0], zeros, trs[:, 0]], dim=1),
            torch.stack([zeros, scl[:, 1], trs[:, 1]], dim=1),
        ], dim=1)
        theta_full = torch.stack([
            torch.stack([1 / scl[:, 0], zeros, -trs[:, 0] / scl[:, 0]], dim=1),
            torch.stack([zeros, 1 / scl[:, 1], -trs[:, 1] / scl[:, 1]], dim=1),
        ], dim=1)
        grid_crop = nn_func.affine_grid(theta_crop, [batch_size, 1, *self.crop_shape[1:]], align_corners=False)
        grid_full = nn_func.affine_grid(theta_full, [batch_size, 1, *self.image_shape[1:]], align_corners=False)
        return grid_crop, grid_full

    def compute_kld_pres(self, result_obj):
        logits_cond_zeta = result_obj['logits_cond_zeta']
        cond_zeta = torch.sigmoid(logits_cond_zeta)
        batch_size = cond_zeta.shape[0]
        zeros = torch.zeros([batch_size, 1, 1], device=cond_zeta.device)
        ones = torch.ones([batch_size, 1, 1], device=cond_zeta.device)
        obj_slots = logits_cond_zeta.shape[1]
        log_alpha = self.prior_log_alpha.expand(batch_size, obj_slots + 1, -1)
        log1m_alpha = self.prior_log1m_alpha.expand(batch_size, obj_slots, -1)
        log_p_step = log_alpha + torch.cat([zeros, log1m_alpha], dim=1).cumsum(1)
        log_cond_zeta = nn_func.logsigmoid(logits_cond_zeta)
        log1m_cond_zeta = log_cond_zeta - logits_cond_zeta
        q_step = torch.cat([1 - cond_zeta, ones], dim=1) * torch.cat([ones, cond_zeta], dim=1).cumprod(1)
        log_q_step = torch.cat([log1m_cond_zeta, zeros], dim=1) + torch.cat([zeros, log_cond_zeta], dim=1).cumsum(1)
        kld = (q_step * (log_q_step - log_p_step)).sum([1, 2])
        return kld

    def compute_loss_values(self, images, result_bck, result_obj):
        recon = result_bck['back'] + (result_obj['what'] * result_obj['pres'][..., None, None]).sum(1)
        loss_nll = 0.5 * (self.normal_const + self.normal_invvar * (recon - images).pow(2))
        loss_nll = loss_nll.sum([*range(1, loss_nll.dim())])
        # Loss KLD
        kld_pres = self.compute_kld_pres(result_obj)
        kld_back = result_bck['back_kld']
        kld_where = result_obj['where_kld'].sum(1)
        kld_what = result_obj['what_kld'].sum(1)
        # Loss back prior
        bck_prior = images.reshape(*images.shape[:-2], -1).median(-1).values[..., None, None]
        sq_diff = (result_bck['back'] - bck_prior).pow(2)
        loss_bck_prior = 0.5 * self.normal_invvar * sq_diff.sum([*range(1, sq_diff.dim())])
        # Losses
        losses = {'nll': loss_nll, 'kld_back': kld_back, 'kld_pres': kld_pres, 'kld_where': kld_where,
                  'kld_what': kld_what, 'bck_prior': loss_bck_prior}
        kld_key_list = ['kld_{}'.format(key) for key in ['back', 'pres', 'where', 'what']]
        kld = torch.stack([losses[key] for key in kld_key_list]).sum(0)
        return losses, kld, recon

    @staticmethod
    def compute_ari(mask_true, mask_pred):
        def comb2(x):
            x = x * (x - 1)
            if x.dim() > 1:
                x = x.sum([*range(1, x.dim())])
            return x
        num_pixels = mask_true.sum([*range(1, mask_true.dim())])
        mask_true = mask_true.reshape(
            [mask_true.shape[0], mask_true.shape[1], 1, mask_true.shape[-2] * mask_true.shape[-1]])
        mask_pred = mask_pred.reshape(
            [mask_pred.shape[0], 1, mask_pred.shape[1], mask_pred.shape[-2] * mask_pred.shape[-1]])
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

    def compute_metrics(self, images, segments, overlaps, pres, mask_oh_all, mask_oh_obj, recon, nll):
        segments_obj = segments[:, :-1]
        # ARI
        segments_obj_sel = segments_obj if self.seg_overlap else segments_obj * (1 - overlaps)
        ari_all = self.compute_ari(segments_obj_sel, mask_oh_all)
        ari_obj = self.compute_ari(segments_obj_sel, mask_oh_obj)
        # MSE
        sq_diff = (recon - images).pow(2)
        mse = sq_diff.mean([*range(1, sq_diff.dim())])
        # Log-likelihood
        ll = -nll
        # Count
        count_true = segments_obj.reshape(*segments_obj.shape[:-3], -1).max(-1).values.sum(1)
        count_pred = pres.sum(1)
        count_acc = torch.eq(count_true, count_pred).to(dtype=torch.float)
        metrics = {'ari_all': ari_all, 'ari_obj': ari_obj, 'mse': mse, 'll': ll, 'count': count_acc}
        return metrics


def get_model(config):
    net = Model(config).cuda()
    return nn.DataParallel(net)
