import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from scipy.optimize import linear_sum_assignment
from network import InitializerBack, InitializerFull, InitializerCrop, UpdaterBack, UpdaterFull, UpdaterCrop, \
    NetworkBack, NetworkFull, NetworkCrop


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # Hyperparameters
        self.image_shape = config['image_shape']
        self.crop_shape = config['crop_shape']
        self.normal_invvar = 1 / pow(config['normal_scale'], 2)
        self.normal_const = math.log(2 * math.pi / self.normal_invvar)
        self.prior_pres_alpha = config['prior_pres_alpha']
        self.prior_pres_log_alpha = math.log(self.prior_pres_alpha)
        self.seq_update = config['seq_update']
        self.seg_bck = config['seg_bck']
        # Neural networks
        self.init_back = InitializerBack(config)
        self.init_full = InitializerFull(config)
        self.init_crop = InitializerCrop(config)
        self.upd_back = UpdaterBack(config)
        self.upd_full = UpdaterFull(config)
        self.upd_crop = UpdaterCrop(config)
        self.net_back = NetworkBack(config)
        self.net_full = NetworkFull(config)
        self.net_crop = NetworkCrop(config)

    def forward(self, images, labels, num_slots, num_steps, step_wt, loss_coef_last, temp=None, hard=True, layers=None):
        # Initializations
        obj_slots = num_slots - 1
        states_back = self.init_back(images)
        result_bck = self.net_back(states_back[0])
        result_obj, states_dict = self.initialize_obj(images, result_bck, obj_slots, temp, hard)
        step_losses, kld_part = self.compute_step_losses(images, layers, result_bck, result_obj, loss_coef_last)
        losses = {key: [val] for key, val in step_losses.items()}
        # Refinements
        for step in range(num_steps):
            upd_back_in = self.compute_upd_back_in(images, result_bck, result_obj)
            states_back = self.upd_back(upd_back_in, states_back)
            result_bck = self.net_back(states_back[0])
            result_obj, states_dict = self.update_obj(images, result_bck, result_obj, states_dict, temp, hard)
            step_losses, kld_part = self.compute_step_losses(images, layers, result_bck, result_obj, loss_coef_last)
            step_losses['bck_prior'] = step_losses['bck_prior'] * (num_steps - step - 1) / num_steps
            for key, val in step_losses.items():
                losses[key].append(val)
        # Outputs
        sum_step_wt = step_wt.sum(1)
        losses = {key: torch.stack(val, dim=1) for key, val in losses.items()}
        losses = {key: (step_wt * val).sum(1) / sum_step_wt for key, val in losses.items()}
        indices = self.compute_indices(images, result_obj)
        perm_vals = {key: self.permute(result_obj[key], indices) for key in ['apc', 'shp', 'pres', 'scl', 'trs']}
        apc_all = torch.cat([perm_vals['apc'], result_bck['bck'][None]]).transpose(0, 1)
        shp = perm_vals['shp']
        shp_all = torch.cat([shp, torch.ones([1, *shp.shape[1:]], device=shp.device)]).transpose(0, 1)
        pres = torch.ge(perm_vals['pres'], 0.5).type(torch.float)
        mask = self.compute_mask(shp, pres).transpose(0, 1)
        log_mask = self.compute_log_mask(shp, pres).transpose(0, 1)
        pres = pres.squeeze(-1).transpose(0, 1)
        pres_all = torch.cat([pres, torch.ones([pres.shape[0], 1], device=images.device)], dim=1)
        recon = (mask * apc_all).sum(1)
        scl = perm_vals['scl'].transpose(0, 1)
        trs = perm_vals['trs'].transpose(0, 1)
        raw_pixel_ll = self.compute_raw_pixel_ll(images, result_bck['bck'], perm_vals['apc'])
        raw_pixel_ll = raw_pixel_ll.transpose(0, 1)
        segment_all = torch.argmax(mask, dim=1, keepdim=True)
        segment_obj = torch.argmax(mask[:, :-1], dim=1, keepdim=True)
        mask_oh_all = torch.zeros_like(mask)
        mask_oh_obj = torch.zeros_like(mask[:, :-1])
        mask_oh_all.scatter_(1, segment_all, 1)
        mask_oh_obj.scatter_(1, segment_obj, 1)
        results = {'apc': apc_all, 'shp': shp_all, 'pres': pres_all, 'scl': scl, 'trs': trs, 'recon': recon,
                   'mask': mask, 'segment_all': segment_all, 'segment_obj': segment_obj}
        metrics = self.compute_metrics(images, labels, pres, mask_oh_all, mask_oh_obj, recon, log_mask, raw_pixel_ll)
        losses['compare'] = -metrics['ll'] + kld_part
        return results, metrics, losses

    def initialize_obj(self, images, result_bck, obj_slots, temp, hard):
        result_obj = {
            'apc': torch.zeros([0, *images.shape], device=images.device),
            'shp': torch.zeros([0, images.shape[0], 1, *images.shape[2:]], device=images.device),
            'pres': torch.zeros([0, images.shape[0], 1], device=images.device),
        }
        states_dict = {key: [] for key in ['full', 'crop']}
        states_main = None
        for _ in range(obj_slots):
            # Full
            init_full_in = self.compute_init_full_in(images, result_bck, result_obj)
            states_full, states_main = self.init_full(init_full_in, states_main)
            result_full = self.net_full(states_full[0])
            # Crop
            grid_crop, grid_full = self.compute_grid(result_full['scl'], result_full['trs'])
            init_crop_in = nn_func.grid_sample(init_full_in, grid_crop, align_corners=False).detach()
            states_crop = self.init_crop(init_crop_in)
            result_crop = self.net_crop(states_full[0], states_crop[0], grid_full, temp, hard)
            # Update storage
            update_dict = {**result_full, **result_crop}
            for key, val in update_dict.items():
                if key in result_obj:
                    result_obj[key] = torch.cat([result_obj[key], val[None]])
                else:
                    result_obj[key] = val[None]
            update_dict = {'full': states_full, 'crop': states_crop}
            for key, val in update_dict.items():
                states_dict[key].append(val)
        if not self.seq_update:
            for key, val in states_dict.items():
                states_dict[key] = tuple([torch.cat(n) for n in zip(*val)])
        return result_obj, states_dict

    def update_obj(self, images, result_bck, result_obj, states_dict, temp, hard):
        obj_slots = result_obj['apc'].shape[0]
        if self.seq_update:
            for idx in range(obj_slots):
                # Full
                upd_full_in = self.compute_upd_full_in_seq(images, result_bck, result_obj, idx)
                states_full = self.upd_full(upd_full_in, states_dict['full'][idx])
                result_full = self.net_full(states_full[0])
                # Crop
                grid_crop, grid_full = self.compute_grid(result_full['scl'], result_full['trs'])
                upd_crop_in = self.compute_upd_crop_in_seq(result_obj, upd_full_in, grid_crop, idx)
                states_crop = self.upd_crop(upd_crop_in, states_dict['crop'][idx])
                result_crop = self.net_crop(states_full[0], states_crop[0], grid_full, temp, hard)
                # Update storage
                update_dict = {**result_full, **result_crop}
                for key, val in result_obj.items():
                    result_obj[key] = torch.cat([val[:idx], update_dict[key][None], val[idx + 1:]])
                update_dict = {'full': states_full, 'crop': states_crop}
                for key, val in states_dict.items():
                    states_dict[key] = val[:idx] + [update_dict[key]] + val[idx + 1:]
        else:
            # Full
            upd_full_in = self.compute_upd_full_in_para(images, result_bck, result_obj)
            states_full = self.upd_full(upd_full_in, states_dict['full'])
            result_full = self.net_full(states_full[0])
            # Crop
            grid_crop, grid_full = self.compute_grid(result_full['scl'], result_full['trs'])
            upd_crop_in = self.compute_upd_crop_in_para(result_obj, upd_full_in, grid_crop)
            states_crop = self.upd_crop(upd_crop_in, states_dict['crop'])
            result_crop = self.net_crop(states_full[0], states_crop[0], grid_full, temp, hard)
            # Update storage
            update_dict = {**result_full, **result_crop}
            for key, val in update_dict.items():
                result_obj[key] = val.reshape(obj_slots, -1, *val.shape[1:])
            update_dict = {'full': states_full, 'crop': states_crop}
            for key, val in update_dict.items():
                states_dict[key] = val
        return result_obj, states_dict

    def compute_grid(self, scl, trs):
        batch_size = scl.shape[0]
        zeros = torch.zeros_like(scl[:, 0])
        theta_crop = torch.stack([
            torch.stack([scl[:, 0], zeros, trs[:, 0]], dim=1),
            torch.stack([zeros, scl[:, 0], trs[:, 1]], dim=1),
        ], dim=1)
        theta_full = torch.stack([
            torch.stack([1 / scl[:, 0], zeros, -trs[:, 0] / scl[:, 0]], dim=1),
            torch.stack([zeros, 1 / scl[:, 0], -trs[:, 1] / scl[:, 0]], dim=1),
        ], dim=1)
        grid_crop = nn_func.affine_grid(theta_crop, [batch_size, 1, *self.crop_shape[1:]], align_corners=False)
        grid_full = nn_func.affine_grid(theta_full, [batch_size, 1, *self.image_shape[1:]], align_corners=False)
        return grid_crop, grid_full

    @staticmethod
    def compute_mask(shp, pres):
        x = shp * pres[..., None, None]
        ones = torch.ones([1, *x.shape[1:]], device=x.device)
        return torch.cat([x, ones]) * torch.cat([ones, 1 - x]).cumprod(0)

    @staticmethod
    def compute_log_mask(shp, pres, eps=1e-5):
        x = shp * pres[..., None, None]
        log_x = torch.log(x + eps)
        log1m_x = torch.log(1 - x + eps)
        zeros = torch.zeros([1, *x.shape[1:]], device=x.device)
        return torch.cat([log_x, zeros]) + torch.cat([zeros, log1m_x]).cumsum(0)

    def compute_indices(self, images, result_obj, eps=1e-5):
        sq_diffs = (result_obj['apc'] - images[None]).pow(2).sum(-3, keepdim=True).detach()
        visibles = result_obj['shp'].clone().detach()
        pres = result_obj['pres'].detach()
        coefs = torch.ones(visibles.shape[:-2], device=visibles.device)
        if coefs.shape[0] == 0:
            return coefs.type(torch.long)
        indices_list = []
        for _ in range(coefs.shape[0]):
            vis_sq_diffs = (visibles * sq_diffs).sum([-2, -1])
            vis_areas = visibles.sum([-2, -1])
            vis_max_vals = visibles.reshape(*visibles.shape[:-2], -1).max(-1).values
            scores = torch.exp(-0.5 * self.normal_invvar * vis_sq_diffs / (vis_areas + eps))
            scaled_scores = coefs * (vis_max_vals * pres * scores + 1)
            indices = torch.argmax(scaled_scores, dim=0, keepdim=True)
            indices_list.append(indices)
            vis = torch.gather(visibles, 0, indices[..., None, None].expand(-1, -1, *visibles.shape[2:]))
            visibles *= 1 - vis
            coefs.scatter_(0, indices, -1)
        return torch.cat(indices_list)

    @staticmethod
    def permute(x, indices):
        if x.ndim == 3:
            indices_expand = indices
        else:
            indices_expand = indices[..., None, None]
        x = torch.gather(x, 0, indices_expand.expand(-1, -1, *x.shape[2:]))
        return x

    def compute_init_full_in(self, images, result_bck, result_obj):
        indices = self.compute_indices(images, result_obj)
        perm_vals = {key: self.permute(result_obj[key], indices) for key in ['apc', 'shp', 'pres']}
        mask = self.compute_mask(perm_vals['shp'], perm_vals['pres'])
        recon = (mask * torch.cat([perm_vals['apc'], result_bck['bck'][None]])).sum(0)
        return torch.cat([images, recon, mask[-1]], dim=1).detach()

    def compute_upd_back_in(self, images, result_bck, result_obj):
        inputs_excl = self.compute_init_full_in(images, result_bck, result_obj)
        return torch.cat([inputs_excl, result_bck['bck']], dim=1).detach()

    def compute_upd_full_in_seq(self, images, result_bck, result_obj, idx):
        indices = self.compute_indices(images, result_obj)
        perm_vals = {key: self.permute(result_obj[key], indices) for key in ['apc', 'shp', 'pres']}
        pos_cur = torch.eq(indices, idx)
        pres_excl = torch.where(pos_cur, torch.zeros_like(perm_vals['pres']), perm_vals['pres'])
        mask_excl = self.compute_mask(perm_vals['shp'], pres_excl)
        recon_excl = (mask_excl * torch.cat([perm_vals['apc'], result_bck['bck'][None]])).sum(0)
        pos_above = torch.eq(pos_cur.type(torch.int32).cumsum(0), 0)
        pos_above = pos_above[..., None, None].expand(-1, -1, -1, *self.image_shape[1:])
        shp_above = torch.where(pos_above, mask_excl[:-1], torch.zeros_like(mask_excl[:-1])).sum(0)
        apc_cur = result_obj['apc'][idx]
        shp_cur = result_obj['shp'][idx]
        pres_cur = result_obj['pres'][idx, ..., None, None].expand(-1, -1, *self.image_shape[1:])
        return torch.cat([images, recon_excl, shp_above, apc_cur, shp_cur, pres_cur], dim=1).detach()

    def compute_upd_full_in_para(self, images, result_bck, result_obj):
        obj_slots = result_obj['apc'].shape[0]
        indices = self.compute_indices(images, result_obj)
        perm_vals = {key: self.permute(result_obj[key], indices) for key in ['apc', 'shp', 'pres']}
        images = images[None].expand(obj_slots, -1, -1, -1, -1)
        bck = result_bck['bck'][None].expand(obj_slots, -1, -1, -1, -1)
        indices = indices[:, None].expand(-1, obj_slots, -1, -1)
        perm_vals = {key: val[:, None].expand(-1, obj_slots, *([-1] * (val.ndim - 1)))
                     for key, val in perm_vals.items()}
        pos_cur = torch.eq(indices, torch.arange(obj_slots, device=indices.device)[None, :, None, None])
        pres_excl = torch.where(pos_cur, torch.zeros_like(perm_vals['pres']), perm_vals['pres'])
        mask_excl = self.compute_mask(perm_vals['shp'], pres_excl)
        recon_excl = (mask_excl * torch.cat([perm_vals['apc'], bck[None]])).sum(0)
        pos_above = torch.eq(pos_cur.type(torch.int32).cumsum(0), 0)
        pos_above = pos_above[..., None, None].expand(-1, -1, -1, -1, *self.image_shape[1:])
        shp_above = torch.where(pos_above, mask_excl[:-1], torch.zeros_like(mask_excl[:-1])).sum(0)
        apc_cur = result_obj['apc']
        shp_cur = result_obj['shp']
        pres_cur = result_obj['pres'][..., None, None].expand(-1, -1, -1, *self.image_shape[1:])
        inputs = torch.cat([images, recon_excl, shp_above, apc_cur, shp_cur, pres_cur], dim=2).detach()
        return inputs.reshape(-1, *inputs.shape[2:])

    def compute_upd_crop_in_seq(self, result_obj, upd_full_in, grid_crop, idx):
        excl_dims = self.image_shape[0] + 2
        inputs_excl = nn_func.grid_sample(upd_full_in[:, :-excl_dims], grid_crop, align_corners=False)
        apc_cur = result_obj['apc_crop'][idx]
        shp_cur = result_obj['shp_crop'][idx]
        pres_cur = result_obj['pres'][idx, ..., None, None].expand(-1, -1, *self.crop_shape[1:])
        return torch.cat([inputs_excl, apc_cur, shp_cur, pres_cur], dim=1).detach()

    def compute_upd_crop_in_para(self, result_obj, upd_full_in, grid_crop):
        obj_slots = result_obj['apc'].shape[0]
        excl_dims = self.image_shape[0] + 2
        inputs_excl = nn_func.grid_sample(upd_full_in[:, :-excl_dims], grid_crop, align_corners=False)
        inputs_excl = inputs_excl.reshape(obj_slots, -1, *inputs_excl.shape[1:])
        apc_cur = result_obj['apc_crop']
        shp_cur = result_obj['shp_crop']
        pres_cur = result_obj['pres'][..., None, None].expand(-1, -1, -1, *self.crop_shape[1:])
        inputs = torch.cat([inputs_excl, apc_cur, shp_cur, pres_cur], dim=2).detach()
        return inputs.reshape(-1, *inputs.shape[2:])

    def compute_raw_pixel_ll(self, images, bck, apc):
        sq_diff = (torch.cat([apc, bck[None]]) - images[None]).pow(2)
        raw_pixel_ll = -0.5 * (self.normal_const + self.normal_invvar * sq_diff).sum(-3, keepdim=True)
        return raw_pixel_ll

    def compute_kld_pres(self, result_obj):
        tau1 = result_obj['tau1']
        tau2 = result_obj['tau2']
        logits_zeta = result_obj['logits_zeta']
        psi1 = torch.digamma(tau1)
        psi2 = torch.digamma(tau2)
        psi12 = torch.digamma(tau1 + tau2)
        kld_1 = torch.lgamma(tau1 + tau2) - torch.lgamma(tau1) - torch.lgamma(tau2) - self.prior_pres_log_alpha
        kld_2 = (tau1 - self.prior_pres_alpha) * psi1
        kld_3 = (tau2 - 1) * psi2
        kld_4 = -(tau1 + tau2 - self.prior_pres_alpha - 1) * psi12
        zeta = torch.sigmoid(logits_zeta)
        log_zeta = nn_func.logsigmoid(logits_zeta)
        log1m_zeta = log_zeta - logits_zeta
        psi1_le_sum = psi1.cumsum(0)
        psi12_le_sum = psi12.cumsum(0)
        kappa1 = psi1_le_sum - psi12_le_sum
        psi1_lt_sum = torch.cat([torch.zeros([1, *psi1_le_sum.shape[1:]], device=zeta.device), psi1_le_sum[:-1]])
        logits_coef = psi2 + psi1_lt_sum - psi12_le_sum
        kappa2_list = []
        for idx in range(logits_coef.shape[0]):
            coef = torch.softmax(logits_coef[:idx + 1], dim=0)
            log_coef = nn_func.log_softmax(logits_coef[:idx + 1], dim=0)
            coef_le_sum = coef.cumsum(0)
            coef_lt_sum = torch.cat([torch.zeros([1, *coef_le_sum.shape[1:]], device=zeta.device), coef_le_sum[:-1]])
            part1 = (coef * psi2[:idx + 1]).sum(0)
            part2 = ((1 - coef_le_sum[:-1]) * psi1[:idx]).sum(0)
            part3 = -((1 - coef_lt_sum) * psi12[:idx + 1]).sum(0)
            part4 = -(coef * log_coef).sum(0)
            kappa2_list.append(part1 + part2 + part3 + part4)
        kappa2 = torch.stack(kappa2_list)
        kld_5 = zeta * (log_zeta - kappa1) + (1 - zeta) * (log1m_zeta - kappa2)
        kld = kld_1 + kld_2 + kld_3 + kld_4 + kld_5
        return kld.sum([0, *range(2, kld.ndim)])

    @staticmethod
    def compute_order(cost_list):
        order_list = []
        for cost in cost_list:
            _, cols = linear_sum_assignment(cost)
            order_list.append(cols)
        return np.array(order_list)

    def compute_step_losses(self, images, layers, result_bck, result_obj, loss_coef_last):
        indices = self.compute_indices(images, result_obj)
        perm_vals = {key: self.permute(result_obj[key], indices)
                     for key in ['apc', 'shp', 'pres', 'tau1', 'tau2', 'logits_zeta']}
        mask = self.compute_mask(perm_vals['shp'], perm_vals['pres'])
        log_mask = self.compute_log_mask(perm_vals['shp'], perm_vals['pres'])
        raw_pixel_ll = self.compute_raw_pixel_ll(images, result_bck['bck'], perm_vals['apc'])
        masked_pixel_ll = log_mask + raw_pixel_ll
        gamma = torch.softmax(masked_pixel_ll, dim=0)
        log_gamma = nn_func.log_softmax(masked_pixel_ll, dim=0)
        # Loss NLL
        loss_nll = -(gamma * raw_pixel_ll).sum([0, *range(2, gamma.ndim)])
        # Loss KLD
        kld_bck = result_bck['bck_kld']
        kld_obj = result_obj['obj_kld'].sum(0)
        kld_stn = result_obj['stn_kld'].sum(0)
        kld_pres = self.compute_kld_pres(perm_vals)
        kld_mask = (gamma * (log_gamma - log_mask)).sum([0, *range(2, gamma.ndim)])
        # Loss recon
        recon = (mask * torch.cat([perm_vals['apc'], result_bck['bck'][None]])).sum(0)
        sq_diff = (recon - images).pow(2)
        loss_recon = 0.5 * self.normal_invvar * sq_diff.sum([*range(1, sq_diff.ndim)])
        # Loss back prior
        sq_diff = (result_bck['bck'] - images).pow(2)
        loss_bck_prior = 0.5 * self.normal_invvar * sq_diff.sum([*range(1, sq_diff.ndim)])
        # Losses
        losses = {'nll': loss_nll, 'kld_bck': kld_bck, 'kld_obj': kld_obj, 'kld_stn': kld_stn, 'kld_pres': kld_pres,
                  'kld_mask': kld_mask, 'recon': loss_recon, 'bck_prior': loss_bck_prior}
        kld_part_key_list = ['kld_{}'.format(key) for key in ['bck', 'obj', 'stn', 'pres']]
        kld_part = torch.stack([loss_coef_last[key] * losses[key] for key in kld_part_key_list]).sum(0)
        # Extra
        if layers is not None:
            target_shp = layers[:, :-1, -1:].transpose(0, 1)
            pad_slots = perm_vals['shp'].shape[0] - target_shp.shape[0]
            zeros = torch.zeros([pad_slots, *target_shp.shape[1:]], device=target_shp.device)
            target_shp = torch.cat([target_shp, zeros])
            target_pres = target_shp.reshape(*target_shp.shape[:-3], -1).max(-1).values[..., None]
            target_mask = self.compute_mask(target_shp, target_pres)
            order_cost = -(mask[:-1, None] * target_mask[None, :-1]).sum([-3, -2, -1])
            order_cost = order_cost.permute(2, 0, 1).data.cpu().numpy()
            order = self.compute_order(order_cost)
            indices = torch.tensor(order, device=target_mask.device).transpose(0, 1)
            indices_shp = indices[..., None, None, None].expand(-1, -1, *target_shp.shape[2:])
            indices_pres = indices[..., None]
            target_shp = torch.gather(target_shp, 0, indices_shp)
            target_pres = torch.gather(target_pres, 0, indices_pres)
            target_mask = self.compute_mask(target_shp, target_pres)
            target_log_mask = self.compute_log_mask(target_shp, target_pres)
            loss_mask = target_mask * (target_log_mask - log_mask)
            losses['mask'] = loss_mask.sum([0, *range(2, loss_mask.ndim)])
            loss_pres = self.image_shape[1] * self.image_shape[2] * nn_func.binary_cross_entropy_with_logits(
                perm_vals['logits_zeta'], target_pres, reduction='none')
            losses['pres'] = loss_pres.sum([0, *range(2, loss_pres.ndim)])
        return losses, kld_part

    @staticmethod
    def compute_ari(mask_true, mask_pred):
        def comb2(x):
            x = x * (x - 1)
            if x.ndim > 1:
                x = x.sum([*range(1, x.ndim)])
            return x
        num_pixels = mask_true.sum([*range(1, mask_true.ndim)])
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

    def compute_metrics(self, images, labels, pres, mask_oh_all, mask_oh_obj, recon, log_mask, raw_pixel_ll):
        # ARI
        ari_all = self.compute_ari(labels, mask_oh_all)
        ari_obj = self.compute_ari(labels, mask_oh_obj)
        # MSE
        sq_diff = (recon - images).pow(2)
        mse = sq_diff.mean([*range(1, sq_diff.ndim)])
        # Log-likelihood
        pixel_ll = torch.logsumexp(log_mask + raw_pixel_ll, dim=1)
        ll = pixel_ll.sum([*range(1, pixel_ll.ndim)])
        # Count
        pres_true = labels.reshape(*labels.shape[:-3], -1).max(-1).values
        if self.seg_bck:
            pres_true = pres_true[:, 1:]
        count_true = pres_true.sum(1)
        count_pred = pres.sum(1)
        count_acc = (count_true == count_pred).to(dtype=torch.float)
        metrics = {'ari_all': ari_all, 'ari_obj': ari_obj, 'mse': mse, 'll': ll, 'count': count_acc}
        return metrics

    @staticmethod
    def convert_single(x_in, boarder_size, color=None):
        x = nn_func.pad(x_in, [boarder_size] * 4, value=0)
        if color is not None:
            boarder = nn_func.pad(torch.zeros_like(x_in), [boarder_size] * 4, value=1) * color
            x += boarder
        x = nn_func.pad(x, [boarder_size] * 4, value=1)
        return x

    def convert_multiple(self, x, boarder_size, color=None):
        batch_size, num_slots = x.shape[:2]
        x = x.reshape(batch_size * num_slots, *x.shape[2:])
        if color is not None:
            color = color.reshape(batch_size * num_slots, *color.shape[2:])
        x = self.convert_single(x, boarder_size, color=color)
        x = x.reshape(batch_size, num_slots, *x.shape[1:])
        x = torch.cat(torch.unbind(x, dim=1), dim=-1)
        return x

    def compute_overview(self, images, results):
        boarder_size = round(min(images.shape[-2:]) / 32)
        images = images.expand(-1, 3, -1, -1)
        recon = results['recon'].expand(-1, 3, -1, -1)
        apc = results['apc'].expand(-1, -1, 3, -1, -1)
        shp = results['shp'].expand(-1, -1, 3, -1, -1)
        pres = results['pres'][..., None, None, None].expand(-1, -1, 3, -1, -1)
        scl = results['scl'].reshape(-1, results['scl'].shape[-1])
        trs = results['trs'].reshape(-1, results['trs'].shape[-1])
        _, grid_full = self.compute_grid(scl, trs)
        white_crop = torch.ones([scl.shape[0], 3, *self.crop_shape[1:]], device=images.device)
        shp_obj_mask = nn_func.grid_sample(white_crop, grid_full, align_corners=False)
        shp_obj_mask = shp_obj_mask.reshape(*results['scl'].shape[:2], 3, *self.image_shape[1:])
        area_color = 1 - shp_obj_mask
        area_color[..., 0, :, :] *= 0.5
        area_color[..., 2, :, :] *= 0.5
        shp = torch.cat([shp[:, :-1] + area_color, shp[:, -1:]], dim=1)
        color_0 = torch.zeros_like(pres)
        color_1 = torch.zeros_like(pres)
        color_0[..., 1, :, :] = 0.5
        color_0[..., 2, :, :] = 1
        color_1[..., 0, :, :] = 1
        color_1[..., 1, :, :] = 0.5
        boarder_color = pres * color_1 + (1 - pres) * color_0
        boarder_color[:, -1] = 0
        row1 = torch.cat([
            self.convert_single(images, boarder_size),
            self.convert_multiple(apc, boarder_size),
        ], dim=-1)
        row2 = torch.cat([
            self.convert_single(recon, boarder_size),
            self.convert_multiple(shp, boarder_size, color=boarder_color),
        ], dim=-1)
        overview = torch.cat([row1, row2], dim=-2)
        overview = nn_func.pad(overview, [boarder_size * 4] * 4, value=1)
        overview = (overview.clamp_(0, 1) * 255).to(dtype=torch.uint8)
        return overview

    def generate(self, batch_size, num_slots):
        obj_slots = num_slots - 1
        mu = self.net_back.prior_mu[None].expand(batch_size, -1)
        logvar = self.net_back.prior_logvar[None].expand(batch_size, -1)
        bck = self.net_back.decode(mu, logvar)
        mu = self.net_full.prior_mu[None].expand(obj_slots * batch_size, -1)
        logvar = self.net_full.prior_logvar[None].expand(obj_slots * batch_size, -1)
        scl, trs = self.net_full.decode(mu, logvar)
        _, grid_full = self.compute_grid(scl, trs)
        mu = self.net_crop.prior_mu[None].expand(obj_slots * batch_size, -1)
        logvar = self.net_crop.prior_logvar[None].expand(obj_slots * batch_size, -1)
        logits_zeta = torch.zeros([obj_slots * batch_size, 1], device=mu.device)
        _, apc, shp, _, _ = self.net_crop.decode(logits_zeta, mu, logvar, grid_full, None, True)
        apc = apc.reshape([obj_slots, batch_size, *apc.shape[1:]])
        shp = shp.reshape([obj_slots, batch_size, *shp.shape[1:]])
        num_obj = torch.randint(num_slots, size=[batch_size], device=mu.device)
        pres = (torch.arange(obj_slots, device=mu.device)[:, None] < num_obj[None])[..., None].type(torch.float)
        shp = shp * pres[..., None, None]
        apc_all = torch.cat([apc, bck[None]]).transpose(0, 1)
        shp_all = torch.cat([shp, torch.ones([1, *shp.shape[1:]], device=shp.device)]).transpose(0, 1)
        mask = self.compute_mask(shp, pres).transpose(0, 1)
        recon = (mask * apc_all).sum(1)
        layers = torch.cat([apc_all, shp_all], dim=-3)
        if self.seg_bck:
            segment_all = torch.argmax(mask, dim=1, keepdim=True)
            mask_oh = torch.zeros_like(mask)
            mask_oh.scatter_(1, segment_all, 1)
        else:
            segment_obj = torch.argmax(mask[:, :-1], dim=1, keepdim=True)
            mask_oh = torch.zeros_like(mask[:, :-1])
            mask_oh.scatter_(1, segment_obj, 1)
        outputs = {'image': recon, 'label': mask_oh, 'layers': layers}
        outputs = {key: val.detach() for key, val in outputs.items()}
        return outputs

    def compute_overview_gen(self, data_gen):
        images = data_gen['image']
        apc = data_gen['layers'][:, :, :-1]
        shp = data_gen['layers'][:, :, -1:]
        boarder_size = round(min(images.shape[-2:]) / 32)
        images = images.expand(-1, 3, -1, -1)
        apc = apc.expand(-1, -1, 3, -1, -1)
        shp = shp.expand(-1, -1, 3, -1, -1)
        row1 = torch.cat([
            self.convert_single(images, boarder_size),
            self.convert_multiple(apc, boarder_size),
        ], dim=-1)
        row2 = torch.cat([
            self.convert_single(torch.zeros_like(images), boarder_size),
            self.convert_multiple(shp, boarder_size),
        ], dim=-1)
        overview = torch.cat([row1, row2], dim=-2)
        overview = nn_func.pad(overview, [boarder_size * 4] * 4, value=1)
        overview = (overview.clamp_(0, 1) * 255).to(dtype=torch.uint8)
        return overview


def get_model(config):
    net = Model(config).cuda()
    return net
