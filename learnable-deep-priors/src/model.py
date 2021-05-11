import math
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from network import InitializerBck, InitializerObj, UpdaterBck, UpdaterObj, DecoderBck, DecoderObj


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # Hyperparameters
        self.normal_scale = config['normal_scale']
        self.normal_invvar = 1 / pow(self.normal_scale, 2)
        self.normal_const = math.log(2 * math.pi / self.normal_invvar)
        self.seg_overlap = config['seg_overlap']
        # Neural networks
        self.init_bck = InitializerBck(config)
        self.init_obj = InitializerObj(config)
        self.upd_bck = UpdaterBck(config)
        self.upd_obj = UpdaterObj(config)
        self.dec_bck = DecoderBck(config)
        self.dec_obj = DecoderObj(config)

    def forward(self, data, phase_param, require_extra=True):
        num_slots = phase_param['num_slots']
        num_steps = phase_param['num_steps']
        step_wt = self.get_step_wt(phase_param)
        data = {key: val.cuda(non_blocking=True) for key, val in data.items()}
        images = data['image'].float() / 255
        segments = data['segment'][:, None, None].long()
        scatter_shape = [segments.shape[0], segments.max() + 1, *segments.shape[2:]]
        segments = torch.zeros(scatter_shape, device=segments.device).scatter_(1, segments, 1)
        overlaps = torch.gt(data['overlap'][:, None, None], 1).float()
        # Initializations
        obj_slots = num_slots - 1
        states_bck = self.init_bck(images)
        result_bck = self.dec_bck(states_bck[0])
        result_obj = {
            'apc': torch.zeros([0, *images.shape], device=images.device),
            'shp': torch.zeros([0, images.shape[0], 1, *images.shape[2:]], device=images.device),
            'logits_shp': torch.zeros([0, images.shape[0], 1, *images.shape[2:]], device=images.device),
        }
        states_main = None
        states_obj_list = []
        for _ in range(obj_slots):
            indices = self.compute_indices(images, result_obj)
            perm_vals = {key: self.permute(result_obj[key], indices) for key in ['apc', 'shp']}
            mask = self.compute_mask(perm_vals['shp'])
            recon = (mask * torch.cat([perm_vals['apc'], result_bck['bck'][None]])).sum(0)
            init_obj_in = torch.cat([images, recon, mask[-1]], dim=1).detach()
            states_obj, states_main = self.init_obj(init_obj_in, states_main)
            update_dict = self.dec_obj(states_obj[0], obj_slots=1)
            for key, val in result_obj.items():
                result_obj[key] = torch.cat([val, update_dict[key]])
            states_obj_list.append(states_obj)
        states_obj = tuple([torch.cat(n) for n in zip(*states_obj_list)])
        states_obj = self.adjust_order(images, result_obj, states_obj)
        result_prob = self.compute_probabilities(images, result_bck, result_obj)
        step_losses = self.compute_step_losses(images, result_bck, result_prob)
        losses = {key: [val] for key, val in step_losses.items()}
        # Refinements
        for _ in range(num_steps):
            noisy_images = images + torch.randn_like(images) * self.normal_scale
            gamma = result_prob['gamma']
            upd_bck_in = gamma[-1] * (noisy_images - result_bck['bck'])
            upd_apc_in = gamma[:-1] * (noisy_images[None] - result_obj['apc'])
            upd_shp_in = gamma[:-1] * (1 - result_obj['shp']) - (1 - gamma[:-1].cumsum(0)) * result_obj['shp']
            states_bck = self.upd_bck(upd_bck_in, states_bck)
            states_obj = self.upd_obj(upd_apc_in, upd_shp_in, states_obj)
            result_bck = self.dec_bck(states_bck[0])
            result_obj = self.dec_obj(states_obj[0], obj_slots)
            states_obj = self.adjust_order(images, result_obj, states_obj)
            result_prob = self.compute_probabilities(images, result_bck, result_obj)
            step_losses = self.compute_step_losses(images, result_bck, result_prob)
            for key, val in step_losses.items():
                losses[key].append(val)
        # Outputs
        apc_all = torch.cat([result_obj['apc'], result_bck['bck'][None]]).transpose(0, 1)
        shp = result_obj['shp']
        mask = self.compute_mask(shp)
        segment_all = torch.argmax(mask, dim=0, keepdim=True)
        mask_oh = torch.zeros_like(mask).scatter_(0, segment_all, 1)[:-1]
        pres = mask_oh.reshape(*mask_oh.shape[:-3], -1).max(-1).values
        shp *= pres[..., None, None, None]
        shp_all = torch.cat([shp, torch.ones([1, *shp.shape[1:]], device=shp.device)]).transpose(0, 1)
        mask = self.compute_mask(shp).transpose(0, 1)
        recon = (mask * apc_all).sum(1)
        segment_all = torch.argmax(mask, dim=1, keepdim=True)
        segment_obj = torch.argmax(mask[:, :-1], dim=1, keepdim=True)
        mask_oh_all = torch.zeros_like(mask).scatter_(1, segment_all, 1)
        mask_oh_obj = torch.zeros_like(mask[:, :-1]).scatter_(1, segment_obj, 1)
        pres_all = torch.cat([pres, torch.ones([1, *pres.shape[1:]], device=pres.device)]).transpose(0, 1)
        pres = pres_all[:, :-1]
        if require_extra:
            results = {'apc': apc_all, 'shp': shp_all, 'pres': pres_all, 'recon': recon, 'mask': mask}
            results = {key: (val.clamp(0, 1) * 255).to(torch.uint8) for key, val in results.items()}
        else:
            results = {}
        metrics = self.compute_metrics(
            images, segments, overlaps, mask_oh_all, mask_oh_obj, recon, result_prob['log_prob'], pres)
        losses = {key: torch.stack([loss for loss in val], dim=1) for key, val in losses.items()}
        losses = {key: (step_wt * val).sum(1) for key, val in losses.items()}
        losses['compare'] = -metrics['ll']
        return results, metrics, losses

    @staticmethod
    def get_step_wt(phase_param):
        if phase_param['step_wt'] is None:
            step_wt = torch.ones([1, phase_param['num_steps'] + 1])
        else:
            step_wt = torch.tensor([phase_param['step_wt']]).reshape(1, phase_param['num_steps'] + 1)
        step_wt /= step_wt.sum(1, keepdim=True)
        return step_wt.cuda(non_blocking=True)

    def compute_probabilities(self, images, result_bck, result_obj):
        def compute_log_mask(logits_shp):
            log_shp = nn_func.logsigmoid(logits_shp)
            log1m_shp = log_shp - logits_shp
            zeros = torch.zeros([1, *logits_shp.shape[1:]], device=logits_shp.device)
            return torch.cat([log_shp, zeros]) + torch.cat([zeros, log1m_shp]).cumsum(0)
        def compute_raw_pixel_ll(bck, apc):
            sq_diff = (torch.cat([apc, bck[None]]) - images[None]).pow(2)
            return -0.5 * (self.normal_const + self.normal_invvar * sq_diff).sum(-3, keepdim=True)
        log_mask = compute_log_mask(result_obj['logits_shp'])
        raw_pixel_ll = compute_raw_pixel_ll(result_bck['bck'], result_obj['apc'])
        log_prob = log_mask + raw_pixel_ll
        gamma = nn_func.softmax(log_prob, dim=0)
        log_gamma = nn_func.log_softmax(log_prob, dim=0)
        return {'log_prob': log_prob, 'gamma': gamma, 'log_gamma': log_gamma}

    @staticmethod
    def compute_mask(shp):
        ones = torch.ones([1, *shp.shape[1:]], device=shp.device)
        return torch.cat([shp, ones]) * torch.cat([ones, 1 - shp]).cumprod(0)

    @staticmethod
    def permute(x, indices):
        indices_reshape = indices.reshape([*indices.shape] + [1] * (x.ndim - indices.ndim))
        return torch.gather(x, 0, indices_reshape.expand(-1, -1, *x.shape[2:]))

    def compute_indices(self, images, result_obj, eps=1e-5):
        sq_diffs = (result_obj['apc'] - images[None]).pow(2).sum(-3, keepdim=True).detach()
        visibles = result_obj['shp'].clone().detach()
        coefs = torch.ones(visibles.shape[:-2], device=visibles.device)
        if coefs.shape[0] == 0:
            return coefs.type(torch.long)
        indices_list = []
        for _ in range(coefs.shape[0]):
            vis_sq_diffs = (visibles * sq_diffs).sum([-2, -1])
            vis_areas = visibles.sum([-2, -1])
            vis_max_vals = visibles.reshape(*visibles.shape[:-2], -1).max(-1).values
            scores = torch.exp(-0.5 * self.normal_invvar * vis_sq_diffs / (vis_areas + eps))
            scaled_scores = coefs * (vis_max_vals * scores + 1)
            indices = torch.argmax(scaled_scores, dim=0, keepdim=True)
            indices_list.append(indices)
            vis = torch.gather(visibles, 0, indices[..., None, None].expand(-1, -1, *visibles.shape[2:]))
            visibles *= 1 - vis
            coefs.scatter_(0, indices, -1)
        return torch.cat(indices_list)

    def adjust_order(self, images, result_obj, states_obj):
        indices = self.compute_indices(images, result_obj)
        for key, val in result_obj.items():
            result_obj[key] = self.permute(val, indices)
        states_obj = [n.reshape(indices.shape[0], -1, *n.shape[1:]) for n in states_obj]
        states_obj = [self.permute(n, indices) for n in states_obj]
        states_obj = [n.reshape(-1, *n.shape[2:]) for n in states_obj]
        return tuple(states_obj)

    def compute_step_losses(self, images, result_bck, result_prob):
        # Loss ELBO
        log_prob = result_prob['log_prob']
        gamma = result_prob['gamma'].detach()
        log_gamma = result_prob['log_gamma'].detach()
        loss_elbo = (gamma * (log_gamma - log_prob)).sum([0, *range(2, gamma.ndim)])
        # Loss back prior
        bck_prior = images.reshape(*images.shape[:-2], -1).median(-1).values[..., None, None]
        sq_diff = (result_bck['bck'] - bck_prior).pow(2)
        loss_bck_prior = 0.5 * self.normal_invvar * sq_diff.sum([*range(1, sq_diff.ndim)])
        # Losses
        losses = {'elbo': loss_elbo, 'bck_prior': loss_bck_prior}
        return losses

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

    def compute_metrics(self, images, segments, overlaps, mask_oh_all, mask_oh_obj, recon, log_prob, pres):
        segments_obj = segments[:, :-1]
        # ARI
        segments_obj_sel = segments_obj if self.seg_overlap else segments_obj * (1 - overlaps)
        ari_all = self.compute_ari(segments_obj_sel, mask_oh_all)
        ari_obj = self.compute_ari(segments_obj_sel, mask_oh_obj)
        # MSE
        sq_diff = (recon - images).pow(2)
        mse = sq_diff.mean([*range(1, sq_diff.ndim)])
        # Log-likelihood
        pixel_ll = torch.logsumexp(log_prob, dim=0)
        ll = pixel_ll.sum([*range(1, pixel_ll.ndim)])
        # Count
        count_true = segments_obj.reshape(*segments_obj.shape[:-3], -1).max(-1).values.sum(1)
        count_pred = pres.sum(1)
        count_acc = torch.eq(count_true, count_pred).to(dtype=torch.float)
        metrics = {'ari_all': ari_all, 'ari_obj': ari_obj, 'mse': mse, 'll': ll, 'count': count_acc}
        return metrics


def get_model(config):
    net = Model(config).cuda()
    return nn.DataParallel(net)
