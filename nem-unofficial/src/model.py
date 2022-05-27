import math
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
from network import Updater, Decoder


class ModelBase(nn.Module):

    def __init__(self, config):
        super(ModelBase, self).__init__()
        # Hyperparameters
        self.noise_prob = config['noise_prob']
        self.seg_overlap = config['seg_overlap']
        # Neural networks
        self.upd = Updater(config)
        self.dec = Decoder(config)

    def forward(self, data, phase_param, require_extra=True, eps=1e-10):
        num_slots = phase_param['num_slots']
        num_iters = phase_param['num_iters']
        iter_wt = self.get_iter_wt(phase_param)
        data = {key: val.cuda(non_blocking=True) for key, val in data.items()}
        data = self.convert_data(data, num_iters)
        images_seq = data['image'].float() / 255
        segments = data['segment'][:, None, None].long()
        scatter_shape = [segments.shape[0], segments.max() + 1, *segments.shape[2:]]
        segments = torch.zeros(scatter_shape, device=segments.device).scatter_(1, segments, 1)
        overlaps = torch.gt(data['overlap'][:, None, None], 1).float()
        # Initializations
        log_mask = -math.log(num_slots)
        apc_prior = images_seq.transpose(1, 2)
        apc_prior = apc_prior.reshape(*apc_prior.shape[:2], -1).median(-1).values
        apc_prior = apc_prior[..., None, None].expand(-1, -1, *images_seq.shape[-2:])
        apc = apc_prior[None].expand(num_slots, -1, -1, -1, -1)
        images_seq = images_seq.transpose(0, 1).contiguous()
        states = torch.zeros([num_slots * images_seq.shape[1], self.upd.state_size], device=images_seq.device)
        gamma = torch.randn_like(apc).abs_().add_(eps)
        gamma /= gamma.sum(dim=0, keepdim=True)
        losses = {key: [] for key in ['elbo', 'kld']}
        raw_pixel_ll, log_gamma = None, None
        # Iterations
        for idx_iter in range(num_iters):
            noisy_images = self.add_noise(images_seq[idx_iter])
            targets = images_seq[idx_iter + 1]
            upd_in = gamma * (noisy_images[None] - apc)
            outputs, states = self.upd(upd_in, states)
            apc_base = self.dec(outputs, num_slots)
            apc, raw_pixel_ll, kld = self.compute_basic_values(targets, apc_prior, apc_base)
            log_prob = raw_pixel_ll + torch.full_like(raw_pixel_ll, log_mask)
            gamma = torch.softmax(log_prob, dim=0).detach()
            log_gamma = torch.log_softmax(log_prob, dim=0).detach()
            iter_losses = {
                'elbo': gamma * (log_gamma - log_prob),
                'kld': (1 - gamma) * kld,
            }
            for key, val in iter_losses.items():
                losses[key].append(val.sum([0, *range(2, val.dim())]))
        # Outputs
        losses = {key: torch.stack(val, dim=1) for key, val in losses.items()}
        losses = {key: (iter_wt * val).sum(1) for key, val in losses.items()}
        apc = apc.transpose(0, 1)
        mask = gamma.transpose(0, 1)
        recon = (mask * apc).sum(1)
        log_prob = raw_pixel_ll + log_gamma
        segment = torch.argmax(mask, dim=1, keepdim=True)
        mask_oh = torch.zeros_like(mask).scatter_(1, segment, 1)
        pres = mask_oh.reshape(*mask_oh.shape[:-3], -1).max(-1).values
        if require_extra:
            results = {'apc': apc, 'mask': mask, 'pres': pres, 'recon': recon}
            results = {key: (val.clamp(0, 1) * 255).to(torch.uint8) for key, val in results.items()}
        else:
            results = {}
        metrics = self.compute_metrics(images_seq[-1], segments, overlaps, pres, mask_oh, recon, log_prob)
        losses['compare'] = -metrics['ll']
        return results, metrics, losses

    def add_noise(self, images):
        raise NotImplementedError

    def compute_basic_values(self, targets, apc_prior, apc_base):
        raise NotImplementedError

    @staticmethod
    def get_iter_wt(phase_param):
        if phase_param['iter_wt'] is None:
            iter_wt = torch.ones([1, phase_param['num_iters']])
        else:
            iter_wt = torch.tensor([phase_param['iter_wt']]).reshape(1, phase_param['num_iters'])
        iter_wt /= iter_wt.sum(1, keepdim=True)
        return iter_wt.cuda(non_blocking=True)

    @staticmethod
    def convert_data(data, num_iters):
        for key, val in data.items():
            if val.shape[1] == 1:
                val = val.expand(-1, num_iters + 1, *([-1] * (val.dim() - 2)))
            else:
                val = val[:, :num_iters + 1]
            if key != 'image':
                val = val[:, -1]
            data[key] = val
        return data

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

    def compute_metrics(self, images, segments, overlaps, pres, mask_oh, recon, log_prob):
        segments_obj = segments[:, :-1]
        # ARI
        segments_obj_sel = segments_obj if self.seg_overlap else segments_obj * (1 - overlaps)
        ari_obj = self.compute_ari(segments_obj_sel, mask_oh)
        # MSE
        sq_diff = (recon - images).pow(2)
        mse = sq_diff.mean([*range(1, sq_diff.dim())])
        # Log-likelihood
        pixel_ll = torch.logsumexp(log_prob, dim=0)
        ll = pixel_ll.sum([*range(1, pixel_ll.dim())])
        # Count
        count_true = segments_obj.reshape(*segments_obj.shape[:-3], -1).max(-1).values.sum(1)
        count_pred = pres.sum(1)
        count_acc = torch.eq(count_true, count_pred).to(dtype=torch.float)
        metrics = {'ari_obj': ari_obj, 'mse': mse, 'll': ll, 'count': count_acc}
        return metrics


class ModelBinary(ModelBase):

    def __init__(self, config):
        super(ModelBinary, self).__init__(config)

    def add_noise(self, images):
        noise_mask = torch.bernoulli(
            torch.full([images.shape[0], 1, *images.shape[2:]], self.noise_prob, device=images.device))
        return images + noise_mask - 2 * images * noise_mask

    def compute_basic_values(self, targets, apc_prior, logits_apc):
        def compute_prob(x):
            return (log_apc + (x[None] - 1) * logits_apc).sum(-3, keepdim=True)
        apc = torch.sigmoid(logits_apc)
        log_apc = nn_func.logsigmoid(logits_apc)
        raw_pixel_ll = compute_prob(targets)
        kld = -compute_prob(apc_prior)
        return apc, raw_pixel_ll, kld


class ModelReal(ModelBase):

    def __init__(self, config):
        super(ModelReal, self).__init__(config)
        self.normal_invvar = 1 / pow(config['normal_scale'], 2)
        self.normal_const = math.log(2 * math.pi / self.normal_invvar)

    def add_noise(self, images):
        noise_mask = torch.bernoulli(
            torch.full([images.shape[0], 1, *images.shape[2:]], self.noise_prob, device=images.device))
        noise_value = torch.rand_like(images)
        return noise_mask * noise_value + (1 - noise_mask) * images

    def compute_basic_values(self, targets, apc_prior, normalized_apc):
        def compute_prob(x):
            return -0.5 * (self.normal_const + self.normal_invvar * (apc - x[None]).pow(2)).sum(-3, keepdim=True)
        apc = (normalized_apc + 1) * 0.5
        raw_pixel_ll = compute_prob(targets)
        kld = -compute_prob(apc_prior)
        return apc, raw_pixel_ll, kld


def get_model(config):
    if config['normal_scale'] is None:
        net = ModelBinary(config)
    else:
        net = ModelReal(config)
    return nn.DataParallel(net.cuda())
