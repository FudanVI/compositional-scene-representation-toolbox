import math
from typing import Dict, Tuple

import torch
from building_block import compile_disable, compile_mode
from decoder import Decoder
from encoder import Encoder
from omegaconf import DictConfig
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli


def reparameterize_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    noise = torch.randn_like(std)
    latent = mu + std * noise
    return latent


class Model(torch.nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.coef_normal = 0.125 / pow(cfg.run_training.loss.normal_scale, 2)
        self.pres_alpha = cfg.run_training.loss.pres_alpha
        self.max_shadow_ratio = cfg.run_training.loss.max_shadow_ratio
        self.coef_px = cfg.dataset.image_shape[0] * cfg.dataset.image_shape[1]
        self.seg_overlap = cfg.dataset.seg_overlap
        self.enc = Encoder(cfg)
        self.dec = Decoder(cfg)

    def forward(
        self,
        image: torch.Tensor,
        temp_pres: torch.Tensor,
        temp_shp: torch.Tensor,
        noise_scale: torch.Tensor,
        noise_min: torch.Tensor,
        noise_max: torch.Tensor,
        ratio_stick_breaking: torch.Tensor,
        num_slots: int,
    ) -> Dict[str, torch.Tensor]:
        torch.cuda.nvtx.range_push('encoder')
        outputs_enc = self.enc(image, num_slots)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('sample')
        view_latent, bck_latent, obj_latent, logits_pres, pres = \
            self.sample_latent(outputs_enc, temp_pres)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('decoder')
        outputs_dec = self.dec(view_latent, bck_latent, obj_latent, pres,
                               logits_pres, temp_shp, noise_scale, noise_min,
                               noise_max, ratio_stick_breaking)
        torch.cuda.nvtx.range_pop()
        outputs = {
            **outputs_enc,
            **outputs_dec,
            'image': image,
            'pres': torch.ge(pres.squeeze(-1), 0.5).to(pres.dtype),
            'logits_pres': logits_pres,
            'view_latent': view_latent,
            'bck_latent': bck_latent,
            'obj_latent': obj_latent,
        }
        return outputs

    @staticmethod
    @torch.compile(mode=compile_mode, disable=compile_disable)
    def sample_latent(
        outputs_enc: Dict[str, torch.Tensor],
        temp_pres: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        view_latent = reparameterize_normal(
            outputs_enc['view_mu'],
            outputs_enc['view_logvar'],
        )
        bck_latent = reparameterize_normal(
            outputs_enc['bck_mu'],
            outputs_enc['bck_logvar'],
        )
        obj_latent = reparameterize_normal(
            outputs_enc['obj_mu'],
            outputs_enc['obj_logvar'],
        )
        logits_pres = LogitRelaxedBernoulli(
            temperature=temp_pres,
            logits=outputs_enc['logits_zeta'],
        ).rsample()
        pres = torch.sigmoid(logits_pres)
        return view_latent, bck_latent, obj_latent, logits_pres, pres

    @torch.compile(mode=compile_mode, disable=compile_disable)
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        loss_coef: Dict[str, torch.Tensor],
        eps: float = 1.0e-5,
    ) -> Dict[str, torch.Tensor]:
        def compute_kld_normal(
            mu: torch.Tensor,
            logvar: torch.Tensor,
        ) -> torch.Tensor:
            kld = 0.5 * (mu.square() + logvar.exp() - logvar - 1)
            return kld

        def compute_kld_pres(
            tau1: torch.Tensor,
            tau2: torch.Tensor,
            zeta: torch.Tensor,
            logits_zeta: torch.Tensor,
        ) -> torch.Tensor:
            tau1 = tau1.squeeze(-1)
            tau2 = tau2.squeeze(-1)
            zeta = zeta.squeeze(-1)
            logits_zeta = logits_zeta.squeeze(-1)
            coef_alpha = self.pres_alpha / logits_zeta.shape[1]
            psi1 = torch.digamma(tau1)
            psi2 = torch.digamma(tau2)
            psi12 = torch.digamma(tau1 + tau2)
            kld_1 = torch.lgamma(tau1 + tau2) - torch.lgamma(tau1) - \
                torch.lgamma(tau2) - math.log(coef_alpha)
            kld_2 = (tau1 - coef_alpha) * psi1 + (tau2 - 1) * psi2 - \
                (tau1 + tau2 - coef_alpha - 1) * psi12
            log_zeta = torch.nn.functional.logsigmoid(logits_zeta)
            log1m_zeta = log_zeta - logits_zeta
            kld_3 = zeta * (log_zeta - psi1) + \
                (1 - zeta) * (log1m_zeta - psi2) + psi12
            kld = kld_1 + kld_2 + kld_3
            return kld

        image = outputs['image']
        apc_all = torch.cat(
            [outputs['apc'], outputs['bck'][:, :, None]],
            dim=2,
        )
        mask_all = torch.cat(
            [outputs['mask_obj'], outputs['mask_bck'][:, :, None]],
            dim=2,
        )
        raw_pixel_ll = -self.coef_normal * \
            (apc_all - image[:, :, None]).square().sum(-1, keepdim=True)
        log_mask_all = torch.log(mask_all * (1 - 2 * eps) + eps)
        masked_pixel_ll = log_mask_all + raw_pixel_ll
        loss_nll_sm = -torch.logsumexp(masked_pixel_ll, dim=2)
        loss_nll_ws = self.coef_normal * \
            (outputs['recon_aux'] - image).square()
        loss_nll_ws_imp = self.coef_normal * \
            (outputs['recon_aux_imp'] - image).square()
        ratio_imp_sdw = loss_coef['ratio_imp_sdw']
        loss_nll_ws = ratio_imp_sdw * loss_nll_ws_imp + \
            (1 - ratio_imp_sdw) * loss_nll_ws
        ratio_mixture = loss_coef['ratio_mixture']
        loss_nll = ratio_mixture * loss_nll_sm + \
            (1 - ratio_mixture) * loss_nll_ws
        sdw_shp_hard = torch.gt(outputs['sdw_shp'],
                                0.1).to(outputs['sdw_shp'].dtype)
        sdw_ratio = (outputs['shp_imp'] * sdw_shp_hard).flatten(
            start_dim=-3).sum(-1) / (
                outputs['shp_imp'].flatten(start_dim=-3).sum(-1) + eps)
        reg_sdw_ratio = torch.gt(
            sdw_ratio, self.max_shadow_ratio)[..., None, None, None] * \
                torch.abs(outputs['sdw_logits_shp'] + 3)
        reg_sdw_ratio = (1 - ratio_imp_sdw) * reg_sdw_ratio
        losses = {
            'nll': loss_nll,
            'kld_view': compute_kld_normal(
                outputs['view_mu'],
                outputs['view_logvar'],
            ),
            'kld_bck': compute_kld_normal(
                outputs['bck_mu'],
                outputs['bck_logvar'],
            ),
            'kld_obj': compute_kld_normal(
                outputs['obj_mu'],
                outputs['obj_logvar'],
            ),
            'kld_pres': compute_kld_pres(
                outputs['tau1'],
                outputs['tau2'],
                outputs['zeta'],
                outputs['logits_zeta'],
            ),
            'reg_bck': self.coef_normal * (outputs['bck'] - image).square(),
            'reg_pres': self.coef_px * outputs['zeta'].detach() * \
                outputs['logits_zeta'],
            'reg_shp': torch.abs(outputs['logits_shp_imp'] + 3),
            'reg_sdw': torch.abs(outputs['sdw_logits_shp'] + 3),
            'reg_sdw_ratio': reg_sdw_ratio,
        }
        losses = {
            key: val.flatten(start_dim=1).sum(1).mean()
            for key, val in losses.items()
        }
        losses['opt'] = torch.stack(
            [loss_coef[key] * val for key, val in losses.items()]).sum()
        return losses

    @torch.no_grad()
    @torch.compile(mode=compile_mode, disable=compile_disable)
    def compute_metrics(
        self,
        data: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        segment = data['segment']
        overlap = data['overlap']
        segment_obj = segment[:, :, :-1].contiguous()

        # ARI
        if self.seg_overlap:
            segment_all_sel = segment
            segment_obj_sel = segment_obj
        else:
            segment_all_sel = segment * (1 - overlap)
            segment_obj_sel = segment_obj * (1 - overlap)
        mask_all = torch.cat(
            [outputs['mask_obj'], outputs['mask_bck'][:, :, None]],
            dim=2,
        ).squeeze(-1)
        mask_obj = outputs['mask_obj'].squeeze(-1)

        def compute_ari_values(
            mask_true: torch.Tensor,
            mask_pred_soft: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            mask_true_s = mask_true.flatten(end_dim=1)
            mask_true_m = mask_true.transpose(1, 2).contiguous()
            mask_pred_index = torch.argmax(mask_pred_soft, dim=2, keepdim=True)
            mask_pred = torch.zeros_like(mask_pred_soft).scatter_(
                2, mask_pred_index, 1)
            mask_pred_s = mask_pred.flatten(end_dim=1)
            mask_pred_m = mask_pred.transpose(1, 2).contiguous()

            def compute_ari(
                mask_true: torch.Tensor,
                mask_pred: torch.Tensor,
            ) -> torch.Tensor:
                mask_true = mask_true.flatten(start_dim=2)
                mask_pred = mask_pred.flatten(start_dim=2)
                num_px = mask_true.flatten(start_dim=1).sum(1, keepdims=True)
                mat = torch.einsum('bin,bjn->bij', mask_true, mask_pred)
                sum_row = mat.sum(1)
                sum_col = mat.sum(2)

                def comb2(x: torch.Tensor) -> torch.Tensor:
                    x = x * (x - 1)
                    x = x.flatten(start_dim=1).sum(1)
                    return x

                comb_mat = comb2(mat)
                comb_row = comb2(sum_row)
                comb_col = comb2(sum_col)
                comb_num = comb2(num_px)
                comb_prod = (comb_row * comb_col) / comb_num
                comb_mean = 0.5 * (comb_row + comb_col)
                diff = comb_mean - comb_prod
                score = (comb_mat - comb_prod) / diff
                invalid = torch.logical_or(
                    torch.eq(comb_num, 0),
                    torch.eq(diff, 0),
                )
                score.masked_fill_(invalid, 1)
                return score

            ari_all_s = compute_ari(mask_true_s, mask_pred_s).view(
                mask_true.shape[:2]).mean(1)
            ari_all_m = compute_ari(mask_true_m, mask_pred_m)
            return ari_all_s, ari_all_m

        ari_all_s, ari_all_m = compute_ari_values(segment_all_sel, mask_all)
        ari_obj_s, ari_obj_m = compute_ari_values(segment_obj_sel, mask_obj)

        # MSE
        mse = 0.25 * (outputs['recon'] -
                      outputs['image']).square().flatten(start_dim=1).mean(1)

        # Count
        count_true = segment_obj.transpose(
            1, 2).flatten(start_dim=2).max(-1).values.sum(-1)
        count_pred = outputs['pres'].sum(1)
        count_acc = torch.eq(count_true, count_pred).to(dtype=count_true.dtype)

        # Outputs
        metrics = {
            'ari_all_s': ari_all_s,
            'ari_all_m': ari_all_m,
            'ari_obj_s': ari_obj_s,
            'ari_obj_m': ari_obj_m,
            'mse': mse,
            'count': count_acc,
        }
        metrics = {key: val.mean() for key, val in metrics.items()}
        return metrics

    @torch.no_grad()
    @torch.compile(mode=compile_mode, disable=compile_disable)
    def convert_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        recon = torch.clamp((outputs['recon'] + 1) * 0.5, 0, 1)
        bck_sdw = torch.clamp((outputs['bck'] + 1) * 0.5, 0, 1)
        apc = torch.clamp(
            (torch.cat([outputs['apc'], outputs['bck_imp'][:, :, None]], dim=2)
             + 1) * 0.5, 0, 1)
        ones = torch.ones(
            [*outputs['shp'].shape[:2], 1, *outputs['shp'].shape[3:]],
            device=outputs['shp'].device)
        shp = torch.cat([outputs['shp'], ones], dim=2)
        shp_imp = torch.cat([outputs['shp_imp'], ones], dim=2)
        ones = torch.ones([outputs['pres'].shape[0], 1],
                          device=outputs['pres'].device)
        pres = torch.cat([outputs['pres'], ones], dim=1)
        sdw_apc = torch.clamp((outputs['sdw_apc'] + 1) * 0.5, 0, 1)
        sdw_shp = outputs['sdw_shp']
        mask = torch.cat(
            [outputs['mask_obj'], outputs['mask_bck'][:, :, None]], dim=2)
        outputs_extra = {
            'recon': recon,
            'bck_sdw': bck_sdw,
            'apc': apc,
            'shp': shp,
            'shp_imp': shp_imp,
            'pres': pres,
            'sdw_apc': sdw_apc,
            'sdw_shp': sdw_shp,
            'mask': mask,
        }
        outputs_extra = {
            key: (val * 255).to(torch.uint8)
            for key, val in outputs_extra.items()
        }
        key_list = [
            'logits_pres', 'view_latent', 'bck_latent', 'obj_latent',
            'log_ord', 'trs'
        ]
        outputs = {key: val for key, val in outputs.items() if key in key_list}
        outputs.update(outputs_extra)
        return outputs
