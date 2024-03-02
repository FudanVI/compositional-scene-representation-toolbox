from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from building_block import (ConvTBlock, ConvTLayer, LinearBlock, Permute,
                            SinusoidPosEmbedLayer, compile_disable,
                            compile_mode, get_activation, get_grid)
from omegaconf import DictConfig
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli


def select_by_index(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    x_ndim = x.ndim
    index_ndim = index.ndim
    index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
    index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
    x = torch.gather(x, index_ndim - 1, index)
    return x


class DecoderImageBasic(torch.nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_shape: Tuple[int, Union[int, None], Union[int, None]],
        channel_list_rev: List[int],
        kernel_list_rev: List[int],
        stride_list_rev: List[int],
        feature_list_rev: List[int],
        activation: str,
    ) -> None:
        net_conv = ConvTBlock(
            out_shape=out_shape,
            channel_list_rev=channel_list_rev,
            kernel_list_rev=kernel_list_rev,
            stride_list_rev=stride_list_rev,
            act_inner=activation,
            act_out=None,
        )
        net_linear = LinearBlock(
            in_features=in_features,
            feature_list=[*reversed(feature_list_rev)] +
            [np.prod(net_conv.in_shape)],
            act_inner=activation,
            act_out=None if len(channel_list_rev) == 0 else activation,
        )
        net_list = [
            net_linear,
            torch.nn.Unflatten(-1, net_conv.in_shape),
            net_conv,
            Permute([0, 2, 3, 1]),
        ]
        super().__init__(*net_list)


class DecoderImageComplex(torch.nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_shape: Tuple[int, Union[int, None], Union[int, None]],
        channel_list_rev: List[int],
        kernel_list_rev: List[int],
        stride_list_rev: List[int],
        feature_list_rev: List[int],
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: str,
    ) -> None:
        net_conv_out = ConvTBlock(
            out_shape=out_shape,
            channel_list_rev=channel_list_rev,
            kernel_list_rev=kernel_list_rev,
            stride_list_rev=stride_list_rev,
            act_inner=activation,
            act_out=None,
        )
        net_conv_cvt = torch.compile(
            ConvTLayer(
                in_channels=d_model,
                out_channels=net_conv_out.in_shape[0],
                kernel_size=3,
                stride=1,
                activation=activation,
            ),
            mode=compile_mode,
            disable=compile_disable,
        )
        net_transformer = torch.compile(
            torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=0,
                    activation=get_activation(activation),
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=num_layers,
                enable_nested_tensor=False,
            ),
            mode=compile_mode,
            disable=compile_disable,
        )
        net_pos_embed = SinusoidPosEmbedLayer(
            ht=net_conv_out.in_shape[1],
            wd=net_conv_out.in_shape[2],
            out_features=d_model,
        )
        net_linear = LinearBlock(
            in_features=in_features,
            feature_list=[*reversed(feature_list_rev)] + [d_model],
            act_inner=activation,
            act_out=activation,
        )
        net_list = [
            net_linear,
            torch.nn.Unflatten(-1, [net_linear.out_features, 1, 1]),
            net_pos_embed,
            torch.nn.Flatten(start_dim=2),
            Permute([0, 2, 1]),
            net_transformer,
            Permute([0, 2, 1]),
            torch.nn.Unflatten(-1, net_conv_out.in_shape[1:]),
            net_conv_cvt,
            net_conv_out,
            Permute([0, 2, 3, 1]),
        ]
        super().__init__(*net_list)


class Decoder(torch.nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        image_ht, image_wd, image_ch = cfg.dataset.image_shape
        self.register_buffer(
            'pos_grid_noise',
            get_grid(image_ht, image_wd)[:, None, None],
            persistent=False,
        )
        self.image_ch = image_ch
        self.use_shadow = cfg.model.use_shadow
        self.max_shadow_val = cfg.model.max_shadow_val

        # Background
        in_features = cfg.model.latent_view_size + cfg.model.latent_bck_size
        if cfg.model.dec_bck.use_complex:
            self.net_bck = DecoderImageComplex(
                in_features=in_features,
                out_shape=[image_ch, image_ht, image_wd],
                channel_list_rev=cfg.model.dec_bck.channel_list_rev,
                kernel_list_rev=cfg.model.dec_bck.kernel_list_rev,
                stride_list_rev=cfg.model.dec_bck.stride_list_rev,
                feature_list_rev=cfg.model.dec_bck.feature_list_rev,
                num_layers=cfg.model.dec_bck.num_layers,
                d_model=cfg.model.dec_bck.d_model,
                nhead=cfg.model.dec_bck.nhead,
                dim_feedforward=cfg.model.dec_bck.dim_feedforward,
                activation=cfg.model.dec_bck.activation,
            )
        else:
            self.net_bck = DecoderImageBasic(
                in_features=in_features,
                out_shape=[image_ch, image_ht, image_wd],
                channel_list_rev=cfg.model.dec_bck.channel_list_rev,
                kernel_list_rev=cfg.model.dec_bck.kernel_list_rev,
                stride_list_rev=cfg.model.dec_bck.stride_list_rev,
                feature_list_rev=cfg.model.dec_bck.feature_list_rev,
                activation=cfg.model.dec_bck.activation,
            )

        # Objects
        in_features = cfg.model.latent_view_size + cfg.model.latent_obj_size
        out_channels = image_ch + 3 if self.use_shadow else image_ch + 1
        self.net_obj_misc = LinearBlock(
            in_features=in_features,
            feature_list=cfg.model.dec_obj_misc.feature_list + [3],
            act_inner=cfg.model.dec_obj_misc.activation,
            act_out=None,
        )
        if cfg.model.dec_obj_img.use_complex:
            self.net_obj_img = DecoderImageComplex(
                in_features=in_features,
                out_shape=[out_channels, image_ht, image_wd],
                channel_list_rev=cfg.model.dec_obj_img.channel_list_rev,
                kernel_list_rev=cfg.model.dec_obj_img.kernel_list_rev,
                stride_list_rev=cfg.model.dec_obj_img.stride_list_rev,
                feature_list_rev=cfg.model.dec_obj_img.feature_list_rev,
                num_layers=cfg.model.dec_obj_img.num_layers,
                d_model=cfg.model.dec_obj_img.d_model,
                nhead=cfg.model.dec_obj_img.nhead,
                dim_feedforward=cfg.model.dec_obj_img.dim_feedforward,
                activation=cfg.model.dec_obj_img.activation,
            )
        else:
            self.net_obj_img = DecoderImageBasic(
                in_features=in_features,
                out_shape=[out_channels, image_ht, image_wd],
                channel_list_rev=cfg.model.dec_obj_img.channel_list_rev,
                kernel_list_rev=cfg.model.dec_obj_img.kernel_list_rev,
                stride_list_rev=cfg.model.dec_obj_img.stride_list_rev,
                feature_list_rev=cfg.model.dec_obj_img.feature_list_rev,
                activation=cfg.model.dec_obj_img.activation,
            )

    def forward(
        self,
        view_latent: torch.Tensor,  # [B, V, D_v]
        bck_latent: torch.Tensor,  # [B, D_b]
        obj_latent: torch.Tensor,  # [B, S, D_o]
        pres: torch.Tensor,  # [B, S, 1]
        logits_pres: torch.Tensor,  # [B, S, 1]
        temp_shp: torch.Tensor,
        noise_scale: torch.Tensor,
        noise_min: torch.Tensor,
        noise_max: torch.Tensor,
        ratio_stick_breaking: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_views = view_latent.shape[:2]
        num_slots = obj_latent.shape[1]
        torch.cuda.nvtx.range_push('latent')
        full_bck_latent, full_obj_latent = self.sample_latent(
            view_latent, bck_latent, obj_latent)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('net_bck')
        bck_imp = self.net_bck(full_bck_latent).unflatten(
            0, [batch_size, num_views])  # [B, V, H, W, C]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('net_obj_misc')
        x_misc = self.net_obj_misc(full_obj_latent).unflatten(
            0, [batch_size, num_views, num_slots])  # [B, V, S, 3]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('net_obj_img')
        x_img = self.net_obj_img(full_obj_latent).unflatten(
            0, [batch_size, num_views, num_slots])
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('attr')
        log_ord, trs, apc, shp_imp, log_shp_imp, logits_shp_imp = \
            self.compute_attr(x_misc, x_img, temp_shp, ratio_stick_breaking)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('sdw')
        if self.use_shadow:
            x_sdw = x_img[..., self.image_ch + 1:]  # [B, V, S, H, W, 2]
            sdw_logits_apc_raw = x_sdw[..., :-1].contiguous()
            sdw_logits_shp_raw = x_sdw[..., -1:].contiguous()
            sdw_apc, sdw_shp, sdw_log_shp, sdw_logits_shp = \
                self.compute_sdw_pre(bck_imp, sdw_logits_apc_raw,
                                     sdw_logits_shp_raw, temp_shp,
                                     ratio_stick_breaking)
            sdw_mask_bck, sdw_mask_obj = self.compute_mask(
                sdw_shp,
                sdw_log_shp,
                pres,
                logits_pres,
                torch.zeros_like(log_ord),
                ratio_stick_breaking=None,
            )
            bck, apc_aux, shp, log_shp = self.compute_sdw_post(
                trs, bck_imp, apc, shp_imp, log_shp_imp, sdw_mask_bck,
                sdw_mask_obj, sdw_apc, sdw_shp, sdw_log_shp, sdw_logits_shp,
                noise_scale, noise_min, noise_max)
        else:
            sdw_apc = torch.zeros_like(apc)
            sdw_shp = sdw_logits_shp = torch.zeros_like(shp_imp)
            bck = bck_imp
            apc_aux = apc
            shp = shp_imp
            log_shp = log_shp_imp
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('composite')
        mask_bck, mask_obj = self.compute_mask(
            shp,
            log_shp,
            pres,
            logits_pres,
            log_ord,
            ratio_stick_breaking,
        )
        mask_bck_imp, mask_obj_imp = self.compute_mask(
            shp_imp,
            log_shp_imp,
            pres,
            logits_pres,
            log_ord,
            ratio_stick_breaking,
        )
        recon, recon_aux, recon_aux_imp = self.compute_recon(
            apc, apc_aux, bck, bck_imp, mask_obj, mask_obj_imp, mask_bck,
            mask_bck_imp)
        torch.cuda.nvtx.range_pop()
        outputs = {
            'recon': recon,
            'recon_aux': recon_aux,
            'recon_aux_imp': recon_aux_imp,
            'mask_bck': mask_bck,
            'mask_obj': mask_obj,
            'log_ord': log_ord,
            'trs': trs,
            'bck': bck,
            'bck_imp': bck_imp,
            'apc': apc,
            'apc_aux': apc_aux,
            'shp': shp,
            'shp_imp': shp_imp,
            'sdw_apc': sdw_apc,
            'sdw_shp': sdw_shp,
            'logits_shp_imp': logits_shp_imp,
            'sdw_logits_shp': sdw_logits_shp,
        }
        return outputs

    @staticmethod
    @torch.compile(mode=compile_mode, disable=compile_disable)
    def sample_latent(
        view_latent: torch.Tensor,
        bck_latent: torch.Tensor,
        obj_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_views = view_latent.shape[1]
        num_slots = obj_latent.shape[1]
        full_bck_latent = torch.cat(
            [
                view_latent,
                bck_latent[:, None].expand(-1, num_views, -1),
            ],
            dim=-1,
        ).flatten(end_dim=1)  # [B * V, D_v + D_b]
        full_obj_latent = torch.cat(
            [
                view_latent[:, :, None].expand(-1, -1, num_slots, -1),
                obj_latent[:, None].expand(-1, num_views, -1, -1),
            ],
            dim=-1,
        ).flatten(end_dim=2)  # [B * V * S, D_v + D_o]
        return full_bck_latent, full_obj_latent

    @torch.compile(mode=compile_mode, disable=compile_disable)
    def compute_attr(
        self,
        x_misc: torch.Tensor,
        x_img: torch.Tensor,
        temp_shp: torch.Tensor,
        ratio_stick_breaking: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        log_ord = x_misc[..., :1].contiguous() / temp_shp  # [B, V, S, 1]
        trs = torch.tanh(x_misc[..., 1:].contiguous())  # [B, V, S, 2]
        apc = x_img[..., :self.image_ch].contiguous()  # [B, V, S, H, W, C]
        logits_shp_imp = x_img[..., self.image_ch:self.image_ch +
                               1].contiguous() - 3  # [B, V, S, H, W, 1]
        logits_shp_imp_rand = LogitRelaxedBernoulli(
            temperature=temp_shp, logits=logits_shp_imp).rsample()
        logits_shp_imp = ratio_stick_breaking * logits_shp_imp + (
            1 - ratio_stick_breaking) * logits_shp_imp_rand
        shp_imp = torch.sigmoid(logits_shp_imp)  # [B, V, S, H, W, 1]
        log_shp_imp = torch.nn.functional.logsigmoid(
            logits_shp_imp)  # [B, V, S, H, W, 1]
        return log_ord, trs, apc, shp_imp, log_shp_imp, logits_shp_imp

    @torch.compile(mode=compile_mode, disable=compile_disable)
    def compute_sdw_pre(
        self,
        bck_imp: torch.Tensor,
        sdw_logits_apc_raw: torch.Tensor,
        sdw_logits_shp_raw: torch.Tensor,
        temp_shp: torch.Tensor,
        ratio_stick_breaking: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sdw_apc_raw = torch.sigmoid(sdw_logits_apc_raw)
        sdw_apc_raw = sdw_apc_raw * self.max_shadow_val + \
            (1 - self.max_shadow_val)
        sdw_apc = (bck_imp[:, :, None] + 1) * sdw_apc_raw - 1
        sdw_logits_shp = sdw_logits_shp_raw - 3
        sdw_logits_shp_rand = LogitRelaxedBernoulli(
            temperature=temp_shp, logits=sdw_logits_shp).rsample()
        sdw_logits_shp = ratio_stick_breaking * sdw_logits_shp + (
            1 - ratio_stick_breaking) * sdw_logits_shp_rand
        sdw_shp = torch.sigmoid(sdw_logits_shp)
        sdw_log_shp = torch.nn.functional.logsigmoid(sdw_logits_shp)
        return sdw_apc, sdw_shp, sdw_log_shp, sdw_logits_shp

    @torch.compile(mode=compile_mode, disable=compile_disable)
    def compute_sdw_post(
        self,
        trs: torch.Tensor,
        bck_imp: torch.Tensor,
        apc: torch.Tensor,
        shp_imp: torch.Tensor,
        log_shp_imp: torch.Tensor,
        sdw_mask_bck: torch.Tensor,
        sdw_mask_obj: torch.Tensor,
        sdw_apc: torch.Tensor,
        sdw_shp: torch.Tensor,
        sdw_log_shp: torch.Tensor,
        sdw_logits_shp: torch.Tensor,
        noise_scale: torch.Tensor,
        noise_min: torch.Tensor,
        noise_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bck = sdw_mask_bck * bck_imp + (sdw_mask_obj * sdw_apc).sum(
            2)  # [B, V, S, H, W, C]
        noise_coef = self.pos_grid_noise - \
            trs[:, :, :, None, None]  # [B, V, S, H, W, 2]
        noise_coef = -noise_scale * noise_coef.square().sum(
            -1, keepdims=True)  # [B, V, S, H, W, 1]
        noise_coef = noise_min + (noise_max - noise_min) * \
            (1 - torch.exp(noise_coef))  # [B, V, S, H, W, 1]
        apc_aux = apc + noise_coef * torch.randn_like(apc)
        shp = shp_imp * (1 - sdw_shp)
        log_shp = log_shp_imp + sdw_log_shp - sdw_logits_shp
        return bck, apc_aux, shp, log_shp

    @staticmethod
    @torch.compile(mode=compile_mode, disable=compile_disable)
    def compute_mask_obj(
        log_shp: torch.Tensor,
        logits_pres: torch.Tensor,
        log_ord: torch.Tensor,
        shp_mul_pres: torch.Tensor,
        mask_bck: torch.Tensor,
        ratio_stick_breaking: Union[torch.Tensor, None],
    ):
        index_sel = torch.argsort(log_ord.squeeze(-1), dim=2, descending=True)
        log_pres = torch.nn.functional.logsigmoid(logits_pres)[:, None, :,
                                                               None, None]
        log_ord = log_ord[:, :, :, None, None]
        mask_obj_aux_rel = torch.softmax(log_shp + log_pres + log_ord, dim=2)
        mask_obj_aux = (1 - mask_bck[:, :, None]) * mask_obj_aux_rel
        shp_mul_pres = select_by_index(shp_mul_pres, index_sel)
        ones = torch.ones(
            [*shp_mul_pres.shape[:2], 1, *shp_mul_pres.shape[3:]],
            device=shp_mul_pres.device)
        mask_obj = shp_mul_pres * torch.cat(
            [ones, (1 - shp_mul_pres[:, :, :-1]).cumprod(2)], dim=2)
        index_sel = torch.argsort(index_sel, dim=2)
        mask_obj = select_by_index(mask_obj, index_sel)
        mask_obj = mask_obj_aux - mask_obj_aux.detach() + mask_obj.detach()
        if ratio_stick_breaking is not None:
            mask_obj = ratio_stick_breaking * mask_obj + (
                1 - ratio_stick_breaking) * mask_obj_aux
        else:
            mask_obj = mask_obj_aux
        return mask_obj

    def compute_mask(
        self,
        shp: torch.Tensor,  # [B, V, S, H, W, 1]
        log_shp: torch.Tensor,  # [B, V, S, H, W, 1]
        pres: torch.Tensor,  # [B, S, 1]
        logits_pres: torch.Tensor,  # [B, S, 1]
        log_ord: torch.Tensor,  # [B, V, S, 1]
        ratio_stick_breaking: Union[torch.Tensor, None],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shp_mul_pres = shp * pres[:, None, :, None, None]
        mask_bck = (1 - shp_mul_pres).prod(2)
        mask_obj = self.compute_mask_obj(
            log_shp,
            logits_pres,
            log_ord,
            shp_mul_pres,
            mask_bck,
            ratio_stick_breaking,
        )
        return mask_bck, mask_obj

    @staticmethod
    @torch.compile(mode=compile_mode, disable=compile_disable)
    def compute_recon(
        apc: torch.Tensor,
        apc_aux: torch.Tensor,
        bck: torch.Tensor,
        bck_imp: torch.Tensor,
        mask_obj: torch.Tensor,
        mask_obj_imp: torch.Tensor,
        mask_bck: torch.Tensor,
        mask_bck_imp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon = mask_bck * bck + (mask_obj * apc).sum(2)
        recon_aux = mask_bck * bck + (mask_obj * apc_aux).sum(2)
        recon_aux_imp = mask_bck_imp * bck_imp + \
            (mask_obj_imp * apc_aux).sum(2)
        return recon, recon_aux, recon_aux_imp
