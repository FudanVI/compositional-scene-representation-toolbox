import math
from typing import Dict, List, Tuple

import torch
from building_block import (ConvBlock, GRULayer, LinearBlock, LinearLayer,
                            LinearPosEmbedLayer, Permute, compile_disable,
                            compile_mode)
from omegaconf import DictConfig


class SlotAttnMulti(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        num_steps: int,
        qry_size: int,
        slot_view_size: int,
        slot_attr_size: int,
        feature_res_list: List[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.slot_view_size = slot_view_size
        self.slot_attr_size = slot_attr_size
        slot_full_size = slot_view_size + slot_attr_size
        zeros_view = torch.zeros([1, 1, slot_view_size])
        zeros_attr = torch.zeros([1, 1, slot_attr_size])
        self.view_loc = torch.nn.Parameter(zeros_view)
        self.view_log_scl = torch.nn.Parameter(zeros_view)
        self.attr_loc = torch.nn.Parameter(zeros_attr)
        self.attr_log_scl = torch.nn.Parameter(zeros_attr)
        self.coef_key = 1 / math.sqrt(qry_size)
        self.split_key_val = [qry_size, slot_full_size]
        self.net_key_val = torch.nn.Sequential(
            torch.nn.LayerNorm(in_features),
            LinearLayer(
                in_features=in_features,
                out_features=sum(self.split_key_val),
                activation=None,
                bias=False,
            ),
        )
        self.net_qry = torch.nn.Sequential(
            torch.nn.LayerNorm(slot_full_size),
            LinearLayer(
                in_features=slot_full_size,
                out_features=qry_size,
                activation=None,
                bias=False,
            ),
        )
        self.net_upd = GRULayer(slot_full_size, slot_full_size)
        self.net_res = torch.nn.Sequential(
            torch.nn.LayerNorm(slot_full_size),
            LinearBlock(
                in_features=slot_full_size,
                feature_list=feature_res_list + [slot_full_size],
                act_inner=activation,
                act_out=None,
            ),
        )

    @torch.compile(mode=compile_mode, disable=compile_disable)
    def forward(
        self,
        x: torch.Tensor,  # [B, V, N, D]
        num_slots: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views = x.shape[:2]
        x = x.flatten(end_dim=1)  # [B * V, N, D]
        x_key, x_val = self.net_key_val(x).split(
            self.split_key_val, dim=-1)  # [B * V, N, (D_q, D_v + D_a)]
        x_key = x_key.transpose(
            -1, -2).contiguous() * self.coef_key  # [B * V, D_q, N]
        x_val = x_val.contiguous()  # [B * V, N, D_v + D_a]
        noise_view = torch.randn(
            [batch_size, num_views, self.slot_view_size],
            device=x.device,
        )
        slot_view = self.view_loc + torch.exp(
            self.view_log_scl) * noise_view  # [B, V, D_v]
        noise_attr = torch.randn(
            [batch_size, num_slots, self.slot_attr_size],
            device=x.device,
        )
        slot_attr = self.attr_loc + torch.exp(
            self.attr_log_scl) * noise_attr  # [B, S, D_a]
        for _ in range(self.num_steps):
            slot_full = torch.cat(
                [
                    slot_view[:, :, None].expand(-1, -1, num_slots, -1),
                    slot_attr[:, None].expand(-1, num_views, -1, -1),
                ],
                dim=-1,
            ).flatten(end_dim=1)  # [B * V, S, D_v + D_a]
            x_qry = self.net_qry(slot_full)  # [B * V, S, D_q]
            logits_attn = torch.bmm(x_qry, x_key)  # [B * V, S, N]
            attn = torch.softmax(torch.log_softmax(logits_attn, dim=-2),
                                 dim=-1)  # [B * V, S, N]
            x_upd = torch.bmm(attn, x_val)  # [B * V, S, D_v + D_a]
            slot_full = slot_full.flatten(end_dim=1)  # [B * V * S, D_v + D_a]
            x_upd = x_upd.flatten(end_dim=1)  # [B * V * S, D_v + D_a]
            with torch.cuda.amp.autocast(enabled=False):
                x_main = self.net_upd(x_upd.to(
                    torch.float32), slot_full.to(torch.float32)).unflatten(
                        0, [batch_size, num_views, num_slots]).to(
                            slot_full.dtype)  # [B, V, S, D_v + D_a]
            slot_full = x_main + self.net_res(x_main)  # [B, V, S, D_v + D_a]
            slot_view_raw, slot_attr_raw = slot_full.split(
                [self.slot_view_size, self.slot_attr_size], dim=-1)
            slot_view = slot_view_raw.mean(2)  # [B, V, D_v]
            slot_attr = slot_attr_raw.mean(1)  # [B, S, D_a]
        return slot_view, slot_attr


class Encoder(torch.nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        image_ht, image_wd, image_ch = cfg.dataset.image_shape
        net_conv = ConvBlock(
            in_shape=(image_ch, image_ht, image_wd),
            channel_list=cfg.model.enc_img.channel_list,
            kernel_list=cfg.model.enc_img.kernel_list,
            stride_list=cfg.model.enc_img.stride_list,
            act_inner=cfg.model.enc_img.activation,
            act_out=cfg.model.enc_img.activation,
        )
        conv_ch, conv_ht, conv_wd = net_conv.out_shape
        net_pos_embed = LinearPosEmbedLayer(
            ht=conv_ht,
            wd=conv_wd,
            out_features=conv_ch,
        )
        net_ln = torch.compile(
            torch.nn.LayerNorm(conv_ch),
            mode=compile_mode,
            disable=compile_disable,
        )
        net_linear = LinearBlock(
            in_features=conv_ch,
            feature_list=cfg.model.enc_cvt.feature_list,
            act_inner=cfg.model.enc_cvt.activation,
            act_out=None,
        )
        self.net_feat = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=0, end_dim=1),
            Permute([0, 3, 1, 2]),
            net_conv,
            net_pos_embed,
            Permute([0, 2, 3, 1]),
            net_ln,
            net_linear,
        )
        self.net_slot = SlotAttnMulti(
            in_features=cfg.model.enc_cvt.feature_list[-1],
            num_steps=cfg.model.enc_slot.num_steps,
            qry_size=cfg.model.enc_slot.qry_size,
            slot_view_size=cfg.model.enc_slot.slot_view_size,
            slot_attr_size=cfg.model.enc_slot.slot_attr_size,
            feature_res_list=cfg.model.enc_slot.feature_res_list,
            activation=cfg.model.enc_slot.activation,
        )

        # Viewpoint
        self.net_view = LinearBlock(
            in_features=cfg.model.enc_slot.slot_view_size,
            feature_list=cfg.model.enc_view.feature_list +
            [cfg.model.latent_view_size * 2],
            act_inner=cfg.model.enc_view.activation,
            act_out=None,
        )

        # Background
        self.net_bck_in = LinearBlock(
            in_features=cfg.model.enc_slot.slot_attr_size,
            feature_list=cfg.model.enc_bck_in.feature_list[:-1] +
            [cfg.model.enc_bck_in.feature_list[-1] + 1],
            act_inner=cfg.model.enc_bck_in.activation,
            act_out=None,
        )
        self.net_bck_out = LinearBlock(
            in_features=cfg.model.enc_bck_in.feature_list[-1],
            feature_list=cfg.model.enc_bck_out.feature_list +
            [cfg.model.latent_bck_size * 2],
            act_inner=cfg.model.enc_bck_out.activation,
            act_out=None,
        )

        # Objects
        self.split_obj = [cfg.model.latent_obj_size] * 2 + [1] * 3
        self.net_obj = LinearBlock(
            in_features=cfg.model.enc_slot.slot_attr_size,
            feature_list=cfg.model.enc_obj.feature_list +
            [sum(self.split_obj)],
            act_inner=cfg.model.enc_obj.activation,
            act_out=None,
        )

    def forward(
        self,
        image: torch.Tensor,  # [B, V, H, W, C]
        num_slots: int,
    ) -> Dict[str, torch.Tensor]:
        torch.cuda.nvtx.range_push('net_feat')
        x = self.net_feat(image).flatten(start_dim=1, end_dim=2).unflatten(
            0, image.shape[:2])  # [B, V, N, D]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('net_slot')
        slot_view, slot_attr = self.net_slot(x, num_slots)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('net_param')
        outputs = self.compute_posterior_params(slot_view, slot_attr)
        torch.cuda.nvtx.range_pop()
        return outputs

    @torch.compile(mode=compile_mode, disable=compile_disable)
    def compute_posterior_params(
        self,
        slot_view: torch.Tensor,
        slot_attr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Viewpoint
        view_param_list = self.net_view(slot_view).chunk(2, dim=-1)
        view_mu, view_logvar = [
            param.contiguous() for param in view_param_list
        ]

        # Background
        x = self.net_bck_in(slot_attr)  # [B, S, D' + 1]
        attn_sel = torch.softmax(x[..., -1:], dim=1)  # [B, S, 1]
        x = (x[..., :-1] * attn_sel).sum(1)  # [B, D']
        bck_param_list = self.net_bck_out(x).chunk(2, dim=-1)
        bck_mu, bck_logvar = [param.contiguous() for param in bck_param_list]

        # Objects
        obj_param_list = self.net_obj(slot_attr).split(self.split_obj, dim=-1)
        obj_mu, obj_logvar, logits_tau1, logits_tau2, logits_zeta = [
            param.contiguous() for param in obj_param_list
        ]
        tau1 = torch.nn.functional.softplus(logits_tau1)
        tau2 = torch.nn.functional.softplus(logits_tau2)
        zeta = torch.sigmoid(logits_zeta)

        # Outputs
        outputs = {
            'view_mu': view_mu,
            'view_logvar': view_logvar,
            'bck_mu': bck_mu,
            'bck_logvar': bck_logvar,
            'obj_mu': obj_mu,
            'obj_logvar': obj_logvar,
            'tau1': tau1,
            'tau2': tau2,
            'zeta': zeta,
            'logits_zeta': logits_zeta,
        }
        return outputs
