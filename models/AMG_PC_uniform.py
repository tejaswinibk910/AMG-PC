# AMG-PC: Adaptive Modality-Gated Point Cloud Completion
# Authors: Tejaswini Balamurugan Kanimozhi, Luke Liu
# Built on top of IAET (Fang et al., IJCAI 2025)

import torch
import torch.utils.data
import torch.nn as nn
from models.model_utils import InterlacedTransformer, ViewGuidedUpLayer
from models.utils import MLP_Res, PointNet_SA_Module_KNN
from torchvision import models


class PcEncoder(nn.Module):
    def __init__(self, dim_feat=256, num_points=256):
        super(PcEncoder, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128],
                                                  group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(num_points, 16, 128, [128, dim_feat],
                                                  group_all=False, if_bn=False, if_idx=True)

    def forward(self, point_cloud):
        l0_xyz = point_cloud
        l0_points = point_cloud
        l1_xyz, l1_points, _ = self.sa_module_1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa_module_2(l1_xyz, l1_points)
        return l2_points


class ImEncoder(nn.Module):
    def __init__(self, dim_feat=256):
        super().__init__()
        self.dim_feat = dim_feat
        base = models.resnet18(pretrained=False)
        self.base = nn.Sequential(*list(base.children())[:-3])

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), self.dim_feat, -1)
        return x


class TextEncoder(nn.Module):
    """Projects precomputed CLIP text embeddings (512-d) into model feature space."""
    def __init__(self, clip_dim=512, dim_feat=256):
        super().__init__()
        self.proj = nn.Linear(clip_dim, dim_feat)
        self.norm = nn.LayerNorm(dim_feat)

    def forward(self, text_embed):
        """
        Args:
            text_embed: (B, 512)
        Returns:
            (B, dim_feat, 1)
        """
        x = self.norm(self.proj(text_embed))   # (B, dim_feat)
        return x.unsqueeze(-1)                  # (B, dim_feat, 1)


class ModalityGate(nn.Module):
    """
    Lightweight gating module that outputs per-modality attention weights
    at each decoder stage. Takes the current point cloud feature as context
    and returns 3 weights (pc, image, text) via softmax.
    """
    def __init__(self, dim_feat=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim_feat, dim_feat // 4),
            nn.ReLU(),
            nn.Linear(dim_feat // 4, 3),
        )

    def forward(self, pc_feat):
        """
        Args:
            pc_feat: (B, dim_feat, N)
        Returns:
            weights: (B, 3) — softmax weights for [pc, image, text]
        """
        pooled = pc_feat.mean(dim=-1)                      # (B, dim_feat)
        weights = torch.ones(pooled.shape[0], 3, device=pooled.device) / 3  # uniform ablation
        return weights


class CoarseDecoder(nn.Module):
    def __init__(self, dim_feat=256, num_points=256):
        super(CoarseDecoder, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_points, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        f_g = torch.max(feat, -1, keepdim=True)[0]
        x1 = self.ps(f_g)
        x1 = self.mlp_1(torch.cat([x1, f_g.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, f_g.repeat((1, 1, x2.size(2)))], 1))
        completion = self.mlp_4(x3)
        return completion


class GatedDecoder(nn.Module):
    """
    Decoder with per-stage adaptive modality gating.
    At each upsampling stage, a ModalityGate computes weights (alpha_pc,
    alpha_img, alpha_txt) that scale the image and text contributions
    to the ViewGuidedUpLayer.
    """
    def __init__(self, dim_feat, num_points, up_factors):
        super(GatedDecoder, self).__init__()
        self.coarse_decoder = CoarseDecoder(dim_feat, num_points)
        self.uppers = nn.ModuleList([
            ViewGuidedUpLayer(dim_feat, factor) for factor in up_factors
        ])
        self.gates = nn.ModuleList([
            ModalityGate(dim_feat) for _ in up_factors
        ])

    def forward(self, fused_pc_token, fused_im_token, txt_token):
        """
        Args:
            fused_pc_token: (B, dim_feat, N)
            fused_im_token: (B, dim_feat, M)
            txt_token:      (B, dim_feat, 1)
        Returns:
            arr_pcd:      list of point clouds at each stage
            gate_weights: list of (B, 3) tensors — one per upsampling stage
        """
        arr_pcd = []
        gate_weights = []

        pcd = self.coarse_decoder(fused_pc_token)
        arr_pcd.append(pcd.permute(0, 2, 1).contiguous())
        # pcd stays (B, 3, N) from coarse_decoder — correct for ViewGuidedUpLayer

        K_prev = None
        for i, upper in enumerate(self.uppers):
            # Compute gate weights from current pc features
            alpha = self.gates[i](fused_pc_token)     # (B, 3)
            gate_weights.append(alpha)

            a_img = alpha[:, 1].view(-1, 1, 1)        # (B, 1, 1)
            a_txt = alpha[:, 2].view(-1, 1, 1)        # (B, 1, 1)

            # Blend text into image features, weighted by gate
            # txt_token is (B, dim_feat, 1) — broadcast across M image tokens
            gated_im = (a_img * fused_im_token +
                        a_txt * txt_token.expand_as(fused_im_token))

            pcd, K_prev = upper(pcd, fused_pc_token, gated_im, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd, gate_weights


class AMG_PC(nn.Module):
    """
    Adaptive Modality-Gated Point Cloud Completion.
    Extends IAET with a text modality branch and per-stage adaptive
    modality gating across the coarse-to-fine decoder.
    """
    def __init__(self, dim_feat=256, num_points=256, up_factors=(2, 2, 2), num_blocks=1):
        super().__init__()
        self.pc_encoder   = PcEncoder(dim_feat, num_points)
        self.im_encoder   = ImEncoder(dim_feat)
        self.text_encoder = TextEncoder(clip_dim=512, dim_feat=dim_feat)
        self.fusion_module = InterlacedTransformer(dim_feat, dim_feat, dim_feat, num_blocks)
        self.decoder = GatedDecoder(dim_feat, num_points, up_factors)

    def forward(self, pc, image, text_embed):
        """
        Args:
            pc:          (B, N, 3)
            image:       (B, 3, H, W)
            text_embed:  (B, 512)
        Returns:
            arr_pcd:     list of completed point clouds (coarse to fine)
            gate_weights: list of (B, 3) gate weight tensors per stage
        """
        pc_token  = self.pc_encoder(pc.permute(0, 2, 1).contiguous())
        im_token  = self.im_encoder(image)
        txt_token = self.text_encoder(text_embed)

        fused_pc_token, fused_im_token = self.fusion_module(
            pc_token.permute(0, 2, 1).contiguous(),
            im_token.permute(0, 2, 1).contiguous()
        )

        pc_completion, gate_weights = self.decoder(
            fused_pc_token.permute(0, 2, 1).contiguous(),
            fused_im_token.permute(0, 2, 1).contiguous(),
            txt_token
        )
        return pc_completion, gate_weights


if __name__ == '__main__':
    x_part     = torch.randn(4, 2048, 3).cuda()
    view       = torch.randn(4, 3, 224, 224).cuda()
    text_embed = torch.randn(4, 512).cuda()
    model = AMG_PC().cuda()
    out, gates = model(x_part, view, text_embed)
    for i, o in enumerate(out):
        print(f"Stage {i}: {o.shape}")
    for i, g in enumerate(gates):
        print(f"Gate {i} weights (first sample): {g[0].detach().cpu().numpy()}")
