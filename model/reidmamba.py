#改后代码
import torch
import math
from torch import nn
from einops import rearrange
import logging
from .build import MODEL_REGISTRY
from timm import create_model
from timm.layers.helpers import to_2tuple
from timm.layers import trunc_normal_
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import Block
from .bot import BNNeck
from copy import deepcopy
from functools import partial
from .transreid import PatchEmbed_overlap
from mamba_ssm.ops.triton.layer_norm import rms_norm_fn, RMSNorm
from timm.models.layers import DropPath

from .mambar import *
from .mambar import create_block, get_cls_idx
import numpy as np
import math
import random
import einops
from copy import deepcopy

import torch.nn.functional as F


logger = logging.getLogger(__name__)


_backbones = {
   'mambar_base_patch16_224': [
       'mambar_base_patch16_224',
       '/root/data/.cache/models/mambar_base_patch16_224.pth'],
   'mambar_small_patch16_224': [
       'mambar_small_patch16_224',
        'pre-trained/mambar_small_patch16_224.pth'],
}


def get_bb_cls_idxs(old_cls_pos, cls_pos):
    if cls_pos.size(0) > old_cls_pos.size(0):
        # when the number of cls_token is larger than the original one, we need to add some padding to the cls_pos
        assert (cls_pos.size(0) - old_cls_pos.size(0)) % 2 == 0
        t = (cls_pos.size(0) - old_cls_pos.size(0)) // 2
        return t

    old_float_idxs = []
    t = 0
    for i in old_cls_pos:
        old_float_idxs.append((i - 0.5 - t).item())
        t += 1

    new_float_idxs = []
    t = 0
    for i in cls_pos:
        new_float_idxs.append(i - 0.5 - t)
        t += 1

    idxs = []
    old_float_np = np.array(old_float_idxs)
    for i in new_float_idxs:
        idx = np.argmin(np.abs(old_float_np - i.item()))
        idxs.append(idx)
        old_float_np[idx] = np.inf

    return torch.LongTensor(idxs)


class GeneralizedMean(nn.Module):
    def __init__(self, norm=3, eps=1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * norm)
        self.eps = eps

    def forward(self, x):
        # B N R D
        x = x.clamp(min=self.eps).pow(self.p)
        return x.mean(dim=2).pow(1. / self.p)


def get_oth_pos(num_patches, cls_pos):
    ori_indices = torch.arange(num_patches + cls_pos.size(0))
    pre_i = 0
    other_positions_lists = []
    for i in cls_pos:
        other_positions_lists.append(ori_indices[pre_i:i])
        pre_i = i + 1
    other_positions_lists.append(ori_indices[pre_i:])
    return torch.cat(other_positions_lists)


class HistoricalCurrentCoupling(nn.Module):
    """
    Explicitly compares current cls tokens and historical cls tokens,
    then adaptively selects how much new / old information to keep.

    Inputs:
        x_cur:  [B, N_cls, C]  current representation
        x_hist: [B, N_cls, C]  historical representation (residual branch)

    Output:
        out:    [B, N_cls, C]
    """
    def __init__(self, dim, hidden_ratio=1.0, use_residual=True):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.use_residual = use_residual

        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x_cur, x_hist):
        # Explicit discrepancy / consistency modeling
        diff = torch.abs(x_cur - x_hist)
        prod = x_cur * x_hist

        relation = torch.cat([x_cur, x_hist, diff, prod], dim=-1)
        gate = self.gate_mlp(relation)

        out = gate * x_cur + (1.0 - gate) * x_hist
        out = self.refine(out)

        if self.use_residual:
            out = out + x_cur

        return out


@MODEL_REGISTRY.register()
class ReIDMamba(nn.Module):
    def __init__(self, backbone_name='mambar_base_patch16_224', num_classes=751, img_size=224, patch_size=16, stride_size=16,
                 in_chans=3, drop_path_rate=0.1, num_cls_tokens=8, cls_reduce=4, num_branches=1, token_fusion_type='max',
                 use_cid=False, num_cids=0, sie_xishu=3.0,
                 use_hc_coupling=True, coupling_on_branches=None, coupling_hidden_ratio=1.0,
                 use_token_swap=False, swap_ratio=0.1,
                 *args, **kwargs):
        super().__init__()

        name, path = _backbones[backbone_name]
        bb = create_model(name)
        bb.load_state_dict(torch.load(path)['model'])
        logger.info('loading backbone from {}'.format(backbone_name))
        logger.info('\t embedding dim is : {}'.format(bb.embed_dim))
        logger.info('\t number of cls_token is : {}'.format(num_cls_tokens))
        logger.info('\t number of layers is : {}'.format(bb.depth))
        logger.info('\t reduction factor is : {}'.format(cls_reduce))
        logger.info('\t finale feature dim is : {}'.format(bb.embed_dim // cls_reduce * num_cls_tokens * num_branches))

        self.num_classes = num_classes
        self.num_features = self.embed_dim = bb.embed_dim
        self.use_cid = use_cid
        self.cls_reduce = cls_reduce
        self.num_cls_tokens = num_cls_tokens
        self.num_branches = num_branches
        self.token_fusion_type = token_fusion_type
        self.stride_size = stride_size

        # coupling config
        self.use_hc_coupling = bool(use_hc_coupling)
        if coupling_on_branches is None:
            coupling_on_branches = list(range(num_branches))
        self.coupling_on_branches = coupling_on_branches

        # token swap config
        self.use_token_swap = bool(use_token_swap)
        self.swap_ratio = swap_ratio

        if token_fusion_type == 'gem':
            self.gems = nn.ModuleList([GeneralizedMean(norm=3) for _ in range(2 * (num_branches - 1))])

        # patch embedding
        self.patch_embed = PatchEmbed_overlap(
            img_size=img_size,
            patch_size=patch_size,
            stride_size=stride_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim
        )
        self.patch_embed.proj.load_state_dict(bb.patch_embed.proj.state_dict())
        num_patches = self.patch_embed.num_patches

        # initialize cls token and position embedding from Mamba-R
        _, bb_cls_positions = get_cls_idx(self.patch_embed.num_y, self.patch_embed.num_x, bb.num_cls_tokens)
        self.token_idx, self.cls_idx = get_cls_idx(self.patch_embed.num_y, self.patch_embed.num_x, num_cls_tokens)
        self.cls_token = nn.Parameter(torch.zeros(1, num_cls_tokens, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, num_cls_tokens, self.embed_dim))
        trunc_normal_(self.cls_token.data, std=.02)
        trunc_normal_(self.pos_embed_cls.data, std=.02)
        self.oth_idx = get_oth_pos(num_patches, self.cls_idx)

        # copy cls token from Mamba-R
        with torch.no_grad():
            idxs = get_bb_cls_idxs(bb_cls_positions, self.cls_idx)
            if isinstance(idxs, torch.Tensor):
                self.cls_token.data.copy_(bb.cls_token.data[:, idxs])
                self.pos_embed_cls.data.copy_(bb.pos_embed_cls.data[:, idxs])
            else:
                self.cls_token.data[:, idxs:-idxs] = bb.cls_token.data
                self.pos_embed_cls.data[:, idxs:-idxs] = bb.pos_embed_cls.data

        # copy position embedding from Mamba-R
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        inter_pos_embed = resample_abs_pos_embed(
            posemb=bb.pos_embed,
            new_size=[self.patch_embed.num_y, self.patch_embed.num_x],
            num_prefix_tokens=0,
            verbose=True,
        )
        with torch.no_grad():
            self.pos_embed.data = inter_pos_embed.data

        # Initialize SIE Embedding
        self.cam_num = num_cids
        self.sie_xishu = sie_xishu
        if self.use_cid:
            self.sie_embed = nn.Parameter(torch.zeros(num_cids, 1, self.embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            logger.info('camera number is : {}'.format(num_cids))
            logger.info('using SIE_Lambda is : {}'.format(sie_xishu))

        # drop path rate
        logger.info('using drop_path rate is : {}'.format(drop_path_rate))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, bb.depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    self.embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    drop_path=inter_dpr[i]
                )
                for i in range(bb.depth - 2)
            ]
        )
        for i in range(bb.depth - 2):
            self.layers[i].load_state_dict(bb.layers[i].state_dict())

        base_layer = nn.ModuleList(
            [
                create_block(
                    self.embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=i,
                    drop_path=inter_dpr[i]
                ) for i in range(bb.depth - 2, bb.depth)
            ]
        )
        for i in range(bb.depth - 2, bb.depth):
            base_layer[i - bb.depth + 2].load_state_dict(bb.layers[i].state_dict())

        self.multi_layers = nn.ModuleList()
        self.norm_fs = nn.ModuleList()
        self.necks = nn.ModuleList()
        self.norm_necks = nn.ModuleList()
        self.bnnecks = nn.ModuleList()

        # one coupling module per branch
        self.couplings = nn.ModuleList()

        self.down_token_idx = []
        self.down_cls_idx = []
        sampling_rate = 1
        for b in range(num_branches):
            self.multi_layers.append(deepcopy(base_layer))
            self.norm_fs.append(deepcopy(bb.norm_f))
            self.necks.append(nn.Linear(self.embed_dim, self.embed_dim // cls_reduce * sampling_rate, bias=False))
            trunc_normal_(self.necks[b].weight, std=0.02)
            self.norm_necks.append(RMSNorm((self.embed_dim // cls_reduce) * num_cls_tokens, eps=1e-5))
            self.bnnecks.append(BNNeck((self.embed_dim // cls_reduce) * num_cls_tokens, num_classes, False, pool=None, neck_feat='before', init_mode=0))

            self.couplings.append(
                HistoricalCurrentCoupling(
                    dim=self.embed_dim,
                    hidden_ratio=coupling_hidden_ratio,
                    use_residual=True
                )
            )

            if b:
                token_idx, cls_idx = get_cls_idx(self.patch_embed.num_y, self.patch_embed.num_x, num_cls_tokens // sampling_rate)
                self.down_token_idx.append(token_idx)
                self.down_cls_idx.append(cls_idx)

            sampling_rate *= 2

    def token_swap(self, x_cur, x_hist, patch_idx, swap_ratio=0.2):
        """
        Bidirectional discrepancy-aware token swap using cosine distance.

        Strategy:
        - compute cosine discrepancy
        - discard the top 5% most extreme discrepancy tokens
        - select the largest-discrepancy tokens from the remaining candidates
        - swap only patch tokens, cls tokens remain unchanged
        """
        if (not self.training) or (swap_ratio <= 0) or (x_hist is None):
            return x_cur, x_hist

        x_cur = x_cur.clone()
        x_hist = x_hist.clone()

        patch_idx = patch_idx.to(x_cur.device)
        x_hist = x_hist.to(device=x_cur.device, dtype=x_cur.dtype)

        B = x_cur.size(0)
        num_patch = patch_idx.numel()
        num_swap = max(1, int(num_patch * swap_ratio))

        for b in range(B):
            cur_patch = x_cur[b, patch_idx]
            hist_patch = x_hist[b, patch_idx]

            cur_patch_n = F.normalize(cur_patch.float(), dim=-1)
            hist_patch_n = F.normalize(hist_patch.float(), dim=-1)
            score = 1.0 - (cur_patch_n * hist_patch_n).sum(dim=-1)   # [P]

            # ascending sort
            sorted_idx = torch.argsort(score, descending=False)

            # remove the most extreme top 5%
            cutoff = int(num_patch * 0.95)
            if cutoff <= 0:
                candidate = sorted_idx
            else:
                candidate = sorted_idx[:cutoff]

            # from remaining candidates, select the largest-discrepancy ones
            if candidate.numel() <= num_swap:
                selected = candidate
            else:
                candidate_score = score[candidate]
                _, top_idx = torch.topk(candidate_score, k=num_swap, largest=True)
                selected = candidate[top_idx]

            idx = patch_idx[selected]

            # bidirectional swap
            tmp = x_cur[b, idx].clone()
            x_cur[b, idx] = x_hist[b, idx]
            x_hist[b, idx] = tmp

        return x_cur, x_hist

    def forward_features(self, x, cids=None, get_tokens=False):
        x = self.patch_embed(x)
        B, _, _ = x.shape

        x = x + self.pos_embed

        cls_token = self.cls_token.expand(B, -1, -1) + self.pos_embed_cls
        x = torch.cat([x, cls_token], dim=1)[:, self.token_idx]

        if self.use_cid and cids is not None:
            x = x + self.sie_embed[cids] * self.sie_xishu

        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        # patch-level token swap (cls token remains unchanged)
        if self.use_token_swap and (residual is not None):
            hidden_states, residual = self.token_swap(
                hidden_states, residual, self.oth_idx, self.swap_ratio
            )

        hidden_states_cls = []
        sampling_rate = 1
        for b in range(self.num_branches):
            if b == 0:
                _hidden_states, _residual = hidden_states, residual
            else:
                _residual = None
                _hidden_states = residual + self.multi_layers[b][0].drop_path(hidden_states)

                _hidden_states_cls = _hidden_states[:, self.cls_idx]
                _hidden_states_oth = _hidden_states[:, self.oth_idx]

                if self.token_fusion_type == 'max':
                    _hidden_states_cls = torch.max(
                        _hidden_states_cls.view(B, self.num_cls_tokens // sampling_rate, sampling_rate, -1), dim=2
                    )[0]
                elif self.token_fusion_type == 'avg':
                    _hidden_states_cls = torch.mean(
                        _hidden_states_cls.view(B, self.num_cls_tokens // sampling_rate, sampling_rate, -1), dim=2
                    )
                elif self.token_fusion_type == 'gem':
                    _hidden_states_cls = self.gems[2 * b - 2](
                        _hidden_states_cls.view(B, self.num_cls_tokens // sampling_rate, sampling_rate, -1)
                    )
                else:
                    raise NotImplementedError

                _hidden_states = torch.cat([_hidden_states_oth, _hidden_states_cls], dim=1)[:, self.down_token_idx[b - 1]]

            for layer in self.multi_layers[b]:
                _hidden_states, _residual = layer(_hidden_states, _residual)

            _hidden_states = rms_norm_fn(
                self.drop_path(_hidden_states),
                self.norm_fs[b].weight,
                self.norm_fs[b].bias,
                eps=self.norm_fs[b].eps,
                residual=_residual,
                prenorm=False,
                residual_in_fp32=True,
            )

            # ===========================
            # Explicit historical-current coupling on cls tokens
            # ===========================
            if b == 0:
                _hidden_states_cls = _hidden_states[:, self.cls_idx]

                if self.use_hc_coupling and (b in self.coupling_on_branches) and (_residual is not None):
                    _residual_norm = rms_norm_fn(
                        self.drop_path(_hidden_states),
                        self.norm_fs[b].weight,
                        self.norm_fs[b].bias,
                        eps=self.norm_fs[b].eps,
                        residual=_residual,
                        prenorm=False,
                        residual_in_fp32=True,
                    )
                    _residual_cls = _residual_norm[:, self.cls_idx]
                    _hidden_states_cls = self.couplings[b](_hidden_states_cls, _residual_cls)

                _hidden_states_cls = self.necks[b](_hidden_states_cls)

            else:
                _hidden_states_cls = _hidden_states[:, self.down_cls_idx[b - 1]]

                if self.use_hc_coupling and (b in self.coupling_on_branches) and (_residual is not None):
                    _residual_norm = rms_norm_fn(
                        self.drop_path(_hidden_states),
                        self.norm_fs[b].weight,
                        self.norm_fs[b].bias,
                        eps=self.norm_fs[b].eps,
                        residual=_residual,
                        prenorm=False,
                        residual_in_fp32=True,
                    )
                    _residual_cls = _residual_norm[:, self.down_cls_idx[b - 1]]
                    _hidden_states_cls = self.couplings[b](_hidden_states_cls, _residual_cls)

                _hidden_states_cls = self.necks[b](_hidden_states_cls)

            _hidden_states_cls = self.norm_necks[b](_hidden_states_cls.view(B, -1))
            hidden_states_cls.append(_hidden_states_cls)
            sampling_rate *= 2

        return hidden_states_cls

    def forward(self, x, cids=None, get_tokens=False, *args, **kwargs):
        fs = self.forward_features(x, cids, get_tokens)

        if not self.training:
            return torch.cat([F.normalize(f) for f in fs], dim=1)

        else:
            tri = []
            log = []
            for i, f in enumerate(fs):
                res = self.bnnecks[i](f)
                tri.append(res[0][0])
                log.append(res[1][0])
            return tri, log, [[F.normalize(f) for f in fs]], [[F.normalize(f) for f in fs]]

    def get_params(self, *args, **kwargs):
        no_weight_decay_list = {"pos_embed", "cls_token", "sie_embed", "pos_embed_cls"}

        params = []
        for k, v in self.named_parameters():
            if not v.requires_grad:
                continue

            if k in no_weight_decay_list or k.endswith(".A_log") or k.endswith(".D") or k.endswith(".A_b_log") or k.endswith(".D_b") or v.ndim <= 1 or k.endswith(".bias"):
                params += [{"params": [v], "weight_decay": 0}]
            else:
                params += [{"params": [v]}]

        return params

    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'bnneck.' not in n and 'norm_neck.' not in n:
                p.requires_grad_(False)

    def unfreeze_backbone(self):
        for n, p in self.named_parameters():
            if 'bnneck.' not in n and 'norm_neck.' not in n:
                p.requires_grad_(True)

    def eval_backbone(self):
        for name, child in self.named_children():
            if name != "bnneck":
                child.eval()

    def train_backbone(self):
        for name, child in self.named_children():
            if name != "bnneck":
                child.train()

    def expand_classifier(self, new_num_classes):
        if new_num_classes <= 0:
            return

        old_num_classes = self.num_classes
        total_num_classes = old_num_classes + new_num_classes

        for i, bnneck in enumerate(self.bnnecks):
            old_fc = bnneck.cls

            in_features = old_fc.in_features
            bias_flag = old_fc.bias is not None

            new_fc = nn.Linear(
                in_features,
                total_num_classes,
                bias=bias_flag
            ).to(
                device=old_fc.weight.device,
                dtype=old_fc.weight.dtype
            )

            trunc_normal_(new_fc.weight, std=0.02)

            new_fc.weight.data[:old_num_classes].copy_(old_fc.weight.data)

            if bias_flag and old_fc.bias is not None:
                nn.init.zeros_(new_fc.bias)
                new_fc.bias.data[:old_num_classes].copy_(old_fc.bias.data)

            bnneck.cls = new_fc

            logger.info(
                f"BNNeck[{i}] classifier expanded: "
                f"{old_num_classes} -> {total_num_classes}"
            )

        self.num_classes = total_num_classes
        logger.info(
            f"Expanded classifier from {old_num_classes} "
            f"to {total_num_classes} classes."
        )

    def expand_sie_embed(self, new_num_cids):
        """
        Expand SIE embedding for new cameras / domains.
        """
        if not self.use_cid or new_num_cids <= 0:
            return

        old_num_cids = self.cam_num
        total_num_cids = old_num_cids + new_num_cids

        new_sie = nn.Parameter(
            torch.zeros(
                total_num_cids,
                1,
                self.embed_dim,
                device=self.sie_embed.device,
                dtype=self.sie_embed.dtype
            )
        )

        trunc_normal_(new_sie.data, std=.02)
        new_sie.data[:old_num_cids].copy_(self.sie_embed.data)

        self.sie_embed = new_sie
        self.cam_num = total_num_cids

        logger.info(
            f"Expanded SIE embeddings from "
            f"{old_num_cids} to {total_num_cids}."
        )