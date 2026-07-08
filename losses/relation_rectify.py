import torch
import torch.nn.functional as F


def build_affinity(features: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.t())
    aff = F.softmax(sim / tau, dim=1)
    return aff


def rectify_affinity(aff_old: torch.Tensor, pids: torch.Tensor):
    B = aff_old.size(0)

    same = pids.unsqueeze(1).eq(pids.unsqueeze(0))
    same.fill_diagonal_(False)

    eye = torch.eye(B, device=aff_old.device, dtype=torch.bool)
    diff = (~same) & (~eye)

    aff_rect = aff_old.clone()

    for i in range(B):
        pos_mask = same[i]
        neg_mask = diff[i]

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        # weakest positive
        sn = aff_old[i][pos_mask].min()

        # strongest negative
        sp = aff_old[i][neg_mask].max()

        # positive relation should be at least stronger than hardest negative
        aff_rect[i][pos_mask] = torch.maximum(
            aff_old[i][pos_mask],
            sp
        )

        # negative relation should be no stronger than weakest positive
        aff_rect[i][neg_mask] = torch.minimum(
            aff_old[i][neg_mask],
            sn
        )

    # keep probability distribution
    aff_rect = aff_rect / (aff_rect.sum(dim=1, keepdim=True) + 1e-12)

    return aff_rect, same, diff


def relation_rectify_kl_loss(
    feat_new: torch.Tensor,
    feat_old: torch.Tensor,
    pids: torch.Tensor,
    tau: float = 0.1,
    return_affinity: bool = False,
):
    """
    Selective correction + full topology distillation.

    feat_new: [B, C]
    feat_old: [B, C]
    pids: [B]
    """
    aff_new = build_affinity(feat_new, tau=tau)
    aff_old = build_affinity(feat_old, tau=tau)

    aff_old_rect, same, diff = rectify_affinity(aff_old, pids)

    loss = F.kl_div(
        torch.log(aff_new + 1e-12),
        aff_old_rect.detach(),
        reduction='batchmean'
    )

    if return_affinity:
        return loss, aff_new.detach(), aff_old.detach(), aff_old_rect.detach(), same, diff

    return loss

def flexible_relation_kl_loss(
    feat_new: torch.Tensor,
    feat_old: torch.Tensor,
    pids: torch.Tensor = None,
    tau: float = 0.1,
    beta: float = 0.1,
    return_affinity: bool = False,
):
    """
    Flexible relation knowledge distillation.

    feat_new: [B, C]
    feat_old: [B, C]
    beta: smoothing strength, e.g. 0.05 / 0.1 / 0.2
    """

    aff_new = build_affinity(feat_new, tau=tau)
    aff_old = build_affinity(feat_old, tau=tau)

    B = aff_old.size(0)

    # uniform relation distribution
    uniform = torch.ones_like(aff_old) / B

    # flexible historical knowledge
    aff_soft = (1.0 - beta) * aff_old + beta * uniform

    # KL(R_soft || R_new)
    loss = F.kl_div(
        torch.log(aff_new + 1e-12),
        aff_soft.detach(),
        reduction='batchmean'
    )

    if return_affinity:
        if pids is not None:
            same = pids.unsqueeze(1).eq(pids.unsqueeze(0))
            same.fill_diagonal_(False)
            eye = torch.eye(B, device=aff_old.device, dtype=torch.bool)
            diff = (~same) & (~eye)
            return loss, aff_new.detach(), aff_old.detach(), aff_soft.detach(), same, diff
        else:
            return loss, aff_new.detach(), aff_old.detach(), aff_soft.detach()

    return loss