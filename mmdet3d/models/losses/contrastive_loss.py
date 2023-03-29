import torch
import torch.nn as nn
import numpy as np
from ..builder import LOSSES


# @weighted_loss
# def constrastive_loss(vis_feat, target_text_feat):
#     assert vis_feat.size() == target_text_feat.size() and target_text_feat.numel() > 0
#
#     vis_feat = vis_feat / vis_feat.norm(dim=1, keepdim=True)
#     text_feat = target_text_feat / target_text_feat.norm(dim=1, keepdim=True)
#
#     # cosine similarity as logits
#     logit_scale = self.logit_scale.exp()
#     logits_per_vis = logit_scale * vis_feat @ text_feat.t()
#     return loss

@LOSSES.register_module()
class ContrastiveLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self,
                vis_feat,
                text_feat,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        vis_feat = vis_feat / (vis_feat.norm(dim=1, keepdim=True) + 1e-6)
        text_feat = text_feat.float() / (text_feat.norm(dim=1, keepdim=True) + 1e-6)

        # cosine similarity as logits

        logit_scale = self.logit_scale.exp()
        logits_per_vis = (logit_scale * vis_feat.transpose(1, 2)) @ (text_feat.transpose(1, 2))
        return self.clip_loss(logits_per_vis)

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(logits.shape[1], device=logits.device).repeat(len(logits), 1))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.transpose(1, 2))
        return image_loss

