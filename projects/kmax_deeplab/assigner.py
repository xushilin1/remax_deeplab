# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast

from mmdet.registry import TASK_UTILS
from mmdet.models.task_modules.assigners import HungarianAssigner
from mmdet.models.task_modules.assigners import AssignResult




# https://github.com/google-research/deeplab2/blob/c4a533c14fac1a1071a6d24c5379c31a69a3e5e6/model/loss/max_deeplab_loss.py#L158
@torch.no_grad()
def compute_mask_similarity(inputs: torch.Tensor, targets: torch.Tensor,
                            masking_void_pixel=True):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    denominator_epsilon = 1e-5
    inputs = F.softmax(inputs, dim=0)
    inputs = inputs.flatten(1) # N x HW

    pixel_gt_non_void_mask = (targets.sum(0, keepdim=True) > 0).to(inputs) # 1xHW
    if masking_void_pixel:
        inputs = inputs * pixel_gt_non_void_mask

    intersection = torch.einsum("nc,mc->nm", inputs, targets)
    denominator = (inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]) / 2.0
    return intersection / (denominator + denominator_epsilon)

# https://github.com/google-research/deeplab2/blob/c4a533c14fac1a1071a6d24c5379c31a69a3e5e6/model/loss/max_deeplab_loss.py#L941
@torch.no_grad()
def compute_class_similarity(inputs: torch.Tensor, targets: torch.Tensor):
    pred_class_prob = inputs.softmax(-1)[..., :-1] # exclude the void class
    return pred_class_prob[:, targets]


@TASK_UTILS.register_module()
class KMaxHungarianAssigner(HungarianAssigner):

    def __init__(
        self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict]
    ) -> None:

        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, 'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places. It may includes ``masks``, with shape
                (n, h, w) or (n, l).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                ``labels``, with shape (k, ) and ``masks``, with shape
                (k, h, w) or (k, l).
            img_meta (dict): Image information.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),  -1, dtype=torch.long, device=device)
        assigned_labels = torch.full((num_preds, ), -1, dtype=torch.long, device=device)

        if num_gts == 0 or num_preds == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. compute weighted cost
        # cost_list = []
        # dict(type='ClassificationCost', weight=2.0),
        # class_similarity = self.match_costs[0](
        #         pred_instances=pred_instances,
        #         gt_instances=gt_instances,
        #         img_meta=img_meta)
        pred_scores = pred_instances.scores
        gt_labels = gt_instances.labels

        with autocast(enabled=False):
            class_similarity = compute_class_similarity(pred_scores.float(), gt_labels)

        pred_masks = pred_instances.masks.flatten(1)
        target_masks = gt_instances.masks.flatten(1)
        with autocast(enabled=False):
            mask_similarity = compute_mask_similarity(pred_masks.float(), target_masks.float())

        cost = - mask_similarity * class_similarity
        cost = cost.reshape(num_preds, -1)

        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_dice = mask_similarity[matched_row_inds, matched_col_inds].detach()
        matched_cls_prob = class_similarity[matched_row_inds, matched_col_inds].detach()
        
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)
    
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        assign_result = AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)

        assign_result.set_extra_property('matched_dice', matched_dice)
        assign_result.set_extra_property('matched_cls_prob', matched_cls_prob)
        return assign_result
