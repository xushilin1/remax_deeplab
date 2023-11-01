import torch
import torch.nn.functional as F


_SOFTMAX_MASKING_CONSTANT = -99999.0

def divide_no_nan(x: torch.Tensor, y: torch.Tensor):
    return torch.nan_to_num(x / y, nan=0.0, posinf=0.0, neginf=0.0)


# https://github.com/google-research/deeplab2/blob/main/model/loss/max_deeplab_loss.py#L50
def _gumbel_topk_sample(logits: torch.Tensor, k: int):
    """Samples k points from the softmax distribution with Gumbel-Top-k trick."""
    # Note that torch.rand is [0, 1), we need to make it (0, 1) to ensure the log is valid.
    gumbel_noise = torch.rand(size=logits.shape, dtype=logits.dtype, device=logits.device)
    gumbel_noise = -torch.log(-torch.log(gumbel_noise))
    _, indices = torch.topk(logits + gumbel_noise, k)
    return indices


# https://github.com/google-research/deeplab2/blob/main/model/loss/max_deeplab_loss.py#L576
def pixelwise_insdis_loss(
    pixel_feature: torch.Tensor,
    gt_mask: torch.Tensor,
    sample_temperature: float,
    sample_k: int,
    instance_discrimination_temperature: float,
    pixel_gt_void_mask: torch.Tensor,
    inverse_gt_mask_area: torch.Tensor
    ):
    
    # pixel_feature: B x C x H x W
    # gt_mask: B x N x H x W
    pixel_feature = pixel_feature.flatten(2) # B x C x HW
    gt_mask = gt_mask.flatten(2) # B x N x HW
    pixel_gt_void_mask = pixel_gt_void_mask.flatten(1) # B x HW
    inverse_gt_mask_area = inverse_gt_mask_area.flatten(1) # B x HW

    sample_logits = torch.log(inverse_gt_mask_area) * sample_temperature # B x HW
    # sample_logits.masked_fill_(pixel_gt_void_mask, float('-inf'))
    sample_logits += pixel_gt_void_mask.to(sample_logits) * _SOFTMAX_MASKING_CONSTANT

    sample_indices = _gumbel_topk_sample(sample_logits, sample_k) # B x K
    # Sample ground truth one-hot encodings and compute gt_similarity.
    pixel_gt_sampled_feature = torch.gather(gt_mask, dim=2, index=sample_indices.unsqueeze(1).repeat(1, gt_mask.shape[1], 1)) # B x N x K
    sampled_gt_similarity = torch.einsum('bnk,bnj->bkj', pixel_gt_sampled_feature, pixel_gt_sampled_feature) # B x K x K

    # Normalize the ground truth similarity into a distribution (sum to 1).
    pixel_normalizing_constant = sampled_gt_similarity.sum(dim=1, keepdim=True) # B x 1 x K
    sampled_gt_similarity /= torch.clamp(pixel_normalizing_constant, min=1.0) # B x K x K

    # Sample predicted features and compute pred_similarity.
    pixel_pred_sampled_feature = torch.gather(pixel_feature, dim=2, index=sample_indices.unsqueeze(1).repeat(1, pixel_feature.shape[1], 1)) # B x C x K
    sampled_pred_similarity = torch.einsum('bck,bcj->bkj', pixel_pred_sampled_feature, pixel_pred_sampled_feature) # B x K x K
    sampled_pred_similarity /= instance_discrimination_temperature # B x K x K
    loss = F.cross_entropy(sampled_pred_similarity, sampled_gt_similarity, reduction="none") # B x K

    num_non_zero = (loss != 0.0).to(loss).sum(-1) # B
    num_non_zero = torch.clamp(num_non_zero, min=1.0)
    loss_sum_per_sample = loss.sum(-1) # B
    return divide_no_nan(loss_sum_per_sample, num_non_zero).mean() # 1




# https://github.com/google-research/deeplab2/blob/c4a533c14fac1a1071a6d24c5379c31a69a3e5e6/model/loss/base_loss.py#L56
# https://github.com/google-research/deeplab2/blob/main/model/loss/base_loss.py#L510
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        pixel_gt_void_mask: torch.Tensor,
        matched_cls_prob: torch.Tensor,
        masking_void_pixel: bool = True
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.softmax(1) # B N HW
    if masking_void_pixel:
        # https://github.com/google-research/deeplab2/blob/main/model/loss/base_loss.py#L111
        inputs = inputs.masked_fill(pixel_gt_void_mask.unsqueeze(1), 0) # remove void pixels.
    smooth = 1.0
    intersection = 2 * (inputs * targets).sum(-1) + smooth # B x N
    denominator = inputs.sum(-1) + targets.sum(-1) + smooth # B x N
    loss = 1.0 - divide_no_nan(intersection, denominator)
    loss *= matched_cls_prob
    # Note: kMaX-DeepLab sum over num_masks and avg over batches. But here batch and num_mask are one
    # https://github.com/google-research/deeplab2/blob/c4a533c14fac1a1071a6d24c5379c31a69a3e5e6/model/loss/base_loss.py#L559
    # https://github.com/google-research/deeplab2/blob/c4a533c14fac1a1071a6d24c5379c31a69a3e5e6/model/loss/max_deeplab_loss.py#L402
    # As the existing of modifer, it equals to multiplier by 0.75
    return (loss.sum(1) * 0.75/inputs.shape[1]).mean() # sum over masks and mean over batches.



def softmax_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        pixel_gt_void_mask: torch.Tensor,
        masking_void_pixel: bool = True
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.cross_entropy(inputs, targets, reduction="none") # B x HW
    loss = loss.masked_fill(pixel_gt_void_mask, 0) # remove void pixels.

    num_non_zero = (loss != 0.0).to(loss).sum(-1) # B
    num_non_zero = torch.clamp(num_non_zero, min=1.0)
    loss_sum_per_sample = loss.sum(-1) # B
    return divide_no_nan(loss_sum_per_sample, num_non_zero).mean() # 1



# https://github.com/google-research/deeplab2/blob/main/model/loss/base_loss.py#L393
def focal_cross_entropy_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    weight: torch.Tensor, # This is for PQ-loss weighting
    focal_loss_alpha: float = 0.75,
    focal_loss_gamma: float = 0.0,
    background_channel_index: int = -1):
    """
    pred: B x N x C
    gt: B x N
    weight: B x N
    """
    pred = pred.transpose(1, 2) # B x C x N
    gt = F.one_hot(gt, num_classes=pred.shape[1]).transpose(1, 2).to(pred) # B x C x N
    loss = F.cross_entropy(pred, gt, reduction="none") # B x N
    if focal_loss_gamma == 0.0:
        focal_loss = loss
    else:
        pred = F.softmax(pred, dim=1) # B x C x N
        pt = (pred * gt).sum(1)  # B x N
        focal_loss = torch.pow(1.0 - pt, focal_loss_gamma) * loss # B x N
    
    if focal_loss_alpha >= 0:
        alpha_weights = (
          focal_loss_alpha * (1.0 - gt[:, background_channel_index])
          + (1 - focal_loss_alpha) * gt[:, background_channel_index]) # B x N
        focal_loss = alpha_weights * focal_loss # B x N
    
    focal_loss = focal_loss * weight # B x N
    focal_loss = focal_loss.flatten(1)
    num_non_zero = (focal_loss != 0.0).to(focal_loss).sum(-1) # B
    num_non_zero = torch.clamp(num_non_zero, min=1.0)
    loss_sum_per_sample = focal_loss.sum(-1) # B
    return divide_no_nan(loss_sum_per_sample, num_non_zero).mean() # 1



def aux_semantic_loss(
    pred_semantic_logits: torch.Tensor,
    ground_truth_semantic: torch.Tensor,
    sample_temperature: float,
    sample_k: int,
    pixel_gt_void_mask: torch.Tensor,
    inverse_gt_mask_area: torch.Tensor,
    num_classes: int):

    # The pred maybe in lower resolution, we downsample gt beforehand.
    if pred_semantic_logits.shape[-2:] != ground_truth_semantic.shape[-2:]:
        assert (ground_truth_semantic.shape[-1] - 1) // (pred_semantic_logits.shape[-1] - 1) == (ground_truth_semantic.shape[-2] - 1) // (pred_semantic_logits.shape[-2] - 1)
        stride = (ground_truth_semantic.shape[-1] - 1) // (pred_semantic_logits.shape[-1] - 1)
        ground_truth_semantic = ground_truth_semantic[:, ::stride, ::stride]
        pixel_gt_void_mask = pixel_gt_void_mask[:, ::stride, ::stride]
        inverse_gt_mask_area = inverse_gt_mask_area[:, ::stride, ::stride]

    pred_semantic_logits = pred_semantic_logits.flatten(2) # B x C x HW
    ground_truth_semantic = ground_truth_semantic.flatten(1) # B x HW
    pixel_gt_void_mask = pixel_gt_void_mask.flatten(1) # B x HW
    inverse_gt_mask_area = inverse_gt_mask_area.flatten(1) # B x HW
    if sample_k == 0:
        # This falls back to normal cross-entropy loss
        sampled_ground_truth_semantic = ground_truth_semantic # B x HW
        sampled_pred_semantic_logits = pred_semantic_logits # B x C x HW
    else:
        sample_logits = torch.log(inverse_gt_mask_area) * sample_temperature # B x HW
        sample_logits += pixel_gt_void_mask.to(sample_logits) * _SOFTMAX_MASKING_CONSTANT
        sample_indices = _gumbel_topk_sample(sample_logits, sample_k) # B x K
        sampled_ground_truth_semantic = torch.gather(ground_truth_semantic, dim=1, index=sample_indices) # B x K
        sampled_pred_semantic_logits = torch.gather(pred_semantic_logits, dim=2, index=sample_indices.unsqueeze(1).repeat(1, pred_semantic_logits.shape[1], 1)) # B x C x K
    # ignore the class index num_classes.
    keep_mask = (sampled_ground_truth_semantic != num_classes) # B x K
    loss = F.cross_entropy(sampled_pred_semantic_logits, sampled_ground_truth_semantic, ignore_index=num_classes, reduction='none') # B x K
    loss = loss * keep_mask.to(loss)
    num_non_zero = (loss != 0.0).to(loss).sum(-1) # B
    num_non_zero = torch.clamp(num_non_zero, min=1.0)
    loss_sum_per_sample = loss.sum(-1) # B
    return divide_no_nan(loss_sum_per_sample, num_non_zero).mean() # 1