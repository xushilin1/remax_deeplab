from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors import Mask2Former, MaskFormer
from mmdet.models.detectors.single_stage import SingleStageDetector
from typing import Dict, List, Tuple

from mmdet.structures import SampleList
from mmengine.structures import InstanceData, PixelData

import torch
from torch.nn import functional as F
from mmdet.evaluation import INSTANCE_OFFSET


@MODELS.register_module()
class kMaXDeepLab(SingleStageDetector):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # from .backbone import ResNet
        # self.backbone = ResNet()
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        panoptic_head_ = panoptic_head.deepcopy()
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = MODELS.build(panoptic_head_)

        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)
        losses = self.panoptic_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        feats = self.extract_feat(batch_inputs)
        mask_cls_results, mask_pred_results = self.panoptic_head.predict(
            feats, batch_data_samples)

        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        results_list = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, batch_img_metas):
            align_corners = (meta['img_shape'][-1] % 2 == 1) 
            
            img_height, img_width = meta['resize_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]
           
            ori_height, ori_width = meta['ori_shape'][:2]
            mask_pred_result = F.interpolate(
                mask_pred_result[:, None],
                size=(ori_height, ori_width),
                mode='bilinear',
                align_corners=align_corners)[:, 0]

            result = dict()

            mask_cls = mask_cls_result
            mask_pred = mask_pred_result

            reorder_class_weight = 1.0
            reorder_mask_weight = 1.0
            pixel_confidence_threshold = 0.4
            cls_threshold_thing = 0.7
            cls_threshold_stuff = 0.5
            overlap_threshold = 0.8

            cls_scores, cls_labels = F.softmax(mask_cls, dim=-1)[..., :-1].max(-1) # N
            mask_scores = F.softmax(mask_pred, dim=0)
            binary_masks = mask_scores > pixel_confidence_threshold # N x H x W
            mask_scores_flat = mask_scores.flatten(1) # N x HW
            binary_masks_flat = binary_masks.flatten(1).float() # N x HW
            pixel_number_flat = binary_masks_flat.sum(1) # N
            mask_scores_flat = (mask_scores_flat * binary_masks_flat).sum(1) / torch.clamp(pixel_number_flat, min=1.0) # N
            
            reorder_score = (cls_scores ** reorder_class_weight) * (mask_scores_flat ** reorder_mask_weight) # N
            reorder_indices = torch.argsort(reorder_score, dim=-1, descending=True)
            h, w = mask_pred.shape[-2:]
            panoptic_seg = torch.full((h, w), self.num_classes, dtype=torch.int32, device=mask_pred.device)

            current_segment_id = 1
            for i in range(mask_pred.shape[0]):
                cur_idx = reorder_indices[i].item()
                cur_binary_mask = binary_masks[cur_idx]
                cur_cls_score = cls_scores[cur_idx].item()
                cur_cls_label = cls_labels[cur_idx].item()

                is_thing = cur_cls_label < self.num_things_classes
                is_confident = (is_thing and cur_cls_score > cls_threshold_thing) or ((not is_thing) and cur_cls_score > cls_threshold_stuff)

                original_pixel_number = cur_binary_mask.float().sum()
                new_binary_mask = torch.logical_and(cur_binary_mask, (panoptic_seg == self.num_classes))
                new_pixel_number = new_binary_mask.float().sum()
                is_not_overlap_too_much = new_pixel_number > (original_pixel_number * overlap_threshold)

                if is_confident and is_not_overlap_too_much:
                    if is_thing:
                        current_segment_id += 1
                        panoptic_seg[new_binary_mask] = cur_cls_label + current_segment_id * INSTANCE_OFFSET
                    else:
                        panoptic_seg[new_binary_mask] = cur_cls_label

            result['pan_results'] = PixelData(sem_seg=panoptic_seg[None])

            results_list.append(result)
        results = self.add_pred_to_datasample(batch_data_samples, results_list)

        return results

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[dict]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (List[dict]): Instance segmentation, segmantic
                segmentation and panoptic segmentation results.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                data_sample.pred_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                data_sample.pred_instances = pred_results['ins_results']

            assert 'sem_results' not in pred_results, 'segmantic ' \
                'segmentation results are not supported yet.'

        return data_samples

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            tuple[List[Tensor]]: A tuple of features from ``panoptic_head``
            forward.
        """
        feats = self.extract_feat(batch_inputs)
        results = self.panoptic_head.forward(feats, batch_data_samples)
        return results