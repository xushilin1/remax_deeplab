import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmengine.model import ModuleList, caffe2_xavier_init
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean
from mmdet.models.layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.dense_heads import Mask2FormerHead
from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,
                            normal_init, xavier_init)
from mmdet.models.utils import multi_apply

from .kmax_transformer import kMaXTransformerLayer, kMaXPredictor, SemanticPredictor
from .attention import ConvBN
from .utils import pixelwise_insdis_loss, dice_loss, softmax_ce_loss, focal_cross_entropy_loss, aux_semantic_loss
from .kmax_transformer import ASPP

@MODELS.register_module()
class kMaXDeepLabHead(Mask2FormerHead):
    def __init__(self,
                 remax=False,
                 num_remax=4,
                 dec_layers: List[int] = [2, 2, 2],
                 in_channels: List[int] = [2048, 1024, 512, 256],
                 drop_path_prob: float = 0.2,
                 add_aux_semantic_pred: bool = False,
                 use_aux_semantic_decoder: bool = True,
                 feat_channels: int = 256,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 128,
                 pixel_decoder: ConfigType = ...,
                 loss_cls: ConfigType = None,
                 loss_mask: ConfigType = None,
                 loss_dice: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.remax = remax
        self.num_remax = num_remax
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.feat_channels = feat_channels
        self.num_queries = num_queries
        self.eta = 0.1

        os2channels = {
            32: in_channels[0],
            16: in_channels[1],
            8: in_channels[2]
        }
        self._cluster_centers = nn.Embedding(256, num_queries)
        self._kmax_transformer_layers = nn.ModuleList()
        self._num_blocks = dec_layers

        for index, output_stride in enumerate([32, 16, 8]):
            for _ in range(self._num_blocks[index]):
                self._kmax_transformer_layers.append(
                    kMaXTransformerLayer(
                        num_classes=self.num_classes + 1,
                        in_channel_pixel=os2channels[output_stride],
                        in_channel_query=256,
                        base_filters=128,
                        num_heads=8,
                        bottleneck_expansion=2,
                        key_expansion=1,
                        value_expansion=2,
                        drop_path_prob=drop_path_prob))
        if self.remax:
            cur_idx = 0
            self._remax_sem_pred_layers = nn.ModuleList()
            for index, output_stride in enumerate([32, 16, 8]):
                for _ in range(self._num_blocks[index]):
                    if cur_idx == self.num_remax:
                        break
                    self._remax_sem_pred_layers.append(
                        nn.Sequential(
                            ASPP(
                                in_channels=os2channels[output_stride], 
                                output_channels=256,
                                atrous_rates=[6,12,18]),
                            ConvBN(256, self.num_classes+1, kernel_size=1, norm=None, act=None)
                    ))
                    cur_idx += 1

        self._class_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',conv_type='1d')
        self._mask_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu', conv_type='1d')

        self._predcitor = kMaXPredictor(in_channel_pixel=in_channels[-1],
                                        in_channel_query=256,
                                        num_classes=self.num_classes + 1)

        self._add_aux_semantic_pred = add_aux_semantic_pred
        self._use_aux_semantic_decoder = use_aux_semantic_decoder
        if add_aux_semantic_pred:
            if use_aux_semantic_decoder:
                self._auxiliary_semantic_predictor = SemanticPredictor(
                    in_channels=in_channels[0],  # res5
                    os8_channels=in_channels[2],  # res3
                    os4_channels=in_channels[3],  # res2
                    num_classes=self.num_classes + 1)
            else:
                self._auxiliary_semantic_predictor = nn.Sequential(
                    ConvBN(in_channels[0], in_channels[0], groups=in_channels[0], kernel_size=5, padding=2, bias=False, norm='syncbn', act='gelu', conv_init='xavier_uniform'),
                    ConvBN(in_channels[0], 256, kernel_size=1,bias=False, norm='syncbn', act='gelu'),
                    ConvBN(256, self.num_classes, kernel_size=1, norm=None, act=None)
                )

        self.pixel_decoder = MODELS.build(pixel_decoder)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)

    def init_weights(self) -> None:
        pass

    def get_targets(self,
                    cls_scores,
                    mask_preds,
                    pixel_feats,
                    batch_gt_instances,
                    batch_gt_sem_segs,
                    batch_img_metas: List[dict],
                    return_sampling_results: bool = False):
        assign_result_list = []
        src_inds_list, gt_inds_list, pos_inds_list = [], [], []
        for i in range(len(batch_gt_instances)):
            pred_instances = InstanceData(scores=cls_scores[i], masks=mask_preds[i])
            assign_result = self.assigner.assign(
                pred_instances=pred_instances,
                gt_instances=batch_gt_instances[i],
                img_meta=batch_img_metas[i])
            assign_result_list.append(assign_result)
            pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
            # neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
            pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
            gt_inds_list.append(pos_assigned_gt_inds)
            pos_inds_list.append(pos_inds)
            src_inds_list.append(i * torch.ones_like(pos_inds))
        src_inds_list = torch.cat(src_inds_list)
        pos_inds_list = torch.cat(pos_inds_list)

        gt_masks_list = [gt_ins.masks for gt_ins in batch_gt_instances]
        gt_labels_list = [gt_ins.labels for gt_ins in batch_gt_instances]

        src_idx = (src_inds_list, pos_inds_list)

        mask_targets = torch.zeros_like(mask_preds)
        mask_targets_o = torch.cat([
            gt_masks_list[i][gt_inds] for i, gt_inds in enumerate(gt_inds_list)
        ]).to(mask_targets)
        mask_targets[src_idx] = mask_targets_o

        mask_targets = mask_targets / torch.clamp(mask_targets.sum(1, keepdim=True), min=1.0)
        matched_cls_prob = [
            assign_result.get_extra_property('matched_cls_prob')
            for assign_result in assign_result_list
        ]
        matched_cls_prob_o = torch.cat(matched_cls_prob)
        matched_cls_prob_o = torch.clamp(matched_cls_prob_o, min=1e-5)

        matched_cls_prob = torch.full(mask_preds.shape[:2], 0).to(mask_preds)
        matched_cls_prob[src_idx] = matched_cls_prob_o.to(matched_cls_prob)

        pixel_gt_void_mask = (mask_targets.sum(1) < 1)  # B x H x W
        mask_gt_area = mask_targets.sum(2).sum(2)  # B x N
        pixel_gt_area = torch.einsum('bnhw,bn->bhw', mask_targets, mask_gt_area)  # B x H x W
        bs, h, w = pixel_gt_area.shape
        inverse_gt_mask_area = (h * w) / torch.clamp(pixel_gt_area, min=1.0)

        labels = torch.full(cls_scores.shape[:2],
                            self.num_classes,
                            dtype=torch.int64,
                            device=cls_scores.device)

        target_classes_o = torch.cat([
            gt_labels_list[i][gt_inds]
            for i, gt_inds in enumerate(gt_inds_list)
        ]).to(labels)
        labels[src_idx] = target_classes_o

        src_masks_prob = mask_preds.detach().softmax(1)
        void_mask = pixel_gt_void_mask.to(src_masks_prob)  # B x H x W, True means there is no ground truth for this pixel

        def computer_iou_score(x, y):
            # x : B x N x H x W
            # y : B x H x W
            x = x.flatten(2)  # B x N x L
            y = y.flatten(1)  # B x L
            intersection = torch.einsum('bnl,bl->bn', x, y)  # B x N
            denominator = x.sum(-1)  # B x N
            return intersection / (denominator + 1e-5)  # B x N

        matched_dice_o = torch.cat([
            assign_result.get_extra_property('matched_dice')
            for assign_result in assign_result_list
        ])
        matched_dice = computer_iou_score(src_masks_prob, void_mask)
        matched_dice[src_idx] = matched_dice_o.to(matched_dice)
        matched_dice = torch.clamp(matched_dice, min=1e-5)

        return labels, mask_targets, matched_cls_prob, matched_dice, pixel_gt_void_mask, inverse_gt_mask_area

    def _loss_by_feat_single(self, cls_scores: Tensor, mask_preds: Tensor, sem_preds: Tensor, pixel_feature: Tensor,
                             batch_gt_instances: List[InstanceData], batch_gt_sem_seg: List[Tensor],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries, cls_out_channels),
                Note `cls_out_channels` should includes background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            sem_preds (Tensor): Mask logits for semantic label.
                Shape (batch_size, num_classes, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_gt_sem_seg (list[tensor]): each shape is (1, h, w).
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """

        ori_mask_preds = mask_preds.clone().sigmoid().detach()
        if sem_preds is not None:
            sem_preds_detach = sem_preds.sigmoid().detach()
            sem_preds_detach = torch.einsum('bnc,bchw->bnhw', cls_scores.sigmoid().detach(), sem_preds_detach)
            mask_preds = mask_preds + mask_preds * sem_preds_detach

        results = self.get_targets(cls_scores, mask_preds, pixel_feature,
                                   batch_gt_instances, batch_gt_sem_seg, batch_img_metas)
        labels, mask_targets, pq_loss_mask_weight, pq_loss_class_weight, pixel_gt_void_mask, inverse_gt_mask_area = results
        '''
            labels: (B, N). Assignment of ground truth for each prediction, contains num_classes means background.
            mask_targets: (B, N, H, W). Panoptic segmentation mask
            pq_loss_mask_weight: (B, N)
            pq_loss_class_weight: (B, N)
            pixel_gt_void_mask: (B, H, W), True means there is no thing or stuff for this pixel.
        '''

        loss_pixel = pixelwise_insdis_loss(
            pixel_feature=pixel_feature,
            gt_mask=mask_targets,
            sample_temperature=1.5,
            sample_k=4096,
            instance_discrimination_temperature=0.3,
            pixel_gt_void_mask=pixel_gt_void_mask,
            inverse_gt_mask_area=inverse_gt_mask_area
        )

        # ReClass
        one_hot_labels = F.one_hot(labels.to(torch.long), num_classes=self.num_classes + 1)     # (batch_size, num_queries, num_classes+1)
        binary_gt_sem_seg = F.one_hot(torch.cat(batch_gt_sem_seg, dim=0).to(torch.long), 
                                      num_classes=self.num_classes + 1).flatten(1, 2)     # (batch_size, hw, num_classes+1) 
        _panoptic_seg = ori_mask_preds.flatten(2)      # (batch_size, num_queries, hw)
        class_weight = torch.einsum("bnl, blc -> bnc", _panoptic_seg, binary_gt_sem_seg.to(torch.float)) / (binary_gt_sem_seg.sum(1) + 1e-5)
        labels = self.eta * class_weight + (1 - self.eta * class_weight) * one_hot_labels
        
        loss_cls = self.loss_cls.loss_weight * focal_cross_entropy_loss(cls_scores, labels, pq_loss_class_weight)

        if sem_preds is not None:
            gt_sem_seg = torch.cat(batch_gt_sem_seg).long()
            loss_sem = aux_semantic_loss(
                pred_semantic_logits=sem_preds,
                ground_truth_semantic=gt_sem_seg,
                sample_temperature=2.0,
                sample_k=4096,
                pixel_gt_void_mask=pixel_gt_void_mask,
                inverse_gt_mask_area=inverse_gt_mask_area,
                num_classes=self.num_classes
            )
        else:
            loss_sem = mask_preds.sum() * 0.0

        mask_preds = mask_preds.flatten(2)  # B x N x HW
        mask_targets = mask_targets.flatten(2)  # B x N x HW
        pixel_gt_void_mask = pixel_gt_void_mask.flatten(1)  # B x HW

        loss_mask = self.loss_mask.loss_weight * softmax_ce_loss(mask_preds, mask_targets, pixel_gt_void_mask)
        loss_dice = self.loss_dice.loss_weight * dice_loss(mask_preds, mask_targets, pixel_gt_void_mask, pq_loss_mask_weight)

        return loss_cls, loss_mask, loss_dice, loss_pixel, loss_sem

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        bs = len(batch_img_metas)
        panoptic_features, semantic_features, multi_scale_memorys = self.pixel_decoder(x)
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(bs, 1, 1)    # B x C x L
        
        tgt_size = panoptic_features.shape[-2:]
        align_corners = (tgt_size[0] % 2 == 1)
        
        cur_idx = 0

        pixel_feature_list = []
        cls_pred_list = []
        mask_pred_list = []
        sem_pred_list = []
        for i, feat in enumerate(multi_scale_memorys):
            for _ in range(self._num_blocks[i]):
                cluster_centers, prediction_result = self._kmax_transformer_layers[cur_idx](pixel_feature=feat, query_feature=cluster_centers)
                mask_logits = prediction_result['mask_logits']
                class_logits = prediction_result['class_logits']
                pixel_feature = prediction_result['pixel_feature']
                
                sem_seg_pred = None
                if self.remax and self.training and cur_idx < self.num_remax:
                    sem_seg_pred = self._remax_sem_pred_layers[cur_idx](feat)
                    sem_seg_pred = F.interpolate(
                        sem_seg_pred,
                        size=tgt_size,
                        mode="bilinear",
                        align_corners=align_corners)    # (bs, num_classes, H, W)
                
                mask_logits = F.interpolate(mask_logits,
                                            size=tgt_size,
                                            mode="bilinear",
                                            align_corners=align_corners)
                
                pixel_feature = F.interpolate(
                    pixel_feature,
                    size=tgt_size,
                    mode="bilinear",
                    align_corners=align_corners)

                cls_pred_list.append(class_logits)
                mask_pred_list.append(mask_logits)
                pixel_feature_list.append(pixel_feature)
                sem_pred_list.append(sem_seg_pred)
                cur_idx += 1

        class_embeddings = self._class_embedding_projection(cluster_centers)
        mask_embeddings = self._mask_embedding_projection(cluster_centers)

        prediction_result = self._predcitor(
            class_embeddings=class_embeddings,
            mask_embeddings=mask_embeddings,
            pixel_feature=panoptic_features,
        )
        
        cls_pred_list.append(prediction_result['class_logits'])
        mask_pred_list.append(prediction_result['mask_logits'])
        pixel_feature_list.append(prediction_result['pixel_feature'])
        sem_pred_list.append(None)

        sem_pred = None
        if self._add_aux_semantic_pred:
            semantic_features, low_features_os8, low_features_os4 = semantic_features
            if self._use_aux_semantic_decoder:
                sem_pred = self._auxiliary_semantic_predictor(
                    x=semantic_features, low_features_os8=low_features_os8, low_features_os4=low_features_os4)
            else:
                sem_pred = self._auxiliary_semantic_predictor(semantic_features)

        return cls_pred_list, mask_pred_list, pixel_feature_list, sem_pred_list
    

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList):
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        all_cls_scores, all_mask_preds, all_pixel_feats, all_sem_preds = self(x, batch_data_samples)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances, batch_gt_semantic_segs)

        for gt_instance in batch_gt_instances:
            gt_instance.masks = gt_instance.masks[:, ::4, ::4]  # upsample
        
        batch_gt_sem_seg = []
        for gt_sem_seg in batch_gt_semantic_segs:
            if gt_sem_seg is not None:
                sem_seg = gt_sem_seg.sem_seg[:,::4,::4]
                sem_seg[sem_seg == 255] = self.num_classes  # padding pixel -> 133
                batch_gt_sem_seg.append(sem_seg)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, 
                                   all_pixel_feats, all_sem_preds, 
                                   batch_gt_instances, batch_gt_sem_seg,
                                   batch_img_metas)
        
        # if self._add_aux_semantic_pred:
        #     losses['loss_aux_semantic'] = aux_semantic_loss(
        #         pred_semantic_logits=sem_preds,
        #         ground_truth_semantic=ground_truth_semantic,
        #         sample_temperature=2.0,
        #         sample_k=4096,
        #         pixel_gt_void_mask=pixel_gt_void_mask,
        #         inverse_gt_mask_area=inverse_gt_mask_area,
        #         num_classes=self.num_classes
        # )

        return losses

    def loss_by_feat(self, all_cls_scores: List[Tensor], all_mask_preds: List[Tensor],
                     all_pixel_feats: List[Tensor], all_sem_preds: List[Tensor],
                     batch_gt_instances: List[InstanceData], batch_gt_sem_seg: List[Tensor],
                     batch_img_metas: List[dict]):
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        batch_gt_sem_seg_list = [batch_gt_sem_seg for _ in range(num_dec_layers)]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice, losses_pixel, losses_sem = multi_apply(
            self._loss_by_feat_single, all_cls_scores, all_mask_preds, all_sem_preds,
            all_pixel_feats, batch_gt_instances_list, batch_gt_sem_seg_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_pixel'] = losses_pixel[-1]
        loss_dict['loss_sem'] = losses_sem[3]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_pixel_i, loss_sem_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_pixel[:-1], losses_sem[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_pixel'] = loss_pixel_i
            loss_dict[f'd{num_dec_layer}.loss_sem'] = loss_sem_i
            num_dec_layer += 1
        return loss_dict

    def predict(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> Tuple[Tensor]:
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        # all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        all_cls_scores, all_mask_preds, all_pixel_feats, sem_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # upsample masks
        img_shape = batch_img_metas[0]['batch_input_shape']
        align_corners = (img_shape[-1] % 2 == 1)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=align_corners)

        return mask_cls_results, mask_pred_results