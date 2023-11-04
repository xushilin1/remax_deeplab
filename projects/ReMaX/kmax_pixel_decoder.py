from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from mmengine.model import trunc_normal_init as trunc_normal_
from torch.cuda.amp import autocast
from mmengine.model import (BaseModule, ModuleList, caffe2_xavier_init,
                            normal_init, xavier_init)

from .attention import ResizedFuse, BlockGroup, LayerNorm

from mmdet.utils import ConfigType, OptMultiConfig

from mmdet.registry import MODELS

@MODELS.register_module()
class kMaXPixelDecoder(BaseModule):
    
    def __init__(
        self,
        in_channels: List[int] = [2048, 1024, 512, 256],
        dec_layers: List[int] = [1, 5, 1, 1],
        dec_channels: List[int] = [512, 256, 128, 64],
        layer_types: List[str] = ['axial', 'axial', 'bottleneck', 'bottleneck'],
        drop_path_prob: float = 0.0,
        spatial_shape: List[int] = [1281, 1281],
        # spatial_shape: List[int] = [1024, 1024],
        init_cfg: OptMultiConfig = None
    ):       
        super().__init__(init_cfg=init_cfg)

        self.num_stages = len(dec_channels)
        
        # input_shape = sorted(input_shape.items(), key=lambda x: -x[1].stride)
        # self.in_features = [k for k, v in input_shape]  # starting from "res5" to "res2"/"stem"
        # in_channels = [v.channels for k, v in input_shape]

        add_one = (spatial_shape[0] % 2, spatial_shape[1] % 2)
        query_shape = [(spatial_shape[0] // 32 + add_one[0],
                        spatial_shape[1] // 32 + add_one[1]),
                       (spatial_shape[0] // 16 + add_one[0],
                        spatial_shape[1] // 16 + add_one[1]),
                       (spatial_shape[0] // 8 + add_one[0],
                        spatial_shape[1] // 8 + add_one[1]),
                       (spatial_shape[0] // 4 + add_one[0],
                        spatial_shape[1] // 4 + add_one[1]),
                       (spatial_shape[0] // 2 + add_one[0],
                        spatial_shape[1] // 2 + add_one[1])]

        self._in_norms = nn.ModuleList()
        self._stages = nn.ModuleList()
        self._resized_fuses = nn.ModuleList()

        for i in range(self.num_stages):
            self._in_norms.append(
                LayerNorm(in_channels[i], data_format="channels_first"))
            inplanes = in_channels[i] if i == 0 else dec_channels[i]
            self._stages.append(
                BlockGroup(inplanes=inplanes,
                           base_filter=dec_channels[i],
                           num_blocks=dec_layers[i],
                           block_type=layer_types[i],
                           query_shape=query_shape[i],
                           key_expansion=1,
                           value_expansion=2,
                           num_heads=8,
                           drop_path_prob=drop_path_prob))

            if i > 0:
                self._resized_fuses.append(
                    ResizedFuse(low_in_channels=dec_channels[i - 1] * 4,
                                high_in_channels=in_channels[i],
                                out_channels=dec_channels[i]))

    def forward(self, features):

        features = features[::-1]   # from "res5" to "res2"

        out = []
        multi_scale_features = []
        x = self._in_norms[0](features[0])

        for idx in range(self.num_stages - 1):
            x = self._stages[idx](x)
            out.append(x)
            x = self._resized_fuses[idx](
                lowres_x=x,
                highres_x=self._in_norms[idx + 1](features[idx+1])
            )

        x = self._stages[-1](x)
        out.append(x)
        multi_scale_features = out[:3]
        panoptic_features = out[-1]
        semantic_features = [features[0], features[2], features[3]] # res5, res3, res2
        return panoptic_features, semantic_features, multi_scale_features
