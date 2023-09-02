# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A pipeline that dynamically chooses the number of rays to sample.
"""
import os
import json

from dataclasses import dataclass, field
from typing import Literal, Type, Optional

import numpy as np
import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.data.datamanagers.gaussian_splatting_datamanager import GaussianSplattingDatamanager, \
    GaussianSplattingDatamanagerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipelineConfig


@dataclass
class GaussianSplattingPipelineConfig(VanillaPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingPipeline)


class GaussianSplattingPipeline(Pipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    config: GaussianSplattingPipelineConfig
    datamanager: GaussianSplattingDatamanager
    dynamic_num_rays_per_batch: int

    def __init__(
            self,
            config: VanillaPipelineConfig,
            device: str,
            ref_orientation,
            **kwargs,
    ):
        super().__init__()
        self.model_path = kwargs["model_path"]
        self.ref_orientation = ref_orientation
        orientation_transform = self.get_orientation_transform()
        kwargs["orientation_transform"] = orientation_transform

        self.datamanager = GaussianSplattingDatamanager(kwargs["model_path"], orientation_transform)
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            **kwargs,
        )
        self.model.to(device)

    def get_eval_image_metrics_and_images(self, step: int):
        pass

    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        pass

    def get_param_groups(self):
        return {}

    def get_orientation_transform(self):
        if self.ref_orientation is None:
            return None

        # load camera information
        cameras_json_path = os.path.join(self.model_path, "cameras.json")
        if os.path.exists(cameras_json_path) is False:
            return None
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)

        # find specific camera by image name
        ref_camera = None
        for i in cameras:
            if i["img_name"] != self.ref_orientation:
                continue
            ref_camera = i
            break
        if ref_camera is None:
            raise ValueError("camera {} not found".format(self.ref_orientation))

        def rx(theta):
            return np.matrix([
                [1, 0, 0, 0],
                [0, np.cos(theta), -np.sin(theta), 0],
                [0, np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, 1]
            ])

        # get camera rotation
        rotation = np.eye(4)
        rotation[:3, :3] = np.asarray(ref_camera["rotation"])
        rotation[:3, 1:3] *= -1

        transform = np.matmul(rotation, rx(-np.pi / 2))

        return torch.tensor(transform, dtype=torch.float)
