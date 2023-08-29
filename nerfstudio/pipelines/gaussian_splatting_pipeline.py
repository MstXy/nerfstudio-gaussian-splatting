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

from dataclasses import dataclass, field
from typing import Literal, Type, Optional

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
            model_path: str = None,
            load_iteration: int = -1,
    ):
        super().__init__()

        self.datamanager = GaussianSplattingDatamanager()
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            model_path=model_path,
            load_iteration=load_iteration,
        )
        self.model.to(device)

    def get_eval_image_metrics_and_images(self, step: int):
        pass

    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        pass

    def get_param_groups(self):
        return {}
