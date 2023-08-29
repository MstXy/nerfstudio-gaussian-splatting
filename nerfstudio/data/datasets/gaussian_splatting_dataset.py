"""
Dataset.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path
from nerfstudio.data.datasets.base_dataset import InputDataset


class GaussianSplattingDataset(InputDataset):

    def __init__(self):
        Dataset().__init__()

        self.image = torch.ones(800, 800, 3, dtype=torch.float32)
        self.cameras = Cameras(
            fx=1111.,
            fy=1111.,
            cx=400.,
            cy=400.,
            distortion_params=torch.tensor([0., 0., 0., 0., 0.], dtype=torch.float32),
            height=800,
            width=800,
            camera_to_worlds=torch.tensor([
                [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                ]
            ], dtype=torch.float32),
            camera_type=CameraType.PERSPECTIVE,
        )
        self.metadata = {}

        aabb_scale = 1.
        self.scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

    def __len__(self):
        return 1

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        return self.image.cpu().numpy()

    def get_image(self, image_idx: int):
        return self.image

    def get_data(self, image_idx: int) -> Dict:
        return {
            "image_idx": image_idx,
            "image": self.image,
        }

    def get_metadata(self, data: Dict) -> Dict:
        return {}

    @property
    def image_filenames(self) -> List[Path]:
        return [Path("01.png")]
