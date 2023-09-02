from typing import Tuple

from dataclasses import dataclass, field, fields

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.datamanagers.gaussian_splatting_datamanager import GaussianSplattingDatamanagerConfig
from nerfstudio.data.datasets.gaussian_splatting_dataset import GaussianSplattingDataset
from nerfstudio.pipelines.gaussian_splatting_pipeline import GaussianSplattingPipelineConfig
from nerfstudio.models.gaussian_splatting import GaussianSplattingModelConfig


@dataclass
class GaussianSplattingConfig:
    config: TrainerConfig = TrainerConfig(
        method_name="gaussian_splatting",
        steps_per_eval_batch=999999999,
        steps_per_save=999999999,
        max_num_iterations=999999999,
        mixed_precision=True,
        pipeline=GaussianSplattingPipelineConfig(
            datamanager=GaussianSplattingDatamanagerConfig(),
            model=GaussianSplattingModelConfig(
                eval_num_rays_per_chunk=1,
            ),
        ),
        optimizers={},
        viewer=ViewerConfig(),
        vis="viewer",
    )

    model_path: str = None

    load_iteration: int = -1

    ref_orientation: str = None

    appearance_name: str = None

    appearance_values: Tuple[float, float, float, float] = (-1., -1., -1., -1.)

    def get_pipeline_setup_arguments(self):
        return {
            "model_path": str(self.model_path),
            "load_iteration": self.load_iteration,
            "ref_orientation": self.ref_orientation,
            "appearance_name": self.appearance_name,
            "appearance_values": self.appearance_values,
        }

    def setup_pipeline(self):
        return self.config.pipeline.setup(
            device="cuda",
            **self.get_pipeline_setup_arguments(),
        )
