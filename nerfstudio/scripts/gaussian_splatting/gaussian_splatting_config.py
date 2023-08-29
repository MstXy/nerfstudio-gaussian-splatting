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
