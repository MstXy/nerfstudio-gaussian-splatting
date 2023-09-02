import os
import json
import numpy as np
import torch
import math
import torchvision
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.fields.gaussian_splatting_field import GaussianSplattingField
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from nerfstudio.utils.gaussian_splatting_sh_utils import eval_sh
from nerfstudio.cameras.gaussian_splatting_camera import Camera as GaussianSplattingCamera
from nerfstudio.utils.gaussian_splatting_graphics_utils import getWorld2View2, focal2fov, fov2focal


@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    _target: Type = field(
        default_factory=lambda: GaussianSplatting
    )

    background_color: str = "black"

    sh_degree: int = 3


class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


class GaussianSplatting(Model):
    config: GaussianSplattingModelConfig
    model_path: str
    load_iteration: int
    ref_orientation: str
    orientation_transform: torch.Tensor
    appearance_name: str
    appearance_values: tuple = None
    appearance: Tuple[torch.Tensor, torch.Tensor]
    gaussian_model: GaussianSplattingField

    def __init__(
            self,
            config: ModelConfig,
            scene_box: SceneBox,
            num_train_data: int,
            model_path: str = None,
            load_iteration: int = -1,
            ref_orientation: str = None,
            appearance_name: str = None,
            appearance_values: tuple = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.load_iteration = load_iteration
        self.ref_orientation = ref_orientation
        self.appearance_name = appearance_name
        self.appearance_values = appearance_values
        self.pipeline_params = PipelineParams()
        if self.config.background_color == "black":
            self.bg_color = [0, 0, 0]
        else:
            self.bg_color = [1, 1, 1]

        super().__init__(config, scene_box, num_train_data)

    def populate_modules(self):

        super().populate_modules()

        self.orientation_transform = self.get_orientation_transform()

        # get iteration
        if self.load_iteration == -1:
            self.load_iteration = self.search_for_max_iteration(os.path.join(self.model_path, "point_cloud"))
        print("Loading trained model at iteration {}".format(self.load_iteration))

        # get appearance
        if self.appearance_name is not None:
            # load appearance model
            appearance_model_checkpoint_path = os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_{}".format(self.load_iteration),
                "appearance_embedding.ckpt",
            )
            appearance_checkpoint = torch.load(appearance_model_checkpoint_path)

            from nerfstudio.field_components.gaussian_splatting_appearance import GaussianSplattingAppearanceModel
            appearance_model = GaussianSplattingAppearanceModel(**appearance_checkpoint["model_config"])
            appearance_model.load_state_dict(appearance_checkpoint["model_state_dict"])
            appearance_model = appearance_model.to("cuda")

            # get appearance input
            normalized_appearance_embedding_path = os.path.join(self.model_path, "normalized_appearance_embedding.pt")
            normalized_appearance_embedding = torch.load(normalized_appearance_embedding_path)
            appearance_embedding = self.appearance_name
            with torch.no_grad():
                # map user input to normalized value
                if appearance_embedding not in normalized_appearance_embedding_path:
                    # the key may be int
                    try:
                        appearance_embedding = int(appearance_embedding)
                    except:
                        pass
                normalized_appearance_embedding_value = normalized_appearance_embedding[appearance_embedding]

                with torch.no_grad():
                    self.appearance = appearance_model.get_appearance(normalized_appearance_embedding_value)
                print("Use appearance embedding {} -> {} -> {}".format(
                    appearance_embedding,
                    normalized_appearance_embedding_value,
                    torch.concat(self.appearance, dim=0).reshape((-1,)).cpu().numpy(),
                ))
        elif self.appearance_values[0] > 0:
            print("Use direct appearance embedding values {}".format(self.appearance_values))
            self.appearance = (
                torch.tensor(self.appearance_values[:3], dtype=torch.float).reshape((-1, 1, 1)),
                torch.tensor(self.appearance_values[3:], dtype=torch.float).reshape((-1, 1, 1)),
            )
        else:
            self.appearance = None

        # load gaussian model
        self.gaussian_model = GaussianSplattingField(sh_degree=self.config.sh_degree)

        self.gaussian_model.load_ply(os.path.join(self.model_path,
                                                  "point_cloud",
                                                  "iteration_" + str(self.load_iteration),
                                                  "point_cloud.ply"))

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
            ],
            )

        # get camera rotation
        rotation = np.eye(4)
        rotation[:3, :3] = np.asarray(ref_camera["rotation"])
        rotation[:3, 1:3] *= -1

        transform = np.matmul(rotation, rx(-np.pi / 2))

        return torch.tensor(transform, dtype=torch.float)

    @staticmethod
    def search_for_max_iteration(folder):
        saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
        return max(saved_iters)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        viewpoint_camera = self.ns2gs_camera(camera_ray_bundle.camera)

        background = torch.tensor(self.bg_color, dtype=torch.float32, device=camera_ray_bundle.origins.device)

        render_results = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            bg_color=background,
        )

        render = render_results["render"]

        # apply appearance transformation
        if self.appearance is not None:
            grayscale_factors, gamma = self.appearance
            render = torch.pow(render, gamma.to(render.device))
            render = render * grayscale_factors.to(render.device)

        rgb = torch.permute(torch.clamp(render, max=1.), (1, 2, 0))
        return {
            "rgb": rgb,
        }

    def ns2gs_camera(self, ns_camera):
        c2w = torch.clone(ns_camera.camera_to_worlds)
        c2w = torch.concatenate([c2w, torch.tensor([[0, 0, 0, 1]], device=ns_camera.camera_to_worlds.device)], dim=0)

        # reorient
        if self.orientation_transform is not None:
            c2w = torch.matmul(self.orientation_transform.to(c2w.device), c2w)

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w.cpu().numpy())
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        FovY = focal2fov(ns_camera.fy, ns_camera.height)
        FovX = focal2fov(ns_camera.fx, ns_camera.width)

        return GaussianSplattingCamera(
            R=R,
            T=T,
            width=ns_camera.width,
            height=ns_camera.height,
            FoVx=FovX,
            FoVy=FovY,
            data_device=ns_camera.camera_to_worlds.device,
        )

    @staticmethod
    def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii}