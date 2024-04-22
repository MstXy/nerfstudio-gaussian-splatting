import os
import json
import numpy as np
import torch
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.viewer.server.viewer_elements import *
from nerfstudio.fields.gaussian_splatting_field import GaussianSplattingField
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from nerfstudio.utils.gaussian_splatting_sh_utils import eval_sh
from nerfstudio.cameras.gaussian_splatting_camera import Camera as GaussianSplattingCamera
from nerfstudio.utils.gaussian_splatting_graphics_utils import getWorld2View2, focal2fov, fov2focal

from scipy import interpolate, optimize

# import pdb

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
    complex: str
    load_iteration: int
    ref_orientation: str
    orientation_transform: torch.Tensor
    gaussian_model: GaussianSplattingField

    def __init__(
            self,
            config: ModelConfig,
            scene_box: SceneBox,
            num_train_data: int,
            model_path: str = None,
            complex: str = "False",
            load_iteration: int = -1,
            orientation_transform: torch.Tensor = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.complex_trajectory = (complex == "True")
        print("Complex Trajectory: {}".format(self.complex_trajectory))
        self.load_iteration = load_iteration
        self.orientation_transform = orientation_transform
        self.pipeline_params = PipelineParams()
        if self.config.background_color == "black":
            self.bg_color = [0, 0, 0]
        else:
            self.bg_color = [1, 1, 1]
        self.current_block = None
        self.block_info = None
        if self.complex_trajectory:
            self.next_block = None
            self.spl_tmp = None
            self.spl_computing = False

        super().__init__(config, scene_box, num_train_data)

        self.scaling_modifier_slider = ViewerSlider(name="Scaling Modifier", default_value=1.0, min_value=0.0, max_value=1.0)

        # set FOV: fixed bug in nerfstudio/viewer/server/viewer_state.py:L171
        self.viewer_control = ViewerControl()  
        ## use a slider to control
        def change_fov(slider):
            self.viewer_control.set_fov(slider.value)
        self.viewer_slider = ViewerSlider(name="Viewer FOV", default_value=80, min_value=20, max_value=120, step=1, cb_hook=change_fov)
        
        ## For test: 
        def on_change_callback(handle: ViewerCheckbox) -> None:
            print(handle.value)
            self.load_block(block_idx=1 if not handle.value else 0)

        self.custom_checkbox = ViewerCheckbox(
            name="Change Scene",
            default_value=False,
            cb_hook=on_change_callback,
        )
        
        # # get width & height, fx & fy from model_path
        # with open(os.path.join(self.model_path, "cameras.json"), "r") as f:
        #     camera_data = json.load(f)
        # self._gs_height = camera_data[0]["height"]
        # self._gs_width = camera_data[0]["width"]
        # self._gs_fx = camera_data[0]["fx"]
        # self._gs_fy = camera_data[0]["fy"]

    def populate_modules(self):
        super().populate_modules()

        # initialize gaussian model
        self.gaussian_model = GaussianSplattingField(sh_degree=self.config.sh_degree)

        # TODO: add param here.
        self.get_block_info()
        self.load_block(block_idx=0)
    
    def get_block_info(self):
        with open(os.path.join(self.model_path, "blocks.json"), 'r') as f:
            self.block_info = json.load(f)
            self.total_blocks = len(self.block_info)
            print("Total blocks: {}".format(self.total_blocks))
        if self.complex_trajectory:
            self.tcks = []
            for i in range(self.total_blocks):
                tck = [np.array(self.block_info["block_"+str(i)]["t"]), np.array(self.block_info["block_"+str(i)]["c"]), self.block_info["block_"+str(i)]["k"]]
                self.tcks.append(tck)

    def load_block(self, block_idx: int):
        if block_idx == self.current_block:
            return 
        self.current_block = block_idx
        print("Loading block {}".format(block_idx))
        block_path = os.path.join(self.model_path, "block_{}".format(block_idx))
        # get iteration
        if self.load_iteration == -1:
            self.load_iteration = self.search_for_max_iteration(os.path.join(block_path, "point_cloud"))
        print("Loading trained model at iteration {}".format(self.load_iteration))

        # load block
        self.gaussian_model.load_ply(path=os.path.join(block_path,
                                                  "point_cloud",
                                                  "iteration_" + str(self.load_iteration),
                                                  "point_cloud.ply"))

    def check_block(self, camera_position: torch.Tensor):
        device = camera_position.device
        
        if not self.complex_trajectory: # simple straight trajectory
            eps = 0.5
            this_centroid = torch.tensor(self.block_info["block_{}".format(self.current_block)]["center"], device=device)
            # check if current camera position is close to the left boundary
            left_anchor = torch.tensor(self.block_info["block_{}".format(self.current_block)]["anchor"], device=device)
            left_plane_normal = torch.nn.functional.normalize(left_anchor - this_centroid, dim=0)
            if not self.current_block == 0 and torch.linalg.norm(torch.dot(left_anchor - camera_position, left_plane_normal)) < eps:
                # now check if crossed the left boundary
                prev_centroid = torch.tensor(self.block_info["block_{}".format(self.current_block - 1)]["center"], device=device)
                prev_dist = torch.nn.functional.cosine_similarity(camera_position - left_anchor, prev_centroid - left_anchor, dim=0)
                this_dist = torch.nn.functional.cosine_similarity(camera_position - left_anchor, this_centroid - left_anchor, dim=0)
                if prev_dist > this_dist:
                    self.load_block(self.current_block - 1)
                    return
                
            # check if current camera position is close to the right boundary
            if not self.current_block == self.total_blocks - 1:
                right_anchor = torch.tensor(self.block_info["block_{}".format(self.current_block + 1)]["anchor"], device=device)
                right_plane_normal = torch.nn.functional.normalize(right_anchor - this_centroid, dim=0)
                if torch.linalg.norm(torch.dot(right_anchor - camera_position, right_plane_normal)):
                    # now check if crossed the left boundary
                    next_centroid = torch.tensor(self.block_info["block_{}".format(self.current_block + 1)]["center"], device=device)
                    this_dist = torch.nn.functional.cosine_similarity(camera_position - right_anchor, this_centroid - right_anchor, dim=0)
                    next_dist = torch.nn.functional.cosine_similarity(camera_position - right_anchor, next_centroid - right_anchor, dim=0)
                    if next_dist > this_dist:
                        self.load_block(self.current_block + 1)
                        return
        elif not self.spl_computing: # complex trajectory
            self.spl_computing = True
            eps = 0.25
            camera_position = camera_position.cpu().numpy().reshape(3,1)

            dist_0 = float("inf")
            dist_2 = float("inf")

            if self.current_block > 0:
                self.spl_tmp = 0
                closestu_0_0 = optimize.minimize(self.distToP, 0.95, args=(self.tcks[self.current_block-1],camera_position), bounds=[(0,1)])
                closestu_0_1 = optimize.minimize(self.distToP, 0.5, args=(self.tcks[self.current_block-1],camera_position), bounds=[(0,1)])                
                closestu_0_2 = optimize.minimize(self.distToP, 0.05, args=(self.tcks[self.current_block-1],camera_position), bounds=[(0,1)])                
                # closestu_0_3 = optimize.minimize(self.distToP, 0.05, args=(self.tcks[self.current_block-1],camera_position), bounds=[(0,1)])                
                dist_0 = np.array([closestu_0_0.fun, closestu_0_1.fun, closestu_0_2.fun]).min()
                
                # dist_0 = optimize.basinhopping(self.distToP, 0.5, minimizer_kwargs=dict(args=(self.tcks[self.current_block-1],camera_position), bounds=[(0,1)]), niter=6).fun

            self.spl_tmp = 0
            closestu_1_0 = optimize.minimize(self.distToP, 0.05, args=(self.tcks[self.current_block],camera_position), bounds=[(0,1)])
            closestu_1_1 = optimize.minimize(self.distToP, 0.35, args=(self.tcks[self.current_block],camera_position), bounds=[(0,1)])
            closestu_1_2 = optimize.minimize(self.distToP, 0.65, args=(self.tcks[self.current_block],camera_position), bounds=[(0,1)])
            closestu_1_3 = optimize.minimize(self.distToP, 0.95, args=(self.tcks[self.current_block],camera_position), bounds=[(0,1)])
            dist_1 = np.array([closestu_1_0.fun, closestu_1_1.fun, closestu_1_2.fun, closestu_1_3.fun]).min()

            # dist_1 = optimize.basinhopping(self.distToP, 0.5, minimizer_kwargs=dict(args=(self.tcks[self.current_block],camera_position), bounds=[(0,1)]), niter=6).fun

            if self.current_block < self.total_blocks - 1:
                self.spl_tmp = 0
                closestu_2_0 = optimize.minimize(self.distToP, 0.05, args=(self.tcks[self.current_block+1],camera_position), bounds=[(0,1)])
                closestu_2_1 = optimize.minimize(self.distToP, 0.5, args=(self.tcks[self.current_block+1],camera_position), bounds=[(0,1)])                
                closestu_2_2 = optimize.minimize(self.distToP, 0.95, args=(self.tcks[self.current_block+1],camera_position), bounds=[(0,1)])                
                # closestu_2_3 = optimize.minimize(self.distToP, 0.95, args=(self.tcks[self.current_block+1],camera_position), bounds=[(0,1)])                                
                dist_2 = np.array([closestu_2_0.fun, closestu_2_1.fun, closestu_2_2.fun]).min()

                # dist_2 = optimize.basinhopping(self.distToP, 0.5, minimizer_kwargs=dict(args=(self.tcks[self.current_block+1],camera_position), bounds=[(0,1)]), niter=6).fun
            
            closest_block = np.array([dist_0 + eps, dist_1, dist_2 + eps]).argmin() - 1 # -1, 0, 1 -> prev, curr, next
            if closest_block != 0: # need to load new block
                # but first, check consistency.
                if closest_block != self.next_block:
                    self.next_block = closest_block
                else:
                    print(camera_position.reshape(3))
                    self.load_block(self.current_block + closest_block)
                    self.next_block = None
            self.spl_computing = False

    def distToP(self, u, tck, cam_point):
        s = np.array(interpolate.splev(u, tck))
        self.spl_tmp = s - cam_point
        return np.linalg.norm(self.spl_tmp)

    @staticmethod
    def search_for_max_iteration(folder):
        saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
        return max(saved_iters)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        # pdb.set_trace() 
        ## function called after opening up viewer (in localhost) 
        ## in viewer/server/render_state_machine.py(148)_render_img()
        ## -> outputs = self.viewer.get_model().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        viewpoint_camera = self.ns2gs_camera(camera_ray_bundle.camera)

        background = torch.tensor(self.bg_color, dtype=torch.float32, device=camera_ray_bundle.origins.device)

        render_results = self.render(
            viewpoint_camera=viewpoint_camera,
            pc=self.gaussian_model,
            pipe=self.pipeline_params,
            bg_color=background,
            scaling_modifier=self.scaling_modifier_slider.value,
        )

        render = render_results["render"]

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

        # FovY = focal2fov(self._gs_fy, ns_camera.height)
        # FovX = focal2fov(self._gs_fx, ns_camera.height)

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
