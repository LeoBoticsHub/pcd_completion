#!/usr/bin/env python
'''----------------------------------------------------------------------------------------------------------------------------------
# Copyright (C) 2022
#
# author: Saber Mohammadi, Andrea Zunino
# mail: saber.mohammadi1986@gmail.com
#
# Institute: Leonardo Labs (Leonardo S.p.a - Istituto Italiano di tecnologia)
#
# This file is part of ai_utils. <https://github.com/IASRobolab/ai_utils>
#
# ai_utils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ai_utils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License. If not, see http://www.gnu.org/licenses/
---------------------------------------------------------------------------------------------------------------------------------'''


from Completion_inf import builder
import torch
import open3d as o3d
import numpy as np
from utils.config import *
from utils.logger import *
from utils import parser
from vedo import Points, printc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def get_voxel_center_coordinate(voxel_grid):
    voxels = voxel_grid.get_voxels()
    voxel_center = []

    for voxel in voxels:
        voxel_center.append(voxel_grid.get_voxel_center_coordinate(voxel.grid_index))

    return voxel_center


class PCD_completion:

    def __init__(self, model_weights) -> None:
        # Load parameters
        self.args = parser.get_args()
        self.args.use_gpu = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        self.args.distributed = False
        self.args.ckpts = model_weights
        #Load model
        self.logger = get_logger('SGrasp')
        self.config = get_config(self.args, logger=self.logger)
        self.base_model = builder.model_builder(self.config.model)
        builder.load_model(self.base_model, self.args.ckpts, logger=self.logger)

    def comp_inf(self, pcd_in):
        self.base_model.eval().cuda()  # set model to eval mode
        with torch.no_grad():

            #Load partial Point Cloud
            input_pcd = o3d.io.read_point_cloud(pcd_in)
            input_pcd_xyz = np.asarray(input_pcd.points)

            #Down Sample Point Cloud
            pcd_point_sampled = farthest_point_sample(input_pcd_xyz, 2048)

            # Normalise Point Cloud
            centroid = np.mean(pcd_point_sampled, axis=0)
            pc = pcd_point_sampled - centroid
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            partial = pc / m

            # Convert Point Cloud to tensor
            partial = torch.tensor(partial).float().cuda()
            partial = torch.unsqueeze(partial, 0)

            #Normalise Partial Point Cloud to completion network which outputs completed Point Cloud
            ret = self.base_model(partial)
            dense_points = ret[3]
            dense_points = dense_points.squeeze()
            dense_points = dense_points.cpu().numpy()


            #denormalise Point Cloud to get the original size
            dense_points[:, 0] = dense_points[:, 0] * (m + (m / 6))
            dense_points[:, 1] = dense_points[:, 1] * (m + (m / 6))
            # Note that here we consider the scale of the completed point cloud in the Z axis as zero since we are only interested on the top plane of the completed point cloud to pick
            dense_points[:, 2] = dense_points[:, 2] * (m - (m))
            dense_points = dense_points + centroid

            #Post process the Completed Point Cloud to improve it
            dense_points = farthest_point_sample(dense_points, 1024)
            dense_points_o3d = o3d.geometry.PointCloud()
            dense_points_o3d.points = o3d.utility.Vector3dVector(dense_points)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(dense_points_o3d,voxel_size=0.02)
            voxel_center = get_voxel_center_coordinate(voxel_grid)
            voxel_center = np.asarray(voxel_center)
            pcd = voxel_center
            scals = np.abs(pcd[:, 1])  # let the scalar be the y of the point itself
            pts = Points(pcd, r=3)
            pts.pointdata["scals"] = scals
            densecloud = pts.densify(0.1, nclosest=100, niter=1)  # return a new pointcloud.Points
            #printc('nr. points increased', pts.N(), '\rightarrow ', densecloud.N(), c='lg')
            pcd_point = densecloud.points()
            pcd_point = pcd_point[pcd.shape[0]:]
            pcd_point = pcd_point[:-1]

            #Convert the type to open3d for visualisation
            pc_comp = o3d.geometry.PointCloud()
            pc_comp.points = o3d.utility.Vector3dVector(pcd_point)

            # visualisation of the completed Point Cloud in green and the input Point in red
            input_partial = o3d.io.read_point_cloud(pcd_in)
            input_partial.paint_uniform_color([1.0, 0, 0])
            pc_comp.paint_uniform_color([0, 1.0, 0])
            o3d.visualization.draw_geometries([pc_comp, input_partial])



        return pc_comp




