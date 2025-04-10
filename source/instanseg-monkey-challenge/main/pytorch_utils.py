
import torch
import torch.nn.functional as F
import monai
from typing import Tuple, List, Union
from instanseg.utils.pytorch_utils import torch_sparse_onehot

import pdb

def centroids_from_lab(lab: torch.Tensor):

    mesh_grid = torch.stack(torch.meshgrid(torch.arange(lab.shape[-2], device = lab.device), torch.arange(lab.shape[-1],device = lab.device), indexing="ij")).float()

    sparse_onehot, label_ids = torch_sparse_onehot(lab, flatten=True)

    sum_centroids = torch.sparse.mm(sparse_onehot, mesh_grid.flatten(1).T)

    centroids = sum_centroids / torch.sparse.sum(sparse_onehot, dim=(1,)).to_dense().unsqueeze(-1)

    return centroids, label_ids  # N,2  N


def get_patches(lab: torch.Tensor, image: torch.Tensor, patch_size: int = 64, return_lab_ids: bool = False):
    # lab is 1,H,W with N objects
    # image is C,H,W

    # Returns N,C,patch_size,patch_size

    centroids, label_ids = centroids_from_lab(lab)
    N = centroids.shape[0]

    C, h, w = image.shape[-3:]

    window_size = patch_size // 2
    centroids = centroids.clone()  # N,2
    centroids[:, 0] = centroids[:,0].clamp(min=window_size, max=h - window_size)
    centroids[:, 1] = centroids[:,1].clamp(min=window_size, max=w - window_size)
    window_slices = centroids[:, None] + torch.tensor([[-1, -1], [1, 1]]).to(image.device) * window_size
    window_slices = window_slices.long()  # N,2,2

    slice_size = window_size * 2

    # Create grids of indices for slice windows
    grid_x, grid_y = torch.meshgrid(
        torch.arange(slice_size, device=image.device),
        torch.arange(slice_size, device=image.device), indexing="ij")
    mesh = torch.stack((grid_x, grid_y))

    mesh_grid = mesh.expand(N, 2, slice_size, slice_size)  # N,2,2*window_size,2*window_size
    mesh_flat = torch.flatten(mesh_grid, 2).permute(1, 0, -1)  # 2,N,2*window_size*2*window_size
    idx = window_slices[:, 0].permute(1, 0)[:, :, None]
    mesh_flat = mesh_flat + idx
    mesh_flater = torch.flatten(mesh_flat, 1)  # 2,N*2*window_size*2*window_size


    out = image[:, mesh_flater[0], mesh_flater[1]].reshape(C, N, -1)
    out = out.reshape(C, N, patch_size, patch_size)
    out = out.permute(1, 0, 2, 3)


    if return_lab_ids:
        return out, label_ids

    return out,label_ids  # N,C,patch_size,patch_size


def get_masked_patches(lab: torch.Tensor, image: torch.Tensor, patch_size: int = 64):
    # lab is 1,H,W
    # image is C,H,W

    lab_patches, label_ids = get_patches(lab, lab[0], patch_size)
    mask_patches = lab_patches == label_ids[1:, None, None, None]

    image_patches,_ = get_patches(lab, image, patch_size)

    return image_patches,mask_patches  # N,C,patch_size,patch_size