import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import numpy as np
import imageio
import skimage.measure
import util
from data import get_split_dataset
from model import make_model
from render import NeRFRenderer
import cv2
import tqdm
import warnings
import trimesh
import traceback


def extra_args(parser):
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--source", "-P", type=str, default="2")  # Always 2 views for mesh
    parser.add_argument("--eval_view_list", type=str, default=None)
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--no_compare_gt", action="store_true")
    parser.add_argument("--multicat", action="store_true")
    parser.add_argument("--viewlist", "-L", type=str, default="")
    parser.add_argument("--output", "-O", type=str, default="eval")
    parser.add_argument("--include_src", action="store_true")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--write_depth", action="store_true")
    parser.add_argument("--write_compare", action="store_true")
    parser.add_argument("--free_pose", action="store_true")
    parser.add_argument("--mesh_res", type=int, default=256)
    parser.add_argument("--mesh_thresh", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    args, conf = util.args.parse_args(
        extra_args, default_conf="conf/exp/pollen", default_expname="pollen"
    )
    args.resume = True
    device = util.get_cuda(args.gpu_id[0])

    dset = get_split_dataset(args.dataset_format, args.datadir, want_split=args.split, training=False)
    data_loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(args.output, exist_ok=True)
    finish_file = open(os.path.join(args.output, "finish.txt"), "a", buffering=1)

    net = make_model(conf["model"]).to(device=device).load_weights(args)
    renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp, eval_batch_size=args.ray_batch_size).to(device)
    if args.coarse:
        net.mlp_fine = None
    renderer.n_coarse = max(renderer.n_coarse, 64)
    render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

    z_near, z_far = dset.z_near, dset.z_far
    source = torch.tensor(sorted(list(map(int, args.source.split()))), dtype=torch.long)

    for obj_idx, data in enumerate(data_loader):
        try:
            obj_path = data["path"][0]
            obj_name = os.path.basename(obj_path)
            obj_out_dir = os.path.join(args.output, obj_name)
            os.makedirs(obj_out_dir, exist_ok=True)

            images = data["images"][0]
            poses = data["poses"][0]
            focal = data["focal"][0]
            c = data.get("c")
            c = c[0].to(device).unsqueeze(0) if c is not None else None

            if images.shape[0] < 2:
                print(f"Skipping {obj_name} - less than 2 views", flush=True)
                continue

            src_view_mask = torch.zeros(images.shape[0], dtype=torch.bool)
            src_view_mask[source] = 1
            src_poses = poses[src_view_mask].to(device)

            focal = focal[None].to(device)
            net.encode(images[src_view_mask].to(device).unsqueeze(0), src_poses.unsqueeze(0), focal, c=c)

            print(f"Extracting mesh for {obj_name}...", flush=True)
            res = args.mesh_res
            grid = torch.linspace(-1, 1, res, device=device)
            xs, ys, zs = torch.meshgrid(grid, grid, grid, indexing='ij')
            pts = torch.stack([xs, ys, zs], -1).reshape(-1, 3)

            sigma_list = []
            for i in tqdm.trange(0, pts.size(0), 65536, desc=f"Marching cubes {obj_name}"):
                pts_chunk = pts[i:i+65536]
                viewdirs = torch.zeros((1, pts_chunk.size(0), 3), device=device)
                out = net.forward(pts_chunk.unsqueeze(0), coarse=True, viewdirs=viewdirs)
                sigma_list.append(out[0, :, 3].detach())

            sigma_vals = torch.cat(sigma_list, dim=0)
            sig = torch.relu(sigma_vals).view(res, res, res).cpu().numpy()
            verts, faces, normals, _ = skimage.measure.marching_cubes(sig, level=args.mesh_thresh)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh_path = os.path.join(obj_out_dir, f"{obj_name}_mesh.stl")
            mesh.export(mesh_path)
            print(f"Mesh saved to {mesh_path}", flush=True)
            continue

            if args.no_compare_gt:
                continue

            target_view_mask = torch.ones(images.shape[0], dtype=torch.bool)
            target_view_mask[source] = False
            novel_idxs = target_view_mask.nonzero(as_tuple=False).reshape(-1)
            target_poses = poses[target_view_mask]

            if len(novel_idxs) == 0:
                print(f"No novel views for {obj_name}, skipping evaluation.", flush=True)
                continue

            rays = util.gen_rays(
                target_poses,
                images.shape[-1], images.shape[-2],
                focal, z_near, z_far, c
            ).reshape(-1, 8).to(device)

            rays_spl = torch.split(rays, args.ray_batch_size, dim=0)

            all_rgb = []
            for rays in tqdm.tqdm(rays_spl, desc=f"Rendering eval views {obj_name}"):
                rgb, _ = render_par(rays[None])
                all_rgb.append(rgb[0].cpu())

            all_rgb = torch.clamp(torch.cat(all_rgb, dim=0).reshape(len(novel_idxs), images.shape[-2], images.shape[-1], 3), 0.0, 1.0)
            gt = ((images * 0.5 + 0.5)[target_view_mask]).permute(0, 2, 3, 1)

            psnr = np.mean([skimage.measure.compare_psnr(gt[i].numpy(), all_rgb[i].numpy(), data_range=1) for i in range(len(novel_idxs))])
            ssim = np.mean([skimage.measure.compare_ssim(gt[i].numpy(), all_rgb[i].numpy(), multichannel=True, data_range=1) for i in range(len(novel_idxs))])

            print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}", flush=True)
            finish_file.write(f"{obj_name} {psnr:.2f} {ssim:.4f} 1\n")

        except Exception as e:
            print(f"ERROR processing {obj_name}: {e}", flush=True)
            traceback.print_exc()
            continue

