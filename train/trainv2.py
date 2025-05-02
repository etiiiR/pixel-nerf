# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import sys
import os
import multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import warnings
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util
import numpy as np
import torch
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from dotmap import DotMap
warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_camera_center(M: np.ndarray) -> np.ndarray:
    R = M[:3, :3]; t = M[:3, 3]
    return -R.T @ t


def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Use a fixed view_dest and views_src in vis_step (for reproducible visuals)",
    )
    return parser


# parse global args
args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
device = util.get_cuda(args.gpu_id[0])

dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)
print(f"dset z_near {dset.z_near}, z_far {dset.z_far}, lindisp {dset.lindisp}")

net = make_model(conf["model"]).to(device=device)
net.stop_encoder_grad = args.freeze_enc
if args.freeze_enc:
    print("Encoder frozen")
    net.encoder.eval()

renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp).to(device=device)
render_par = renderer.bind_parallel(net, args.gpu_id).eval()
nviews = list(map(int, args.nviews.split()))


class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, dset, val_dset, args, conf["train"], device=device)
        self.renderer_state_path = f"{self.args.checkpoints_path}/{self.args.name}/_renderer"
        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print(f"lambda coarse {self.lambda_coarse} and fine {self.lambda_fine}")
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        if args.resume and os.path.exists(self.renderer_state_path):
            renderer.load_state_dict(
                torch.load(self.renderer_state_path, map_location=device)
            )

        self.z_near = dset.z_near
        self.z_far = dset.z_far
        self.use_bbox = args.no_bbox_step > 0
        # Merke Dir fixed_test für die vis_step
        self.fixed_test = args.fixed_test

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)
        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device)    # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")                  # (SB, NV, 4)
        all_focals = data["focal"]                     # (SB)
        all_c = data.get("c")                          # (SB)

        if self.use_bbox and global_step >= args.no_bbox_step:
            self.use_bbox = False
            print(">>> Stopped using bbox sampling @ iter", global_step)
        if not is_train or not self.use_bbox:
            all_bboxes = None

        all_rgb_gt = []
        all_rays = []

        # zufällige Anzahl von Quell-Views pro Objekt
        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        image_ord = (
            torch.randint(0, NV, (SB, 1)) if curr_nviews == 1 
            else torch.empty((SB, curr_nviews), dtype=torch.long)
        )

        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]   # (NV,3,H,W)
            poses  = all_poses[obj_idx]    # (NV,4,4)
            focal  = all_focals[obj_idx]
            c      = all_c[obj_idx] if all_c is not None else None

            if curr_nviews > 1 and all_bboxes is None:
                image_ord[obj_idx] = torch.from_numpy(
                    np.random.choice(NV, curr_nviews, replace=False)
                )

            images_0to1 = images * 0.5 + 0.5
            cam_rays = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c)
            rgb_gt_all = images_0to1.permute(0,2,3,1).reshape(-1,3)

            if all_bboxes is not None:
                pix = util.bbox_sample(bboxes, args.ray_batch_size)
                pix_inds = pix[...,0]*H*W + pix[...,1]*W + pix[...,2]
            else:
                pix_inds = torch.randint(0, NV*H*W, (args.ray_batch_size,))

            rgb_gt = rgb_gt_all[pix_inds]
            rays   = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(device=device)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)
        all_rays   = torch.stack(all_rays)

        image_ord = image_ord.to(device)
        src_images = util.batched_index_select_nd(all_images, image_ord)
        src_poses  = util.batched_index_select_nd(all_poses,  image_ord)
        all_bboxes = all_poses = all_images = None

        net.encode(
            src_images,
            src_poses,
            all_focals.to(device=device),
            c = all_c.to(device=device) if all_c is not None else None,
        )

        render_dict = DotMap(render_par(all_rays, want_weights=True))
        coarse = render_dict.coarse
        fine   = render_dict.fine
        using_fine = len(fine) > 0

        loss_dict = {}
        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
        if using_fine:
            fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine

        if is_train:
            rgb_loss.backward()
        loss_dict["t"] = rgb_loss.item()
        return loss_dict

    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
            if "images" not in data:
                return {}

            # 1) Batch‑Index wählen
            batch_idx = idx if idx is not None else np.random.randint(0, data["images"].shape[0])

            # 2) Eingabe extrahieren
            images = data["images"][batch_idx].to(device=device)    # (NV,3,H,W)
            poses  = data["poses"][batch_idx].to(device=device)     # (NV,4,4)
            focal  = data["focal"][batch_idx:batch_idx+1]           # (1)
            c      = data.get("c")
            if c is not None:
                c = c[batch_idx:batch_idx+1]

            NV, _, H, W = images.shape

            # 3) Strahlen generieren und GT umnormieren
            cam_rays    = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c)  # (NV,H,W,8)
            images_0to1 = images * 0.5 + 0.5                                               # [0,1]

            # 4) Quell‑ und Ziel‑Views auswählen
            curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]
            if self.fixed_test:
                views_src = torch.arange(curr_nviews)
                view_dest = curr_nviews if curr_nviews < NV else 0
            else:
                vs = np.sort(np.random.choice(NV, curr_nviews, replace=False))
                views_src = torch.from_numpy(vs)
                view_dest = np.random.randint(0, NV - curr_nviews)
                for v in vs:
                    view_dest += (view_dest >= v)

            # 5) Netzwerk-Encoding & Rendern
            renderer.eval()
            net.encode(
                images[views_src].unsqueeze(0),
                poses[views_src].unsqueeze(0),
                focal.to(device=device),
                c=c.to(device=device) if c is not None else None,
            )
            test_rays   = cam_rays[view_dest].reshape(1, H*W, -1)
            render_dict = DotMap(render_par(test_rays, want_weights=True))
            coarse = render_dict.coarse
            fine   = render_dict.fine
            using_fine = len(fine) > 0

            # 6) Extrahiere RGBA, Depth, Alpha
            rgb_coarse_np   = coarse.rgb[0].cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth[0].cpu().numpy().reshape(H, W)
            alpha_coarse_np = coarse.weights[0].sum(dim=-1).cpu().numpy().reshape(H, W)

            if using_fine:
                rgb_fine_np   = fine.rgb[0].cpu().numpy().reshape(H, W, 3)
                depth_fine_np = fine.depth[0].cpu().numpy().reshape(H, W)
                alpha_fine_np = fine.weights[0].sum(dim=1).cpu().numpy().reshape(H, W)

            # 7) Ground‑Truth View
            source_views = images_0to1[views_src].permute(0,2,3,1).cpu().numpy().reshape(-1, H, W, 3)
            gt           = images_0to1[view_dest].permute(1,2,0).cpu().numpy().reshape(H, W, 3)

            # 8) PSNR
            rgb_psnr = rgb_fine_np if using_fine else rgb_coarse_np
            psnr = util.psnr(rgb_psnr, gt)
            print("psnr", psnr)

            # 9) 2D‑Viz zusammenbauen (optional)
            alpha_cmap = util.cmap(alpha_coarse_np) / 255
            depth_cmap = util.cmap(depth_coarse_np) / 255
            vis_coarse = np.hstack([*source_views, gt, depth_cmap, rgb_coarse_np, alpha_cmap])
            vis = vis_coarse
            if using_fine:
                alpha_fc = util.cmap(alpha_fine_np) / 255
                depth_fc = util.cmap(depth_fine_np) / 255
                vis_fine = np.hstack([*source_views, gt, depth_fc, rgb_fine_np, alpha_fc])
                vis = np.vstack([vis_coarse, vis_fine])

            # ──────────────────────────────────────────────
            # 10) Near/Far‑Frustum als 3D‑Plot darstellen
            M = poses[view_dest].cpu().numpy()           # world->cam
            R, t = M[:3,:3], M[:3,3]
            C = - R.T @ t                                # Kamera‑Zentrum in Welt
            cam2world = np.linalg.inv(M)

            # Bild‑Ecken in Pixelkoords
            # … inner­halb der vis_step(), nachdem Du cam2world und (W,H,focal) hast …

            # korrekte Mittelwerte
            cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
            fx, fy = focal.item(), focal.item()


            uv = np.array([[0,0], [W-1,0], [W-1,H-1], [0,H-1]], float)
            near_pts, far_pts = [], []
            for z_val, acc in [(self.z_near, near_pts), (self.z_far, far_pts)]:
                for (u, v) in uv:
                    x_cam = (u - cx) / fx * z_val
                    y_cam = (cy - v) / fy * z_val    # Y‑Invert!
                    p_cam = np.array([x_cam, y_cam, z_val, 1.])
                    p_wld = cam2world @ p_cam
                    acc.append(p_wld[:3])
            near_pts = np.stack(near_pts)
            far_pts  = np.stack(far_pts)

            # und schließlich plotten
            fig = plt.figure(figsize=(6,6))
            ax  = fig.add_subplot(111, projection='3d')
            ax.scatter(*C,            c='blue',  s=60, label='Camera Center')
            ax.scatter(near_pts[:,0], near_pts[:,1], near_pts[:,2],
                    c='red',   s=30, label='Near Plane')
            ax.scatter(far_pts[:,0],  far_pts[:,1],  far_pts[:,2],
                    c='green', s=30, label='Far Plane')
            ax.set_title(f'View {view_dest} – Near/Far Frustum')
            ax.legend()
            plt.tight_layout()
            #plt.show()

            # ──────────────────────────────────────────────

            renderer.train()
            return vis, {"psnr": psnr}


def main():
    multiprocessing.freeze_support()
    trainer = PixelNeRFTrainer()
    trainer.start()


if __name__ == "__main__":
    main()
