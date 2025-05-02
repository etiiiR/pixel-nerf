# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import sys
import os
import math
import multiprocessing

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


def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--nviews", "-V", type=str, default="1",
        help="Number of source views (multiview); space-delimited list"
    )
    parser.add_argument(
        "--freeze_enc", action="store_true", default=False,
        help="Freeze ResNet encoder during initial warmup epochs"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=0,
        help="Number of initial epochs to keep encoder frozen before fine-tuning"
    )
    parser.add_argument(
        "--no_bbox_step", type=int, default=100000,
        help="Step to stop using bbox sampling"
    )
    parser.add_argument(
        "--fixed_test", action="store_true", default=False,
        help="Use fixed pose for test visualization"
    )
    return parser


def main():
    # parse args and config
    args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
    device = util.get_cuda(args.gpu_id[0])

    # load datasets
    train_dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir)
    print(f"dset z_near {train_dset.z_near}, z_far {train_dset.z_far}, lindisp {train_dset.lindisp}")

    # build model
    net = make_model(conf["model"]).to(device=device)

    # initial encoder freeze for warmup
    if args.freeze_enc and args.warmup_epochs > 0 and hasattr(net, 'encoder'):
        print(f"Freezing ResNet encoder for first {args.warmup_epochs} epochs")
        net.stop_encoder_grad = True
        net.encoder.eval()
        for _, p in net.encoder.named_parameters(): p.requires_grad = False
    else:
        net.stop_encoder_grad = False

    # renderer setup
    renderer = NeRFRenderer.from_conf(conf["renderer"], lindisp=train_dset.lindisp).to(device=device)
    render_par = renderer.bind_parallel(net, args.gpu_id).eval()

    # parse view selection options
    nviews = list(map(int, args.nviews.split()))

    class PixelNeRFTrainer(trainlib.Trainer):
        def __init__(self):
            super().__init__(net, train_dset, val_dset, args, conf["train"], device=device)
            self.renderer_state_path = os.path.join(self.args.checkpoints_path, self.args.name, '_renderer')

            # loss functions
            self.lambda_coarse = conf.get_float("loss.lambda_coarse")
            self.lambda_fine   = conf.get_float("loss.lambda_fine", 1.0)
            self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
            fine_conf = conf["loss.rgb_fine"] if "rgb_fine" in conf["loss"] else conf["loss.rgb"]
            self.rgb_fine_crit   = loss.get_rgb_loss(fine_conf, False)

            # resume renderer state
            if args.resume and os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(torch.load(self.renderer_state_path, map_location=device))

            # near/far bounds
            self.z_near = train_dset.z_near
            self.z_far  = train_dset.z_far
            self.use_bbox = args.no_bbox_step > 0

            # compute warmup steps if freezing
            self.warmup_epochs = args.warmup_epochs
            if args.freeze_enc and self.warmup_epochs > 0:
                steps_per_epoch = math.ceil(len(train_dset) / args.batch_size)
                self.warmup_steps = self.warmup_epochs * steps_per_epoch
                print(f"Encoder frozen for {self.warmup_steps} steps ({self.warmup_epochs} epochs)")
            else:
                self.warmup_steps = 0

        def post_batch(self, epoch, batch):
            renderer.sched_step(args.batch_size)

        def extra_save_state(self):
            torch.save(renderer.state_dict(), self.renderer_state_path)

        def calc_losses(self, data, is_train=True, global_step=0):
            if "images" not in data:
                return {}

            # unfreeze encoder after warmup
            if global_step == self.warmup_steps and self.warmup_steps > 0 and hasattr(net, 'encoder'):
                print("Warmup complete: unfreezing ResNet encoder")
                net.stop_encoder_grad = False
                net.encoder.train()
                for _, p in net.encoder.named_parameters(): p.requires_grad = True

            # unpack batch
            imgs   = data["images"].to(device)  # [SB, NV,3,H,W]
            poses  = data["poses"].to(device)   # [SB, NV,4,4]
            focals = data["focal"]              # [SB]
            cs     = data.get("c")              # [SB,2] or None
            SB, NV, _, H, W = imgs.shape

            # select source views
            k = np.random.choice(nviews)
            if k == 1:
                image_ord = torch.randint(0, NV, (SB,1), device=device)
            else:
                image_ord = torch.stack([
                    torch.from_numpy(np.random.choice(NV, k, replace=False))
                    for _ in range(SB)
                ], dim=0).to(device)

            # sample rays for each sample
            batch_rays, batch_rgb = [], []
            ray_batch = conf["train"].get_int("ray_batch_size", 128)
            for b in range(SB):
                f_b = focals[b].to(device)
                c_b = cs[b].to(device) if cs is not None else None
                cam_rays = util.gen_rays(
                    poses[b], W, H, f_b.unsqueeze(0), self.z_near, self.z_far,
                    c=c_b.unsqueeze(0) if c_b is not None else None
                )
                rays_flat = cam_rays.view(-1,8)
                rgb_flat  = (imgs[b]*0.5+0.5).permute(0,2,3,1).reshape(-1,3).to(device)
                idxs = torch.randint(0, rays_flat.size(0), (ray_batch,), device=device)
                batch_rays.append(rays_flat[idxs])
                batch_rgb.append(rgb_flat[idxs])
            all_rays = torch.stack(batch_rays)  # [SB, ray_batch,8]
            all_rgb  = torch.stack(batch_rgb)   # [SB, ray_batch,3]

            # gather source views
            src_imgs = util.batched_index_select_nd(imgs, image_ord)
            src_ps   = util.batched_index_select_nd(poses, image_ord)

            # encode features
            net.encode(
                src_imgs, src_ps, focals.to(device),
                c=cs.to(device) if cs is not None else None
            )

            # render
            RD = DotMap(render_par(all_rays, want_weights=True))
            c_rgb = RD.coarse.rgb.view(SB, -1,3)
            loss = self.lambda_coarse * self.rgb_coarse_crit(c_rgb, all_rgb)
            if RD.fine.rgb.numel() > 0:
                f_rgb = RD.fine.rgb.view(SB, -1,3)
                loss += self.lambda_fine * self.rgb_fine_crit(f_rgb, all_rgb)

            # backprop
            if is_train:
                loss.backward()
            return {"loss": loss.item()}

        def train_step(self, data, global_step):
            return self.calc_losses(data, True, global_step)

        def eval_step(self, data, global_step):
            renderer.eval()
            res = self.calc_losses(data, False, global_step)
            renderer.train()
            return res

        def vis_step(self, data, global_step, idx=None):
            if "images" not in data:
                return {}
            # select sample
            if idx is None:
                batch_idx = np.random.randint(0, data["images"].shape[0])
            else:
                batch_idx = idx
            images = data["images"][batch_idx].to(device)  # (NV,3,H,W)
            poses  = data["poses"][batch_idx].to(device)   # (NV,4,4)
            focal  = data["focal"][batch_idx:batch_idx+1]
            c      = data.get("c")
            if c is not None:
                c = c[batch_idx:batch_idx+1]
            NV, _, H, W = images.shape

            # compute full ray map
            cam_rays = util.gen_rays(poses, W, H, focal, self.z_near, self.z_far, c=c)
            images_0to1 = images * 0.5 + 0.5

            # choose source & target views
            curr_nviews = np.random.choice(nviews)
            views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
            remaining = list(set(range(NV)) - set(views_src))
            view_dest = np.random.choice(remaining)
            views_src = torch.from_numpy(views_src).to(device)

            # encode
            renderer.eval()
            src_imgs  = images_0to1[views_src].unsqueeze(0)
            src_poses = poses[views_src].unsqueeze(0)
            net.encode(src_imgs, src_poses, focal.to(device), c=(c.to(device) if c is not None else None))

            # render
            test_rays = cam_rays[view_dest].view(1, -1, 8)
            rd = DotMap(render_par(test_rays, want_weights=True))
            out = rd.fine if rd.fine.rgb.numel()>0 else rd.coarse
            rgb = out.rgb[0].view(H,W,3).cpu().numpy()

            # gt and psnr
            gt = images_0to1[view_dest].permute(1,2,0).cpu().numpy()
            psnr_val = util.psnr(torch.from_numpy(rgb), torch.from_numpy(gt))
            print(f"Vis PSNR: {psnr_val}")

            return rgb, {"psnr": psnr_val}

    # start training
    PixelNeRFTrainer().start()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
