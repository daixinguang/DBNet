import os
os.chdir('..')

import argparse
import glob
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from tusimple.evaluate.lane import LaneEval
from tusimple.test_dataset import *

def generate_grid(points_map):
    b, p, h, w = points_map.shape
    y = torch.arange(h)[:, None, None].repeat(1, w, 1)
    x = torch.arange(w)[None, :, None].repeat(h, 1, 1)
    # b, h, w, p
    coods = torch.cat([y, x], dim=-1)[None, :, :, None, :].repeat(b, 1, 1, p//2, 1).float()
    # b, p, h, w
    grid = coods.reshape(b, h, w, p).permute(0, 3, 1, 2).to(points_map.device)
    return grid

parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
parser.add_argument('config_name', default="pointMerge_id3")
args = parser.parse_args()

config = f"configs/magiclanenet/tusimple/{args.config_name}.py"
cfg = Config.fromfile(config)
cfg.data.samples_per_gpu = 1
gt_path = '/mnt/lustre/wangjinsheng/project/lane-detection/conditional-lane-detection/datasets/tusimple/test_label.json'
pred_path = sorted(glob.glob(f'tools/output/tusimple/{args.config_name}_2021*/result/test.json'))[-1]
print(pred_path)
criterias, _ = LaneEval.bench_one_submit(pred_path, gt_path, return_each=True)
bad_p = torch.arange(len(criterias['p']))[torch.tensor(criterias['p']) > 0]
bad_n = torch.arange(len(criterias['n']))[torch.tensor(criterias['n']) > 0]
dataset = build_dataset(cfg.data.train)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False)
checkpoint = sorted(glob.glob(f'tools/output/tusimple/{args.config_name}_2021*/latest.pth'))[-1]
print(checkpoint)
model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
model.load_state_dict(torch.load(checkpoint, map_location='cpu')['state_dict'], strict=True)
model.eval()
model = MMDataParallel(model.cuda(), device_ids=[0])

for i, data in enumerate(data_loader):
    label = 'None'

    image   = data['img'].data[0].cuda()
    thr     = 0.3
    kpt_thr = 0.3
    cpt_thr = 0.3
    b = image.shape[0]
    results = model.module.test_inference(image, thr=thr, kpt_thr=kpt_thr, cpt_thr=cpt_thr)

    if not os.path.exists(f"debug/{args.config_name}/"):
        os.makedirs(f"debug/{args.config_name}/")
    cpts_hm    = results['cpts_hm'][0, 0].detach().cpu().sigmoid().numpy()
    gt_cpts_hm = data['gt_cpts_hm'].data[0][0, 0].detach().cpu().numpy()
    kpts_hm    = results['kpts_hm'][0, 0].detach().cpu().sigmoid().numpy()
    gt_kpts_hm = data['gt_kpts_hm'].data[0][0, 0].detach().cpu().numpy()
    image      = image[0, 0].detach().cpu().numpy()
    if not os.path.exists(f"debug/{args.config_name}/%04d/"%i):
        os.makedirs(f"debug/{args.config_name}/%04d/"%i)
    plt.imsave(f"debug/{args.config_name}/%04d/{label}result_cpts_hm.png"%i, cpts_hm)
    plt.imsave(f"debug/{args.config_name}/%04d/{label}gt_cpts_hm.png"%i,     gt_cpts_hm)
    plt.imsave(f"debug/{args.config_name}/%04d/{label}result_kpts_hm.png"%i, kpts_hm)
    plt.imsave(f"debug/{args.config_name}/%04d/{label}gt_kpts_hm.png"%i,     gt_kpts_hm)
    plt.imsave(f"debug/{args.config_name}/%04d/{label}image.png"%i,          image)

    layer = 0
    dp_num = cfg.dcn_point_num
    if not os.path.exists(f"debug/{args.config_name}/%04d/dcn_points_l{layer}/"%i):
        os.makedirs(f"debug/{args.config_name}/%04d/dcn_points_l{layer}/"%i)
    deform_point = results['deform_points'][layer]
    exist_mask = (data['gt_kpts_hm'].data[0][0, 0] > 0.5).long()
    grid = generate_grid(deform_point)
    pos_abs = deform_point+grid
    grid_filter = grid.contiguous()[:, :, exist_mask.bool()].reshape(b, dp_num[layer], 2, -1)
    pos_abs_filter = pos_abs.contiguous()[:, :, exist_mask.bool()].reshape(b, dp_num[layer], 2, -1)
    print(deform_point.shape)
    plt.figure(figsize=(100, 40))
    gaps = [16, 8, 4, 2]
    gap = gaps[layer]
    print(pos_abs_filter.shape)
    for p_id in range(0, pos_abs_filter.shape[-1], gap):
        c_p = grid_filter[0, 0, :, p_id].long()
        mask = exist_mask.numpy().copy()
        ym, xm = exist_mask.shape
        for p in pos_abs_filter[0, :, :, p_id].long():
            y = torch.clamp(p[0], 0, ym-1).cpu()
            x = torch.clamp(p[1], 0, xm-1).cpu()
            mask[y, x] = 2
        mask[c_p[0], c_p[1]] = 4
        plt.subplot(10, 4, p_id//gap+1)
        mask = np.uint8(mask/4*255)
        heat_img = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
        heat_img = cv2.resize(heat_img, (400, 160), interpolation=cv2.INTER_NEAREST)
        plt.imsave(f"debug/{args.config_name}/%04d/dcn_points_l{layer}/%03d.png"%(i,p_id), heat_img)