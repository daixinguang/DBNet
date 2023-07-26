import os
import math
import torch
from mmdet.core import build_assigner
from .single_stage import SingleStageDetector
from ..builder import DETECTORS, build_loss

import timeit

@DETECTORS.register_module
class DBNet(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss='LaneLossAggress',
                 loss_weights={},
                 output_scale=4,
                 num_classes=1,
                 point_scale=True,
                 sample_gt_points=[11, 11, 11, 11],
                 assigner_cfg=dict(type='LaneAssigner'),
                 use_smooth=False):
        super(DBNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=head,
            train_cfg=None,
            test_cfg=None,
            pretrained=pretrained)
        self.sample_gt_points = sample_gt_points
        self.num_classes = num_classes
        self.head = head
        self.use_smooth = use_smooth
        self.assigner_cfg = assigner_cfg
        self.loss_weights = loss_weights
        self.point_scale = point_scale
        if test_cfg is not None and 'out_scale' in test_cfg.keys():
            self.output_scale = test_cfg['out_scale']
        else:
            self.output_scale = output_scale
        self.loss = build_loss(loss)
        if self.assigner_cfg:
            self.assigner = build_assigner(self.assigner_cfg)

    def forward(self, img, img_metas=None, return_loss=True, **kwargs):
        if img_metas is None:
            return self.test_inference(img, **kwargs)
        elif return_loss:
            return self.forward_train(img, img_metas, **kwargs)  
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, img_metas, **kwargs):
        
        output = self.backbone(img.type(torch.cuda.FloatTensor))  
        output = self.neck(output)  
        if self.head:
            [cpts_hm, kpts_hm, pts_cfe, int_offset,pts_cfe_end,pts_gas_1,pts_gas_2,pts_gas_3] = self.bbox_head.forward_train(output['features'],
                                                                                      output.get("aux_feat", None))  
        cpts_hm = torch.clamp(torch.sigmoid(cpts_hm), min=1e-4, max=1 - 1e-4)
        kpts_hm = torch.clamp(torch.sigmoid(kpts_hm), min=1e-4, max=1 - 1e-4)

        loss_items = [
            {"type": "focalloss", "gt": kwargs['gt_cpts_hm'], "pred": cpts_hm, "weight": self.loss_weights["center"]},
            {"type": "focalloss", "gt": kwargs['gt_kpts_hm'], "pred": kpts_hm, "weight": self.loss_weights["point"]}]
        if not self.use_smooth:
            loss_items.append({"type": "regl1kploss", "gt": kwargs['int_offset'], "pred": int_offset,
                               "mask": kwargs['offset_mask'], "weight": self.loss_weights["error"]})
            loss_items.append({"type": "regl1kploss", "gt": kwargs['pts_cfe'], "pred": pts_cfe,
                               "mask": kwargs['offset_mask_weight'], "weight": self.loss_weights["cfe"]})
            loss_items.append({"type": "regl1kploss", "gt": kwargs['pts_cfe_end'], "pred": pts_cfe_end,
                              "mask": kwargs['offset_mask_weight1'], "weight": self.loss_weights["cfe_end"]}) 
            loss_items.append({"type": "regl1kploss", "gt": kwargs['pts_gas_1'], "pred": pts_gas_1,
                              "mask": kwargs['offset_mask_weight2'], "weight": self.loss_weights["gas_1"]}) 
            loss_items.append({"type": "regl1kploss", "gt": kwargs['pts_gas_2'], "pred": pts_gas_2,
                              "mask": kwargs['offset_mask_weight3'], "weight": self.loss_weights["gas_2"]}) 
            loss_items.append({"type": "regl1kploss", "gt": kwargs['pts_gas_3'], "pred": pts_gas_3,
                              "mask": kwargs['offset_mask_weight4'], "weight": self.loss_weights["gas_3"]}) 
        else:
            loss_items.append({"type": "smoothl1loss", "gt": kwargs['int_offset'], "pred": int_offset,
                               "mask": kwargs['offset_mask'], "weight": self.loss_weights["error"]})
            loss_items.append({"type": "smoothl1loss", "gt": kwargs['pts_cfe'], "pred": pts_cfe,
                               "mask": kwargs['offset_mask_weight'], "weight": self.loss_weights["cfe"]})
            loss_items.append({"type": "smoothl1loss", "gt": kwargs['pts_cfe_end'], "pred": pts_cfe_end,
                              "mask": kwargs['offset_mask_weight1'], "weight": self.loss_weights["cfe_end"]}) 
            loss_items.append({"type": "regl1kploss", "gt": kwargs['pts_gas_1'], "pred": pts_gas_1,
                              "mask": kwargs['offset_mask_weight2'], "weight": self.loss_weights["gas_1"]}) 
            loss_items.append({"type": "regl1kploss", "gt": kwargs['pts_gas_2'], "pred": pts_gas_2,
                              "mask": kwargs['offset_mask_weight3'], "weight": self.loss_weights["gas_2"]}) 
            loss_items.append({"type": "regl1kploss", "gt": kwargs['pts_gas_3'], "pred": pts_gas_3,
                              "mask": kwargs['offset_mask_weight4'], "weight": self.loss_weights["gas_3"]}) 

        if "deform_points" in output.keys() and self.loss_weights["aux"] != 0:
            for i, points in enumerate(output['deform_points']):
                if points is None:
                    continue
                gt_points = kwargs[f'lane_points_l{i}']
                gt_matched_points, pred_matched_points = self.assigner.assign(points, gt_points,
                                                                              sample_gt_points=self.sample_gt_points[i])
                if self.point_scale:
                    loss_item = {"type": "smoothl1loss", "gt": gt_matched_points / (2 ** (3 - i)),
                                 "pred": pred_matched_points / (2 ** (3 - i)), "weight": self.loss_weights["aux"]}
                else:
                    loss_item = {"type": "smoothl1loss", "gt": gt_matched_points, "pred": pred_matched_points,
                                 "weight": self.loss_weights["aux"]}
                loss_items.append(loss_item)

        losses = self.loss(loss_items)
        return losses

    def test_inference(self, img, hack_seeds=None, **kwargs):
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)  
        if self.head:
            [cpts_hm, kpts_hm, pts_cfe, int_offset,pts_cfe_end,gas_1,gas_2,gas_3] = self.bbox_head.forward_train(output['features'],
                                                                                      output.get("aux_feat", None))
            seeds, hm = self.bbox_head.forward_test(output['features'], output.get("aux_feat", None), hack_seeds,
                                                    kwargs['thr'], kwargs['kpt_thr'],
                                                    kwargs['cpt_thr'])
        output['cpts_hm'] = cpts_hm
        output['kpts_hm'] = kpts_hm
        output['pts_cfe'] = pts_cfe
        output['int_offset'] = int_offset
        output['pts_cfe_end'] = pts_cfe_end  
        output['gas_1']=gas_1 
        output['gas_2']=gas_2 
        output['gas_3']=gas_3 
        output['deform_points'] = output['deform_points']
        output['seeds'] = seeds
        output['hm'] = hm
        return output

    def forward_test(self, img, img_metas,
                     hack_seeds=None,
                     **kwargs):

        print("time.clock ++++++++++")
        start = timeit.default_timer()

        """Test without augmentation."""
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        output = self.neck(output)
        if self.head:
            seeds, hm = self.bbox_head.forward_test(output['features'], output.get("aux_feat", None), hack_seeds,
                                                    kwargs['thr'], kwargs['kpt_thr'],
                                                    kwargs['cpt_thr'])

        end = timeit.default_timer()
        print('Running time: %s Seconds' % (end - start))

        return [seeds, hm]

    def forward_dummy(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        x = self.bbox_head.forward_train(x['features'], x.get("aux_feat", None))
        return x
