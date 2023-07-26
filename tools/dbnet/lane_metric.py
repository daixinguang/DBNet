# -*- coding:utf-8 -*-
import sys
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
from itertools import product


def calc_x(f, t):
    return f['a_x'] + f['b_x'] * t + f['c_x'] * t * t + f['d_x'] * t * t * t
def calc_y(f, t):
    return f['a_y'] + f['b_y'] * t + f['c_y'] * t * t + f['d_y'] * t * t * t

def spline_interp(*, lane, step_t=1):
    interp_lane = []
    if len(lane) < 2:
        return lane
    interp_param = calc_params(lane)
    for f in interp_param:
        t = 0
        while t < f['h']:
            x = calc_x(f, t)
            y = calc_y(f, t)
            interp_lane.append({"x": x, "y": y})
            t += step_t
    interp_lane.append(lane[-1])
    return interp_lane


def calc_params(lane):
    params = []
    n_pt = len(lane)
    if n_pt < 2:
        return params
    if n_pt == 2:
        h0 = np.sqrt((lane[0]['x'] - lane[1]['x']) * (lane[0]['x'] - lane[1]['x']) +
                     (lane[0]['y'] - lane[1]['y']) * (lane[0]['y'] - lane[1]['y']))
        a_x = lane[0]['x']
        a_y = lane[0]['y']
        b_x = (lane[1]['x'] - a_x) / h0
        b_y = (lane[1]['y'] - a_y) / h0
        params.append({"a_x": a_x, "b_x": b_x, "c_x": 0, "d_x": 0, "a_y": a_y, "b_y": b_y, "c_y": 0, "d_y": 0, "h": h0})
        return params
    h = []
    for i in range(n_pt - 1):
        dx = lane[i]['x'] - lane[i + 1]['x']
        dy = lane[i]['y'] - lane[i + 1]['y']
        h.append(np.sqrt(dx * dx + dy * dy))
    A = []
    B = []
    C = []
    D_x = []
    D_y = []
    for i in range(n_pt - 2):
        A.append(h[i])
        B.append(2 * (h[i] + h[i + 1]))
        C.append(h[i + 1])
        dx1 = (lane[i + 1]['x'] - lane[i]['x']) / h[i]
        dx2 = (lane[i + 2]['x'] - lane[i + 1]['x']) / h[i + 1]
        tmpx = 6 * (dx2 - dx1)
        dy1 = (lane[i + 1]['y'] - lane[i]['y']) / h[i]
        dy2 = (lane[i + 2]['y'] - lane[i + 1]['y']) / h[i + 1]
        tmpy = 6 * (dy2 - dy1)
        if i == 0:
            C[i] /= B[i]
            D_x.append(tmpx / B[i])
            D_y.append(tmpy / B[i])
        else:
            base_v = B[i] - A[i] * C[i - 1]
            C[i] /= base_v
            D_x.append((tmpx - A[i] * D_x[i - 1]) / base_v)
            D_y.append((tmpy - A[i] * D_y[i - 1]) / base_v)

    Mx = np.zeros(n_pt)
    My = np.zeros(n_pt)
    Mx[n_pt - 2] = D_x[n_pt - 3]
    My[n_pt - 2] = D_y[n_pt - 3]
    for i in range(n_pt - 4, -1, -1):
        Mx[i + 1] = D_x[i] - C[i] * Mx[i + 2]
        My[i + 1] = D_y[i] - C[i] * My[i + 2]

    Mx[0] = 0
    Mx[-1] = 0
    My[0] = 0
    My[-1] = 0

    for i in range(n_pt - 1):
        a_x = lane[i]['x']
        b_x = (lane[i + 1]['x'] - lane[i]['x']) / h[i] - (2 * h[i] * Mx[i] + h[i] * Mx[i + 1]) / 6
        c_x = Mx[i] / 2
        d_x = (Mx[i + 1] - Mx[i]) / (6 * h[i])

        a_y = lane[i]['y']
        b_y = (lane[i + 1]['y'] - lane[i]['y']) / h[i] - (2 * h[i] * My[i] + h[i] * My[i + 1]) / 6
        c_y = My[i] / 2
        d_y = (My[i + 1] - My[i]) / (6 * h[i])

        params.append(
            {"a_x": a_x, "b_x": b_x, "c_x": c_x, "d_x": d_x, "a_y": a_y, "b_y": b_y, "c_y": c_y, "d_y": d_y, "h": h[i]})

    return params

def resize_lane(lane, x_ratio, y_ratio):
    return [{"x": float(p['x']) / x_ratio, "y": float(p['y']) / y_ratio} for p in lane]


def calc_iou(lane1, lane2, hyperp):
    new_height = hyperp['eval_height']
    new_width = hyperp['eval_width']
    lane_width = hyperp['lane_width']

    im1 = np.zeros((new_height, new_width), np.uint8)
    im2 = np.zeros((new_height, new_width), np.uint8)
    interp_lane1 = spline_interp(lane=lane1, step_t=1)
    interp_lane2 = spline_interp(lane=lane2, step_t=1)
    for i in range(0, len(interp_lane1) - 1):
        cv2.line(im1, (int(interp_lane1[i]['x']), int(interp_lane1[i]['y'])),
                 (int(interp_lane1[i + 1]['x']), int(interp_lane1[i + 1]['y'])), 255, lane_width)
    for i in range(0, len(interp_lane2) - 1):
        cv2.line(im2, (int(interp_lane2[i]['x']), int(interp_lane2[i]['y'])),
                 (int(interp_lane2[i + 1]['x']), int(interp_lane2[i + 1]['y'])), 255, lane_width)
    union_im = cv2.bitwise_or(im1, im2)
    union_sum = union_im.sum()
    intersection_sum = im1.sum() + im2.sum() - union_sum
    if union_sum == 0:
        return 0
    else:
        return intersection_sum / float(union_sum)


def evaluate_core(*, gt_lanes, pr_lanes, gt_wh, pr_wh, hyperp):
    gt_num = len(gt_lanes)
    pr_num = len(pr_lanes)
    hit_num = 0
    pr_list = [False for i in range(pr_num)]
    gt_list = [False for i in range(gt_num)]

    if gt_num > 0 and pr_num > 0:
        iou_thresh = hyperp['iou_thresh']
        new_height = hyperp['eval_height']
        new_width = hyperp['eval_width']

        gt_y_ratio = np.true_divide(gt_wh['height'], new_height)
        gt_x_ratio = np.true_divide(gt_wh['width'], new_width)
        pr_y_ratio = np.true_divide(pr_wh['height'], new_height)
        pr_x_ratio = np.true_divide(pr_wh['width'], new_width)
        gt_lanes = list(map(lambda lane: resize_lane(lane, gt_x_ratio, gt_y_ratio), gt_lanes))
        pr_lanes = list(map(lambda lane: resize_lane(lane, pr_x_ratio, pr_y_ratio), pr_lanes))

        sorted_gt_lanes = gt_lanes
        sorted_pr_lanes = pr_lanes
        iou_mat = np.zeros((gt_num, pr_num))

        for (index_gt, gt_lane), (index_pr, pr_lane) in product(enumerate(sorted_gt_lanes), enumerate(sorted_pr_lanes)):
            iou_mat[index_gt][index_pr] = calc_iou(gt_lane, pr_lane, hyperp)

        cost_matrix = 1 - np.array(iou_mat)
        match_index_list = linear_sum_assignment(cost_matrix)

        for gt_index, pr_index in zip(*match_index_list):
            iou_val = iou_mat[gt_index][pr_index]
            if iou_val > iou_thresh:
                hit_num += 1
                pr_list[pr_index] = True
                gt_list[gt_index] = True
    return dict(gt_num=gt_num, pr_num=pr_num, hit_num=hit_num, pr_list=pr_list, gt_list=gt_list)


class LaneMetricCore(object):

    def __init__(self, *, eval_width, eval_height, iou_thresh, lane_width, prob_thresh=None):
        self.eval_params = dict(
            eval_width=eval_width,
            eval_height=eval_height,
            iou_thresh=iou_thresh,
            lane_width=lane_width,
        )
        self.prob_thresh = prob_thresh
        self.result_record = []
        self.results = []

    def __call__(self, gt_result, pr_result, *args, **kwargs):
        prob_thresh = self.prob_thresh
        predict_spec = pr_result
        target_spec = gt_result

        gt_wh = target_spec['Shape']
        pr_wh = predict_spec['Shape']
        gt_lanes = []
        for line_spec in target_spec['Lines']:
            if len(line_spec) > 0:
                gt_lanes.append(line_spec)

        pr_lanes = []
        for line_spec in predict_spec['Lines']:
            if 'score' in line_spec:
                if float(line_spec['score']) > prob_thresh:
                    line_spec = line_spec['points']
                else:
                    line_spec = []
            if len(line_spec) > 0:
                pr_lanes.append(line_spec)

        result = evaluate_core(gt_lanes=gt_lanes, pr_lanes=pr_lanes, gt_wh=gt_wh, pr_wh=pr_wh, hyperp=self.eval_params)

        self.result_record.append(result)

    def reset(self):
        self.result_record = []

    def summary(self):
        hit_num = sum(result['hit_num'] for result in self.result_record)
        pr_num = sum(result['pr_num'] for result in self.result_record)
        gt_num = sum(result['gt_num'] for result in self.result_record)
        precision = hit_num / (pr_num + sys.float_info.epsilon)
        recall = hit_num / (gt_num + sys.float_info.epsilon)
        f1_measure = 2 * precision * recall / (precision + recall + sys.float_info.epsilon)
        return dict(f1_measure=f1_measure, precision=precision, recall=recall, pr_num=pr_num, gt_num=gt_num)
