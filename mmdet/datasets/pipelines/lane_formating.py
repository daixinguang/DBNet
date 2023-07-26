import random
import math
import copy
from functools import cmp_to_key

import cv2
import PIL.Image
import PIL.ImageDraw
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from shapely.geometry import Polygon, Point, LineString, MultiLineString

from ..builder import PIPELINES
from .formating import Collect, to_tensor
import scipy.interpolate as spi
from .bspline import BS_curve, cal_matrix_all
from .b_spline import scipy_bspline


def gauss_conv(mask, kernel):  
    h, w = mask.shape
    kernel_size = kernel.shape[1]
    kernel_pad = (kernel_size - 1) // 2
    mask = np.pad(mask, ((0, 0), (kernel_pad, kernel_pad)), mode='constant')
    mask_weights = mask[None, :, :] * kernel[0][:, None, None]
    mask_max = []
    for i in range(kernel_size):
        mask_max.append(mask_weights[i:i + 1, :, i:i + w])
    mask_max = np.concatenate(mask_max, axis=0).max(axis=0)
    return mask_max


def GaussKernel1D(kernel_size=9, sigma=2):
    kernel_pad = kernel_size // 2
    kernel = np.arange(-kernel_pad, kernel_pad + 1)
    return np.exp(-kernel ** 2 / (2 * sigma ** 2)).reshape(1, -1)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def cal_dis(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_line_intersection(x, y, line):
    def in_line_range(val, start, end):
        s = min(start, end)
        e = max(start, end)
        if s <= val <= e and s != e:
            return True
        else:
            return False

    def choose_min_reg(val, ref):
        min_val = 1e5
        index = -1
        if len(val) == 0:
            return None
        else:
            for i, v in enumerate(val):
                if abs(v - ref) < min_val:
                    min_val = abs(v - ref)
                    index = i
        return val[index]

    reg_y = []
    reg_x = []

    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(x, point_start[0], point_end[0]):
            k = (point_end[1] - point_start[1]) / (point_end[0] - point_start[0])  
            reg_y.append(k * (x - point_start[0]) + point_start[1])
    reg_y = choose_min_reg(reg_y, y)

    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(y, point_start[1], point_end[1]):
            k = (point_end[0] - point_start[0]) / (point_end[1] - point_start[1])
            reg_x.append(k * (y - point_start[1]) + point_start[0])
    reg_x = choose_min_reg(reg_x, x)
    return reg_x, reg_y


def convert_list(p, downscale=None):
    xy = list()
    if downscale is None:
        for i in range(len(p) // 2):
            xy.append((p[2 * i], p[2 * i + 1]))
    else:
        for i in range(len(p) // 2):
            xy.append((p[2 * i] / downscale, p[2 * i + 1] / downscale))

    
    return xy


def ploy_fitting_cube_extend(line, h, w, sample_num=100):
    
    line_coords = np.array(line).reshape((-1, 2))
    line_coords = np.array(sorted(line_coords, key=lambda x: x[1]))
    line_coords = line_coords[line_coords[:, 0] > 0, :]
    line_coords = line_coords[line_coords[:, 0] < w, :]  
    
    if line_coords.shape[0] < 2:
        return None
    line_coords_extend = extend_line(line_coords, dis=25)
    

    X = line_coords_extend[:, 1]
    Y = line_coords_extend[:, 0]
    if len(X) < 2:
        return None

    new_x = np.linspace(max(X[0], 0), min(X[-1], h), sample_num)

    if len(X) > 3:
        ipo3 = spi.splrep(X, Y, k=3)
        iy3 = spi.splev(new_x, ipo3)
    else:
        ipo3 = spi.splrep(X, Y, k=1)
        iy3 = spi.splev(new_x, ipo3)
    return np.concatenate([iy3[:, None], new_x[:, None]], axis=1)


def ploy_fitting_cube(line, h, w, sample_num=100):
    
    line_coords = np.array(line).reshape((-1, 2))
    line_coords = np.array(sorted(line_coords, key=lambda x: x[1]))
    line_coords = line_coords[line_coords[:, 0] > 0, :]
    line_coords = line_coords[line_coords[:, 0] < w, :]

    X = line_coords[:, 1]
    # print('line coords X : {}'.format(X.shape))
    Y = line_coords[:, 0]
    # print('line coords Y : {}'.format(Y.shape))
    if len(X) < 2:
        return None
    new_x = np.linspace(max(X[0], 0), min(X[-1], h), sample_num)

    if len(X) > 3:
        ipo3 = spi.splrep(X, Y, k=3)
        iy3 = spi.splev(new_x, ipo3)
    else:
        ipo3 = spi.splrep(X, Y, k=1)
        iy3 = spi.splev(new_x, ipo3)
    return np.concatenate([iy3[:, None], new_x[:, None]], axis=1)


def ploy_fitting(line, downscale=1):
    line_coords = np.array(line).reshape((-1, 2))  
    line_x = []
    line_y = []
    xy = list()
    for coord in line_coords:
        line_x.append(coord[0])
        line_y.append(coord[1])
    ploy_fit = np.polyfit(line_x, line_y, 3)
    ploy_func = np.poly1d(ploy_fit)
    
    ploy_x = np.linspace(line_x[0], line_x[-1], int(np.abs(line_x[-1] - line_x[0]) / downscale))
    ploy_y = ploy_func(ploy_x)
    for x, y in zip(ploy_x, ploy_y):
        xy.append((x, y))
    # print('shape: {} length : {}'.format(line_coords.shape[0], len(xy)))
    
    return xy


def draw_label(mask,
               polygon_in,
               val,
               shape_type='polygon',
               width=3,
               convert=False):
    polygon = copy.deepcopy(polygon_in)
    mask = PIL.Image.fromarray(mask)
    xy = []
    if convert:
        for i in range(len(polygon) // 2):
            xy.append((polygon[2 * i], polygon[2 * i + 1]))
    else:
        for i in range(len(polygon)):
            xy.append((polygon[i][0], polygon[i][1]))

    if shape_type == 'polygon':
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=val, fill=val)
    else:
        PIL.ImageDraw.Draw(mask).line(xy=xy, fill=val, width=width)
    mask = np.array(mask, dtype=np.uint8)
    return mask


def clamp_line(line, box, min_length=0):
    left, top, right, bottom = box
    loss_box = Polygon([[left, top], [right, top], [right, bottom],
                        [left, bottom]])
    line_coords = np.array(line).reshape((-1, 2))  
    if line_coords.shape[0] < 2:
        return None
    try:
        line_string = LineString(line_coords)
        I = line_string.intersection(loss_box)  
        if I.is_empty:
            return None
        if I.length < min_length:
            return None
        if isinstance(I, LineString):
            pts = list(I.coords)
            
            
            return pts
        elif isinstance(I, MultiLineString):
            pts = []
            Istrings = list(I)
            for Istring in Istrings:
                pts += list(Istring.coords)
            return pts
    except:
        return None


def clip_line(pts, h, w):
    pts_x = np.clip(pts[:, 0], 0, w - 1)[:, None]
    pts_y = np.clip(pts[:, 1], 0, h - 1)[:, None]
    return np.concatenate([pts_x, pts_y], axis=-1)


def select_mask_points(ct, r, shape, max_sample=5):
    def in_range(pt, w, h):
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            return True
        else:
            return False

    h, w = shape[:2]
    valid_points = []
    r = max(int(r // 2), 1)
    start_x, end_x = ct[0] - r, ct[0] + r
    start_y, end_y = ct[1] - r, ct[1] + r
    for x in range(start_x, end_x + 1):
        for y in range(start_y, end_y + 1):
            if x == ct[0] and y == ct[1]:
                continue
            if in_range((x, y), w, h) and cal_dis((x, y), ct) <= r + 0.1:
                valid_points.append([x, y])
    if len(valid_points) > max_sample - 1:
        valid_points = random.sample(valid_points, max_sample - 1)
    valid_points.append([ct[0], ct[1]])
    return valid_points


def extend_line(line, dis=10):
    extended = copy.deepcopy(line)
    start = line[-2]
    end = line[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    norm = math.sqrt(dx ** 2 + dy ** 2)
    dx = dx / norm  
    dy = dy / norm  
    extend_point = np.array((end[0] + dx * dis, end[1] + dy * dis)).reshape(1, 2)
    extended = np.append(extended, extend_point, axis=0)
    return extended


def sort_line_func(a, b):
    def get_line_intersection(y, line):

        def in_line_range(val, start, end):
            s = min(start, end)
            e = max(start, end)
            if s == e and val == s:
                return 1
            elif s <= val <= e and s != e:
                return 2
            else:
                return 0

        reg_x = []
        
        for i in range(len(line) - 1):
            point_start, point_end = line[i], line[i + 1]
            flag = in_line_range(y, point_start[1], point_end[1])
            if flag == 2:
                k = (point_end[0] - point_start[0]) / (
                        point_end[1] - point_start[1])
                reg_x.append(k * (y - point_start[1]) + point_start[0])
            elif flag == 1:
                reg_x.append((point_start[0] + point_end[0]) / 2)
        reg_x = min(reg_x)

        return reg_x

    line1 = np.array(copy.deepcopy(a))
    line2 = np.array(copy.deepcopy(b))
    line1_ymin = min(line1[:, 1])
    line1_ymax = max(line1[:, 1])
    line2_ymin = min(line2[:, 1])
    line2_ymax = max(line2[:, 1])
    if line1_ymax <= line2_ymin or line2_ymax <= line1_ymin:
        y_ref1 = (line1_ymin + line1_ymax) / 2
        y_ref2 = (line2_ymin + line2_ymax) / 2
        x_line1 = get_line_intersection(y_ref1, line1)
        x_line2 = get_line_intersection(y_ref2, line2)
    else:
        ymin = max(line1_ymin, line2_ymin)
        ymax = min(line1_ymax, line2_ymax)
        y_ref = (ymin + ymax) / 2
        x_line1 = get_line_intersection(y_ref, line1)
        x_line2 = get_line_intersection(y_ref, line2)

    if x_line1 < x_line2:
        return -1
    elif x_line1 == x_line2:
        return 0
    else:
        return 1


@PIPELINES.register_module
class CollectLanePoints(Collect):
    def __init__(
            self,
            down_scale,
            keys,
            meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg'),
            hm_down_scale=None,
            fpn_layer_num=4,
            kpt_downscale=1,
            line_width=3,
            max_mask_sample=5,
            perspective=False,
            radius=2,
            root_radius=6,
            vanished_radius=8,
            joint_nums=4,
            joint_weights=None,
            joint_weights1=None, 
            joint_weights2=None, 
            joint_weights3=None, 
            joint_weights4=None,
            sample_per_lane=[60, 30, 15, 8],
            max_lane_num=6,
            fpn_down_scale=[4, 8, 16, 32],
            lane_extend=False,
    ):
        super(CollectLanePoints, self).__init__(keys, meta_keys)
        self.down_scale = down_scale
        self.fpn_layer_num = fpn_layer_num
        self.hm_down_scale = hm_down_scale if hm_down_scale is not None else down_scale
        self.line_width = line_width
        self.max_mask_sample = max_mask_sample
        self.radius = radius
        self.root_radius = root_radius
        self.vanished_radius = vanished_radius
        self.kpt_downscale = kpt_downscale
        self.joint_nums = joint_nums
        self.joint_weights = joint_weights
        self.joint_weights1 = joint_weights1 
        self.joint_weights2 = joint_weights2 
        self.joint_weights3 = joint_weights3 
        self.joint_weights4 = joint_weights4 
        self.sample_per_lane = sample_per_lane
        self.max_lane_num = max_lane_num
        self.fpn_down_scale = fpn_down_scale
        self.lane_extend = lane_extend
        self.number=0

    def target(self, results):
        def min_dis_one_point(points, idx):
            min_dis = 1e6
            for i in range(len(points)):
                if i == idx:
                    continue
                else:
                    d = cal_dis(points[idx], points[i])
                    if d < min_dis:
                        min_dis = d
            return min_dis

        def assign_weight(dis, h, joints, weights=None):
            if weights is None:
                weights = [1, 0.4, 0.2]
            step = h // joints
            weight = 1
            if dis < 0:
                weight = weights[2]
            elif dis < 2 * step:
                weight = weights[0]
            else:
                weight = weights[1]
            return weight

        output_h = int(results['img_shape'][0])
        output_w = int(results['img_shape'][1])
        mask_h = int(output_h // self.down_scale)
        mask_w = int(output_w // self.down_scale)
        hm_h = int(output_h // self.hm_down_scale)
        hm_w = int(output_w // self.hm_down_scale)
        results['hm_shape'] = [hm_h, hm_w]
        results['mask_shape'] = [mask_h, mask_w]

        
        gt_hm = np.zeros((1, hm_h, hm_w), np.float32)
        reg_hm = np.zeros((2, hm_h, hm_w), np.float32)  
        cfe_hm = np.zeros((2 * self.joint_nums, hm_h, hm_w), np.float32)  
        cfe_hm_end = np.zeros((2 * self.joint_nums, hm_h, hm_w), np.float32)  
        gas_hm_1 = np.zeros((2 * self.joint_nums, hm_h, hm_w), np.float32)  
        gas_hm_2 = np.zeros((2 * self.joint_nums, hm_h, hm_w), np.float32)  
        gas_hm_3 = np.zeros((2 * self.joint_nums, hm_h, hm_w), np.float32)  
        offset_mask = np.zeros((1, hm_h, hm_w), np.float32)
        offset_mask_weight = np.zeros((2 * self.joint_nums, hm_h, hm_w), np.float32)
        offset_mask_weight1 = np.zeros((2 * self.joint_nums, hm_h, hm_w), np.float32)  
        offset_mask_weight2 = np.zeros((2 * self.joint_nums, hm_h, hm_w), np.float32)  
        offset_mask_weight3 = np.zeros((2 * self.joint_nums, hm_h, hm_w), np.float32)  
        offset_mask_weight4 = np.zeros((2 * self.joint_nums, hm_h, hm_w), np.float32)  

        instance_mask = -1 * np.ones([hm_h, hm_w])
        exist_mask = np.zeros([hm_h, hm_w])
        int_error_mask = np.zeros([2, hm_h, hm_w])

        gt_kpts_hm = np.zeros((1, hm_h, hm_w), np.float32)
        gt_vp_hm = np.zeros((1, hm_h, hm_w), np.float32)
        gt_masks = []

        
        gt_points = results['gt_points']  
        end_points = []
        start_points = []
        for l in range(self.fpn_layer_num):
            lane_points = []
            fpn_down_scale = self.fpn_down_scale[l]
            fn_h = int(output_h // fpn_down_scale)
            fn_w = int(output_w // fpn_down_scale)
            for i, pts in enumerate(gt_points):  
                
                pts = convert_list(pts, fpn_down_scale)
                pts = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))  
                if self.lane_extend:
                    pts = ploy_fitting_cube_extend(pts, fn_h, fn_w, self.sample_per_lane[l])
                else:
                    pts = ploy_fitting_cube(pts, fn_h, fn_w, self.sample_per_lane[l])
                if pts is not None:
                    pts_f = clip_line(pts, fn_h, fn_w)
                    pts = np.int32(pts_f)
                    lane_points.append(pts[None, :, ::-1])  
            lane_points_align = -1 * np.ones([self.max_lane_num, self.sample_per_lane[l], 2])
            if len(lane_points) != 0:
                lane_points_align[:len(lane_points)] = np.concatenate(lane_points, axis=0)
            else:
                gauss_mask = gt_hm
            results[f'lane_points_l{l}'] = DC(to_tensor(lane_points_align).float(), stack=True, pad_dims=None)
        
        
        
        


        def get_sample_points_b(control_points,num_cp=100,order=3):
            #in and out: torch.Size([4, 4, 2]) torch.Size([4, 100, 2])
            
            
            
            control_points_matrix=np.array(control_points).reshape(-1,2)
            bspline_matrix=np.zeros((control_points_matrix.shape[0],num_cp,2))
            #cv = control_points_matrix[i,:,:] #np.array([[ 50.,  25.],[ 59.,  12.],[ 50.,  10.],[ 57.,   2.],[ 40.,   4.],[ 40.,   14.]])
            
            bspline_matrix= scipy_bspline(control_points_matrix,n=100,degree=order,periodic=False)
            
            return bspline_matrix.reshape(-1).tolist()

        def get_middle_control_points_b(x, y,key_p=3,order=3):
            '''
            dy = y[1:] - y[:-1]
            dx = x[1:] - x[:-1]
            dt = (dx ** 2 + dy ** 2) ** 0.5
            t = dt / dt.sum()
            t = np.hstack(([0], t))
            t = t.cumsum()
            '''
            bs = BS_curve(key_p,order)
            data = np.array([x,y]).T
            paras = bs.estimate_parameters(data)
            knots = bs.get_knots()
            if bs.check():
                control_points = bs.approximation(data)
                

            '''
            data = np.column_stack((x, y))
            Pseudoinverse = np.linalg.pinv(self.bezier_coeff(t))  
            control_points = Pseudoinverse.dot(data)  
            '''
            medi_ctp = control_points[:, :].flatten().tolist()
            return medi_ctp

        def get_control_points_b(x, y, interpolate=False,key_p=8):
            
            #print('-------',x.shape,y.shape)
            control_points=[]

            
            
            
            

            middle_points = get_middle_control_points_b(x, y,key_p) 
            for idx in range(0, len(middle_points) - 1, 2):
                
                control_points.append(middle_points[idx])
                control_points.append(middle_points[idx + 1])

            return control_points

        def curve_fit(lane_in):
            if len(lane_in)<8:
                return lane_in
            lane_out=[]
            x_points=[lane_in[i] for i in range(len(lane_in)) if i%2==0]
            y_points=[lane_in[i] for i in range(len(lane_in)) if i%2==1]
            lane_out=get_control_points_b(x_points, y_points, interpolate=False,key_p=20)
            lane_out=get_sample_points_b(lane_out,num_cp=100,order=3)
            return lane_out 

        gt_points_temp=[]
        for lane_gt in range(len(gt_points)):
            

            temp=curve_fit(gt_points[lane_gt]) 
            

            
            gt_points_temp.append(temp)
        gt_points=gt_points_temp

        
        for pts in gt_points:  
            
            
            id_class = 1
            
            pts = convert_list(pts, self.hm_down_scale)
            if len(pts) < 2:
                continue
            
            

            if self.lane_extend:
                pts = ploy_fitting_cube_extend(pts, hm_h, hm_w, int(360 / self.hm_down_scale))
            else:
                pts = ploy_fitting_cube(pts, hm_h, hm_w, int(360 / self.hm_down_scale))
            if pts is None:
                continue

            pts = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))  
            pts = clamp_line(pts, box=[0, 0, hm_w - 1, hm_h - 1], min_length=1)

            if pts is not None and len(pts) > 1:
                joint_points = []
                joint_points1 = []  
                start_point, end_point = pts[0], pts[-1]
                delta_idx = len(pts) // self.joint_nums
                end_points.append(end_point)
                start_points.append(start_point)
                for i in range(self.joint_nums):
                    joint_points.append(pts[i * delta_idx])
                    joint_points1.append(pts[-1]) 



                flag=0   
                public_points=joint_points.copy()  
                
                for pt in pts:
                    pt_int = (int(pt[0]), int(pt[1]))
                    draw_umich_gaussian(gt_kpts_hm[0], pt_int, radius=self.radius)  
                    
                    
                    reg_x = pt[0] - pt_int[0]
                    reg_y = pt[1] - pt_int[1]
                    reg_hm[0, pt_int[1], pt_int[0]] = reg_x  
                    reg_hm[1, pt_int[1], pt_int[0]] = reg_y  
                    if abs(reg_x) < 2 and abs(reg_y) < 2:
                        offset_mask[0, pt_int[1], pt_int[0]] = 1  

                    

                    max_x = abs(start_point[0] - end_point[0])
                    max_y = abs(start_point[1] - end_point[1])

                    for i in range(self.joint_nums):
                        offset_x = joint_points[i][0] - pt[0]
                        offset_y = joint_points[i][1] - pt[1]
                        
                        
                        offset_x_end = joint_points1[i][0] - pt[0]  
                        offset_y_end = joint_points1[i][1] - pt[1]   
                        if flag ==1:
                            offset_x1=public_points[i][0] - pt[0]     
                            offset_y1=public_points[i][1] - pt[1]     
                            mask_value2 = assign_weight(offset_y1, max_y, self.joint_nums, self.joint_weights2)  
                            offset_mask_weight2[2 * i + 1, pt_int[1], pt_int[0]] = mask_value2  
                            offset_mask_weight2[2 * i + 1, pt_int[1], pt_int[0]] = mask_value2
                            gas_hm_1[2 * i, pt_int[1], pt_int[0]] = offset_x1  
                            gas_hm_1[2 * i + 1, pt_int[1], pt_int[0]] = offset_y1  

                        if flag ==2:
                            offset_x2=public_points[i][0] - pt[0]     
                            offset_y2=public_points[i][1] - pt[1]     
                            mask_value3 = assign_weight(offset_y2, max_y, self.joint_nums, self.joint_weights3)  
                            offset_mask_weight3[2 * i + 1, pt_int[1], pt_int[0]] = mask_value3 
                            offset_mask_weight3[2 * i + 1, pt_int[1], pt_int[0]] = mask_value3
                            gas_hm_2[2 * i, pt_int[1], pt_int[0]] = offset_x2  
                            gas_hm_2[2 * i + 1, pt_int[1], pt_int[0]] = offset_y2  

                        if flag ==3:
                            offset_x3=public_points[i][0] - pt[0]     
                            offset_y3=public_points[i][1] - pt[1]     
                            mask_value4 = assign_weight(offset_y3, max_y, self.joint_nums, self.joint_weights4)  
                            offset_mask_weight4[2 * i + 1, pt_int[1], pt_int[0]] = mask_value4  
                            offset_mask_weight4[2 * i + 1, pt_int[1], pt_int[0]] = mask_value4
                            gas_hm_3[2 * i, pt_int[1], pt_int[0]] = offset_x3  
                            gas_hm_3[2 * i + 1, pt_int[1], pt_int[0]] = offset_y3  

                        mask_value = assign_weight(offset_y, max_y, self.joint_nums, self.joint_weights)
                        mask_value1 = assign_weight(offset_y_end, max_y, self.joint_nums, self.joint_weights1) 

                        offset_mask_weight[2 * i, pt_int[1], pt_int[0]] = mask_value
                        offset_mask_weight[2 * i + 1, pt_int[1], pt_int[0]] = mask_value

                        offset_mask_weight1[2 * i, pt_int[1], pt_int[0]] = mask_value1
                        offset_mask_weight1[2 * i + 1, pt_int[1], pt_int[0]] = mask_value1

                        cfe_hm[2 * i, pt_int[1], pt_int[0]] = offset_x
                        cfe_hm[2 * i + 1, pt_int[1], pt_int[0]] = offset_y
                        cfe_hm_end[2 * i, pt_int[1], pt_int[0]] = offset_x_end  
                        cfe_hm_end[2 * i + 1, pt_int[1], pt_int[0]] = offset_y_end  
                flag+=1

        
        if len(start_points) > 0:
            for start_point in start_points:
                draw_umich_gaussian(gt_hm[0], start_point, radius=self.root_radius)  

        results['gt_cpts_hm'] = DC(to_tensor(gt_hm).float(), stack=True, pad_dims=None)
        results['gt_kpts_hm'] = DC(to_tensor(gt_kpts_hm).float(), stack=True, pad_dims=None)
        results['int_offset'] = DC(to_tensor(reg_hm).float(), stack=True, pad_dims=None)  
        results['pts_cfe'] = DC(to_tensor(cfe_hm).float(), stack=True,   
                                   pad_dims=None)  

        results['pts_cfe_end'] = DC(to_tensor(cfe_hm_end).float(), stack=True,
                                   pad_dims=None)  
                                   
        results['pts_gas_1'] = DC(to_tensor(gas_hm_1).float(), stack=True,
                                   pad_dims=None)  
                                   
        results['pts_gas_2'] = DC(to_tensor(gas_hm_1).float(), stack=True,
                                   pad_dims=None)  
                                   
        results['pts_gas_3'] = DC(to_tensor(gas_hm_1).float(), stack=True,
                                   pad_dims=None)  
                                   
        results['offset_mask'] = DC(to_tensor(offset_mask).float(), stack=True,
                                    pad_dims=None)  
        results['offset_mask_weight'] = DC(to_tensor(offset_mask_weight).float(), stack=True,
                                           pad_dims=None)  
        results['offset_mask_weight1'] = DC(to_tensor(offset_mask_weight1).float(), stack=True,
                                           pad_dims=None)  
        results['offset_mask_weight2'] = DC(to_tensor(offset_mask_weight2).float(), stack=True,
                                           pad_dims=None)  
        results['offset_mask_weight3'] = DC(to_tensor(offset_mask_weight3).float(), stack=True,
                                           pad_dims=None)  
        results['offset_mask_weight4'] = DC(to_tensor(offset_mask_weight4).float(), stack=True,
                                           pad_dims=None)  
        results['gt_vp_hm'] = DC(to_tensor(gt_vp_hm).float(), stack=True, pad_dims=None)
        results['gt_masks'] = gt_masks
        results['down_scale'] = self.down_scale
        results['hm_down_scale'] = self.hm_down_scale

        return True

    def __call__(self, results):
        data = {}
        img_meta = {}
        valid = self.target(results)  
        if not valid:
            return None
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        self.keys.append('pts_cfe_end') 
        self.keys.append('pts_gas_1') 
        self.keys.append('pts_gas_2') 
        self.keys.append('pts_gas_3') 
        self.keys.append('offset_mask_weight1')  
        self.keys.append('offset_mask_weight2')  
        self.keys.append('offset_mask_weight3')  
        self.keys.append('offset_mask_weight4')  
        for key in self.keys:
            data[key] = results[key]
        return data
