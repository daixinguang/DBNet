import mmcv
import numpy as np
import torch

from .builder import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register_module()
class AnchorGenerator(object):

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        self.strides = strides
        self.base_sizes = list(strides) if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        assert ((octave_base_scale is not None
                and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        return len(self.strides)

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_sizes, device='cuda'):
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=16,
                                  device='cuda'):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),
                                      num_base_anchors).contiguous().view(-1)
        return valid

    def __repr__(self):
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str


@ANCHOR_GENERATORS.register_module()
class SSDAnchorGenerator(AnchorGenerator):

    def __init__(self,
                 strides,
                 ratios,
                 basesize_ratio_range,
                 input_size=300,
                 scale_major=True):
        assert len(strides) == len(ratios)
        assert mmcv.is_tuple_of(basesize_ratio_range, float)

        self.strides = strides
        self.input_size = input_size
        self.centers = [(stride / 2., stride / 2.) for stride in strides]
        self.basesize_ratio_range = basesize_ratio_range
        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (self.num_levels - 2))
        min_sizes = []
        max_sizes = []
        for ratio in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * ratio / 100))
            max_sizes.append(int(input_size * (ratio + step) / 100))
        if input_size == 300:
            if basesize_ratio_range[0] == 0.15: 
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
            else:
                raise ValueError(
                    'basesize_ratio_range[0] should be either 0.15'
                    'or 0.2 when input_size is 300, got '
                    f'{basesize_ratio_range[0]}.')
        elif input_size == 512:
            if basesize_ratio_range[0] == 0.1:  
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            else:
                raise ValueError('basesize_ratio_range[0] should be either 0.1'
                                 'or 0.15 when input_size is 512, got'
                                 ' {basesize_ratio_range[0]}.')
        else:
            raise ValueError('Only support 300 or 512 in SSDAnchorGenerator'
                             f', got {input_size}.')

        anchor_ratios = []
        anchor_scales = []
        for k in range(len(self.strides)):
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            anchor_ratio = [1.]
            for r in ratios[k]:
                anchor_ratio += [1 / r, r]  # 4 or 6 ratio
            anchor_ratios.append(torch.Tensor(anchor_ratio))
            anchor_scales.append(torch.Tensor(scales))

        self.base_sizes = min_sizes
        self.scales = anchor_scales
        self.ratios = anchor_ratios
        self.scale_major = scale_major
        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            base_anchors = self.gen_single_level_base_anchors(
                base_size,
                scales=self.scales[i],
                ratios=self.ratios[i],
                center=self.centers[i])
            indices = list(range(len(self.ratios[i])))
            indices.insert(1, len(indices))
            base_anchors = torch.index_select(base_anchors, 0,
                                              torch.LongTensor(indices))
            multi_level_base_anchors.append(base_anchors)
        return multi_level_base_anchors

    def __repr__(self):
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}input_size={self.input_size},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}basesize_ratio_range='
        repr_str += f'{self.basesize_ratio_range})'
        return repr_str


@ANCHOR_GENERATORS.register_module()
class LegacyAnchorGenerator(AnchorGenerator):

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * (w - 1)
            y_center = self.center_offset * (h - 1)
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = [
            x_center - 0.5 * (ws - 1), y_center - 0.5 * (hs - 1),
            x_center + 0.5 * (ws - 1), y_center + 0.5 * (hs - 1)
        ]
        base_anchors = torch.stack(base_anchors, dim=-1).round()

        return base_anchors


@ANCHOR_GENERATORS.register_module()
class LegacySSDAnchorGenerator(SSDAnchorGenerator, LegacyAnchorGenerator):

    def __init__(self,
                 strides,
                 ratios,
                 basesize_ratio_range,
                 input_size=300,
                 scale_major=True):
        super(LegacySSDAnchorGenerator,
              self).__init__(strides, ratios, basesize_ratio_range, input_size,
                             scale_major)
        self.centers = [((stride - 1) / 2., (stride - 1) / 2.)
                        for stride in strides]
        self.base_anchors = self.gen_base_anchors()
