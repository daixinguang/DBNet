import logging
import sys

from mmdet.core import merge_aug_proposals

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class RPNTestMixin(object):

    if sys.version_info >= (3, 7):

        async def async_test_rpn(self, x, img_metas):
            sleep_interval = self.rpn_head.test_cfg.pop(
                'async_sleep_interval', 0.025)
            async with completed(
                    __name__, 'rpn_head_forward',
                    sleep_interval=sleep_interval):
                rpn_outs = self.rpn_head(x)

            proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas)
            return proposal_list

    def simple_test_rpn(self, x, img_metas):
        rpn_outs = self.rpn_head(x)
        proposal_list = self.rpn_head.get_bboxes(*rpn_outs, img_metas)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas):
        samples_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(samples_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
       
       
        aug_img_metas = []
        for i in range(samples_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
       
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta,
                                self.rpn_head.test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals
