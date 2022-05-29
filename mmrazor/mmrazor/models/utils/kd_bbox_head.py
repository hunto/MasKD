from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead


@HEADS.register_module()
class KDShared2FCBBoxHead(Shared2FCBBoxHead):

    def __init__(self, *args, **kwargs):
        super(KDShared2FCBBoxHead, self).__init__(*args, **kwargs)

    def loss(self, *args, **kwargs):
        return {}
