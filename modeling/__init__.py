from detectron2.layers import ShapeSpec

from detectron2.modeling.anchor_generator import build_anchor_generator, ANCHOR_GENERATOR_REGISTRY
from .backbone import (
    BACKBONE_REGISTRY,
    Backbone,
    ResNet,
    ResNetBlockBase,
    build_backbone,
    build_resnet_backbone_custom,
    make_stage,
)
from detectron2.modeling.meta_arch import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    # GeneralizedRCNN,
    # PanopticFPN,
    # ProposalNetwork,
    SemanticSegmentor,
    build_model,
    build_sem_seg_head,
)
from .meta_arch import (
    GeneralizedFCIS,
    FCISProposalNetwork
)
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import (
    PROPOSAL_GENERATOR_REGISTRY,
    build_proposal_generator,
    RPN_HEAD_REGISTRY,
    build_rpn_head,
)
from detectron2.modeling.roi_heads import (
    ROI_BOX_HEAD_REGISTRY,
    ROI_HEADS_REGISTRY,
    ROI_KEYPOINT_HEAD_REGISTRY,
    ROI_MASK_HEAD_REGISTRY,
    ROIHeads,
    StandardROIHeads,
    BaseMaskRCNNHead,
    BaseKeypointRCNNHead,
    FastRCNNOutputLayers,
    build_box_head,
    build_keypoint_head,
    build_mask_head,
    build_roi_heads,
)
from detectron2.modeling.test_time_augmentation import DatasetMapperTTA, GeneralizedRCNNWithTTA
# from detectron2.modeling.mmdet_wrapper import MMDetBackbone, MMDetDetector

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]


# from detectron2.utils.env import fixup_module_metadata

# fixup_module_metadata(__name__, globals(), __all__)
# del fixup_module_metadata