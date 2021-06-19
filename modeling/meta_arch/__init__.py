from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY, build_model  # isort:skip


# import all the meta_arch, so they will be registered
from .fcis import GeneralizedFCIS, FCISProposalNetwork

from detectron2.modeling.meta_arch.semantic_seg import SEM_SEG_HEADS_REGISTRY, SemanticSegmentor, build_sem_seg_head


__all__ = list(globals().keys())