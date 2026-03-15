from xmuda.models.UniDSeg_clip.xmuda_arch import Net2DSeg, Net3DSeg
from xmuda.models.metric import SegIoU


def build_model_2d(cfg):
    class_names = tuple(cfg.MODEL_2D.CLASS_NAMES)
    if cfg.MODEL_2D.OPEN_VOCAB:
        assert len(class_names) > 0, "MODEL_2D.CLASS_NAMES must be set in open-vocabulary mode."
        assert cfg.MODEL_2D.NUM_CLASSES == len(class_names), \
            "MODEL_2D.NUM_CLASSES must match len(MODEL_2D.CLASS_NAMES) in open-vocabulary mode."
    model = Net2DSeg(num_classes=cfg.MODEL_2D.NUM_CLASSES,
                     backbone_2d=cfg.MODEL_2D.TYPE,
                     dual_head=cfg.MODEL_2D.DUAL_HEAD,
                     open_vocab=cfg.MODEL_2D.OPEN_VOCAB,
                     class_names=class_names,
                     text_prompt_templates=tuple(cfg.MODEL_2D.TEXT_PROMPTS),
                     ov_logit_scale=cfg.MODEL_2D.LOGIT_SCALE,
                     )
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name='seg_iou_2d')
    return model, train_metric


def build_model_3d(cfg):
    class_names = tuple(cfg.MODEL_3D.CLASS_NAMES)
    if cfg.MODEL_3D.OPEN_VOCAB:
        assert len(class_names) > 0, "MODEL_3D.CLASS_NAMES must be set in open-vocabulary mode."
        assert cfg.MODEL_3D.NUM_CLASSES == len(class_names), \
            "MODEL_3D.NUM_CLASSES must match len(MODEL_3D.CLASS_NAMES) in open-vocabulary mode."
    model = Net3DSeg(num_classes=cfg.MODEL_3D.NUM_CLASSES,
                     backbone_3d=cfg.MODEL_3D.TYPE,
                     dual_head=cfg.MODEL_3D.DUAL_HEAD,
                     backbone_3d_kwargs=cfg.MODEL_3D.get(cfg.MODEL_3D.TYPE, None),
                     open_vocab=cfg.MODEL_3D.OPEN_VOCAB,
                     class_names=class_names,
                     backbone_2d_for_text=cfg.MODEL_2D.TYPE,
                     text_prompt_templates=tuple(cfg.MODEL_3D.TEXT_PROMPTS),
                     ov_logit_scale=cfg.MODEL_3D.LOGIT_SCALE,
                     )
    train_metric = SegIoU(cfg.MODEL_3D.NUM_CLASSES, name='seg_iou_3d')
    return model, train_metric
