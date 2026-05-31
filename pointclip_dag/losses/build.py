from pointclip_dag.losses.ov_losses import OpenVocabularyLoss


def build_loss(cfg):
    return OpenVocabularyLoss(cfg.loss)
