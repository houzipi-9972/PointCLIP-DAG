import torch
import torch.nn as nn
import numpy as np

from xmuda.models.UniDSeg_clip.seg_clip import SegCLIP
from xmuda.models.UniDSeg_clip.open_vocab import OpenVocabClassifier
from xmuda.models.spconv_unet_v1m1_base import SpUNetBase


class Net2DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_2d,
                 open_vocab=False,
                 class_names=(),
                 text_prompt_templates=("a photo of a {}",),
                 ov_logit_scale=30.0,
                 ):
        super(Net2DSeg, self).__init__()

        # 2D network
        if backbone_2d == 'ViT-B-16' or backbone_2d == 'ViT-L-14' or backbone_2d == 'SAM_ViT-L':
            self.net_2d = SegCLIP(backbone_2d)
            feat_channels = 64
        else:
            raise NotImplementedError('2D backbone {} not supported'.format(backbone_2d))

        self.open_vocab = open_vocab
        if self.open_vocab:
            self.ov_classifier = OpenVocabClassifier(
                in_channels=feat_channels,
                class_names=class_names,
                backbone_2d=backbone_2d,
                prompt_templates=text_prompt_templates,
                logit_scale=ov_logit_scale,
            )
        else:
            self.linear = nn.Linear(feat_channels, num_classes)

        # 2nd segmentation head
        self.dual_head = dual_head
        if dual_head:
            if self.open_vocab:
                self.ov_classifier2 = OpenVocabClassifier(
                    in_channels=feat_channels,
                    class_names=class_names,
                    backbone_2d=backbone_2d,
                    prompt_templates=text_prompt_templates,
                    logit_scale=ov_logit_scale,
                )
            else:
                self.linear2 = nn.Linear(feat_channels, num_classes)

    def forward(self, data_batch):
        # (batch_size, 3, H, W)
        img = data_batch['img']
        depth = data_batch['depth']
        img_indices = data_batch['img_indices']

        x = self.net_2d(img, depth)

        # 2D-3D feature lifting
        img_feats = []
        num_points = []
        for i in range(x.shape[0]):
            proj_feats = x.permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]]
            img_feats.append(proj_feats)

            num_point = proj_feats.shape[0]
            num_points.append(num_point)
        img_feats = torch.cat(img_feats, 0)

        if self.open_vocab:
            x = self.ov_classifier(img_feats)
        else:
            x = self.linear(img_feats)

        preds = {
            'feats': img_feats,
            'seg_logit': x,
            'num_points': num_points,
        }

        if self.dual_head:
            if self.open_vocab:
                preds['seg_logit2'] = self.ov_classifier2(img_feats)
            else:
                preds['seg_logit2'] = self.linear2(img_feats)

        return preds


### 可用中间层的spconv
class Net3DSeg(nn.Module):
    def __init__(self,
                 num_classes,
                 dual_head,
                 backbone_3d,
                 backbone_3d_kwargs,
                 open_vocab=False,
                 class_names=(),
                 backbone_2d_for_text="ViT-L-14",
                 text_prompt_templates=("a photo of a {}",),
                 ov_logit_scale=30.0,
                 ):
        super(Net3DSeg, self).__init__()

        # 3D network
        if backbone_3d == 'SCN':
            in_channels = getattr(backbone_3d_kwargs, "in_channels", 1)
            self.net_3d = SpUNetBase(in_channels=in_channels, num_classes=num_classes)
        else:
            raise NotImplementedError('3D backbone {} not supported'.format(backbone_3d))

        # 2nd segmentation head
        self.dual_head = dual_head
        self.open_vocab = open_vocab
        if self.open_vocab:
            # SpUNet decoder point features have channel dimension 64 by default.
            self.ov_classifier = OpenVocabClassifier(
                in_channels=64,
                class_names=class_names,
                backbone_2d=backbone_2d_for_text,
                prompt_templates=text_prompt_templates,
                logit_scale=ov_logit_scale,
            )
            if self.dual_head:
                self.ov_classifier2 = OpenVocabClassifier(
                    in_channels=64,
                    class_names=class_names,
                    backbone_2d=backbone_2d_for_text,
                    prompt_templates=text_prompt_templates,
                    logit_scale=ov_logit_scale,
                )

    def forward(self, data_batch):
        inter_feat_3d, _ = self.net_3d.encoder_forward(data_batch['x'])
        # inter_feat_3d中的参数会被pop掉
        x1, x2, decoder_out = self.net_3d.decoder_forward(inter_feat_3d)

        if self.open_vocab:
            point_feats = decoder_out.features
            x1 = self.ov_classifier(point_feats)
            if self.dual_head:
                x2 = self.ov_classifier2(point_feats)

        preds = {
            'seg_logit': x1,
        }

        if self.dual_head:
            preds['seg_logit2'] = x2

        return preds


if __name__ == '__main__':
    test_Net2DSeg()
    test_Net3DSeg()
