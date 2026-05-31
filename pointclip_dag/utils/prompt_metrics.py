from __future__ import annotations

import torch
import torch.nn.functional as F


class PromptConsistencyAccumulator:
    def __init__(self):
        self.total = 0
        self.top1 = 0
        self.js_sum = 0.0
        self.logit_cos_sum = 0.0

    def update(self, base_logits, alt_logits, labels, ignore_index):
        valid = (labels != ignore_index) & (labels >= 0)
        if valid.numel() == 0 or not bool(valid.any()):
            return
        base_logits = base_logits[valid].float()
        alt_logits = alt_logits[valid].float()
        base_pred = base_logits.argmax(dim=-1)
        alt_pred = alt_logits.argmax(dim=-1)
        p = F.softmax(base_logits, dim=-1)
        q = F.softmax(alt_logits, dim=-1)
        m = 0.5 * (p + q)
        js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        cos = F.cosine_similarity(base_logits, alt_logits, dim=-1)
        self.total += int(base_pred.numel())
        self.top1 += int(base_pred.eq(alt_pred).sum().item())
        self.js_sum += float(js.sum().item())
        self.logit_cos_sum += float(cos.sum().item())

    def compute(self):
        return {
            "prompt_top1_consistency": self.top1 / max(self.total, 1),
            "prompt_js_divergence": self.js_sum / max(self.total, 1),
            "prompt_logit_cosine": self.logit_cos_sum / max(self.total, 1),
        }


def make_prompt_variant_groups(vocabulary, mode="name_only"):
    groups = []
    for item in vocabulary.classes:
        if mode == "aliases_only" and item.aliases:
            phrases = list(item.aliases)
        else:
            phrases = [item.name]
        groups.append([f"a photo of a {phrase}." for phrase in phrases])
    return groups


def _kl(p, q):
    return (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum(dim=-1)
