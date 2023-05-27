# -*- coding: utf-8 -*-
# @Time    : 2023/5/18 16:41
import torch
from torch.nn import functional as F
from deep_training.nlp.models.transformer import TransformerForCausalLM

__all__ = [
    'RRHFModelForCausalLM'
]

class RRHFModelForCausalLM(TransformerForCausalLM):
    def __init__(self,*args,**kwargs):
        # 如果显卡支持int8 可以开启
        load_in_8bit = kwargs.get('load_in_8bit', False)
        load_in_4bit = kwargs.get('load_in_4bit', False)
        if not load_in_4bit:
            quantization_config = kwargs.get("quantization_config", None)
            if quantization_config:
                load_in_4bit = quantization_config.load_in_4bit

        if not load_in_8bit and not load_in_4bit:
            kwargs.pop("device_map", None)
            kwargs.pop("quantization_config", None)
        super(RRHFModelForCausalLM, self).__init__(*args, **kwargs)


        self.length_penalty = kwargs.get('length_penalty',1.0)
        self.rrhf_weight = kwargs.get('rrhf_weight', 1.0)

    def enable_input_require_grads(self):
        setattr(self.model, 'model_parallel', True)
        setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

    def gather_logits_labels(self, logits, labels,mask):
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask  # B * L
        return output

    def get_score(self, logit_label, mask):
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length ** self.length_penalty)
        return scores

    def rrhf_loss(self, scores, rw_scores):
        diff = scores.unsqueeze(0) - scores.unsqueeze(-1)  # b * b
        rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1)  # b * b
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)[0]
        return -diff[aval].sum()

    def sft_loss(self, logit_label, rw_scores):
        max_idx = torch.argmax(rw_scores)
        return -logit_label[max_idx].mean()

    def compute_loss(self, **inputs):
        labels = inputs.pop('labels',None)
        scores = inputs.pop('scores', None)
        logits = self.model(**inputs)[0]  # (batch * cand) * L * V
        if labels is not None:
            labels = labels.long()
            logits = F.log_softmax(logits, dim=-1)
            mask = (labels != -100).float()
            logit_label = self.gather_logits_labels(logits, labels,mask)
            compute_scores = self.get_score(logit_label,mask)
            rrhf_loss = self.rrhf_loss(compute_scores, scores)
            sft_loss = self.sft_loss(logit_label, scores)
            loss = self.rrhf_weight * rrhf_loss + sft_loss
            loss_dict = {
                "rrhf_loss": rrhf_loss,
                "sft_loss": sft_loss,
                "loss": loss
            }
            return (loss_dict,)
        return (logits,)

