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
        # 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
        load_in_8bit = kwargs.get('load_in_8bit', False)
        if not load_in_8bit:
            kwargs.pop("device_map", None)
        super(RRHFModelForCausalLM, self).__init__(*args, **kwargs)

        if load_in_8bit:
            setattr(self.model, 'model_parallel', True)
            setattr(self.model, 'is_parallelizable', True)
            self.model.enable_input_require_grads()

        # self.only_use_provide = kwargs.get("only_use_provide",False)
        # self.only_use_sample = kwargs.get("only_use_sample", False)
        self.length_penalty = kwargs.get('length_penalty',1.0)

    def gather_logits_labels(self, logits, labels):
        mask = (labels != -100).float()
        new_logits = logits.clone()  # Create a copy to avoid in-place modification
        labels[labels == -100] = 0
        output = torch.gather(new_logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask  # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length ** self.length_penalty)
        return scores

    def rrhf_loss(self, scores, idxs, rw_scores):
        diff = scores.unsqueeze(0) - scores.unsqueeze(-1)  # b * b
        rw_diff = rw_scores.unsqueeze(0) - rw_scores.unsqueeze(-1)  # b * b
        aval = torch.bitwise_and(rw_diff > 0, diff < 0)[0]
        return -diff[aval].sum()

    def sft_loss(self, logit_label, idxs, rw_scores):
        max_idx = torch.argmax(rw_scores)
        return -logit_label[max_idx].mean()

    def compute_loss(self, inputs, return_outputs=False):
        # if self.only_use_provide:
        #     inputs['input_ids'] = inputs['input_ids'][-2:]
        #     inputs['attention_mask'] = inputs['attention_mask'][-2:]
        #     inputs['labels'] = inputs['labels'][-2:]
        #     inputs["idxs"] = inputs["idxs"][:, -2:]
        #     inputs["scores"] = inputs["scores"][:, -2:]
        # if self.only_use_sample:
        #     inputs['input_ids'] = inputs['input_ids'][:-2]
        #     inputs['attention_mask'] = inputs['attention_mask'][:-2]
        #     inputs['labels'] = inputs['labels'][:-2]
        #     inputs["idxs"] = inputs["idxs"][:, :-2]
        #     inputs["scores"] = inputs["scores"][:, :-2]
        logits = self.forward(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))[
            0]  # (batch * cand) * L * V
        logits = F.log_softmax(logits, dim=-1)
        logit_label = self.gather_logits_labels(logits, inputs.get("labels"))
        scores = self.get_score(logit_label, inputs.get("labels"))
        rrhf_loss = self.rrhf_loss(scores, inputs.get("idxs"), inputs.get("scores"))
        sft_loss = self.sft_loss(logit_label, inputs.get("idxs"), inputs.get("scores"))
        loss = self.args.rrhf_weight * rrhf_loss + sft_loss
        return (loss, scores) if return_outputs else loss