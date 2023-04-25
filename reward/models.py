# @Time    : 2023/4/19 23:03
# @Author  : tk
# @FileName: models.py
from typing import List, Tuple, Optional
import torch
from torch import nn
from deep_training.nlp.models.lora.v2 import LoraModel, LoraArguments,LoraConfig
from deep_training.nlp.models.transformer import TransformerForTokenClassification
from transformers import PreTrainedModel


#如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
load_in_8bit = False


class MyRewardModel(TransformerForTokenClassification):
    def __init__(self, *args, **kwargs):
        if load_in_8bit:
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        super(MyRewardModel, self).__init__(*args, **kwargs)

        self.num_padding_at_beginning = 0

    def forward_reward(self,**batch):
        value = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
        return value.squeeze(-1)


    def forward_loss(self,
                     chosen_ids: torch.Tensor, chosen_values: torch.Tensor,
                     rejected_ids: torch.Tensor, rejected_values: torch.Tensor):
        chosen_mean_scores = []
        rejected_mean_scores = []
        loss = 0.
        seq_len = chosen_ids.size(1)

        pad_id = torch.tensor(self.config.pad_token_id, dtype=chosen_ids.dtype, device=chosen_values.device)
        for i in range(chosen_ids.size(0)):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_value = chosen_values[i]
            rejected_value = rejected_values[i]

            c_inds = (chosen_id == pad_id).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(c_inds) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the seoncd padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_value.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == pad_id).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item() if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_value[divergence_ind:end_ind]
            r_truncated_reward = rejected_value[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_value[c_ind - 1])  # use the end score for refrnence
            rejected_mean_scores.append(rejected_value[r_ind - 1])

            loss += -torch.log(
                torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / chosen_ids.size(0)
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return loss,chosen_mean_scores,rejected_mean_scores

    def forward_value(self,input_ids,values,prompt_length):
        bs = values.size(0)
        seq_len = input_ids.shape[1]
        chosen_end_scores = [
        ]  # we use this name for consistency with the original forwad function
        for i in range(bs):
            input_id = input_ids[i]
            value = values[i]

            c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
            # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
            c_ind = c_inds[0].item() + prompt_length if len(
                c_inds) > 0 else seq_len
            chosen_end_scores.append(value[c_ind - 1])
        return values,torch.stack(chosen_end_scores)


    def compute_loss(self, *args,return_value_only=False,prompt_length=0, **batch) -> tuple:
        value_a = self.forward_reward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if 'input_ids2' in batch:
            value_b = self.forward_reward(input_ids=batch["input_ids2"], attention_mask=batch["attention_mask2"])
            loss,chosen_mean_scores,rejected_mean_scores = self.forward_loss(batch["input_ids"],value_a,batch["input_ids2"],value_b)
            loss_dict = {
                "loss": loss,
                "chosen_mean_scores": chosen_mean_scores,
                "rejected_mean_scores": rejected_mean_scores
            }
            if self.training:
                return (loss_dict,)
            return (loss,value_a,value_b)

        values,chosen_end_scores = self.forward_value(batch["input_ids"],value_a,prompt_length)
        if return_value_only:
            return (values,)
        return (values,chosen_end_scores)




class MyTransformer(MyRewardModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        super(MyTransformer, self).__init__(*args, **kwargs)
        self.lora_args = lora_args
        if lora_args is not None and lora_args.with_lora:
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)

    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model
