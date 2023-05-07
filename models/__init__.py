# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk

from typing import List, Tuple, Optional
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.rl.ppo.configuration import PPOArguments, PPOConfig
from deep_training.nlp.rl.ppo.ppo_module import PPOModelBase
from deep_training.nlp.utils import configure_optimizers
from torch import nn
from deep_training.nlp.models.lora.v2 import LoraModel, LoraArguments,LoraConfig
from deep_training.nlp.models.transformer import TransformerForTokenClassification
from transformers import PreTrainedModel, HfArgumentParser,AutoConfig

from config import reward_config

#如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
load_in_8bit = False


class MyRewardModel(TransformerForTokenClassification):
    def __init__(self, *args, **kwargs):
        if load_in_8bit:
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        super(MyRewardModel, self).__init__(*args, **kwargs)

    def forward_reward(self,**batch):
        value = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
        return value.squeeze(-1)


    def forward_loss(self,
                     chosen_ids: torch.Tensor, chosen_values: torch.Tensor,
                     rejected_ids: torch.Tensor, rejected_values: torch.Tensor):
        chosen_mean_scores = []
        rejected_mean_scores = []
        loss = 0.
        # pad_id = torch.tensor(self.config.pad_token_id, dtype=chosen_ids.dtype, device=chosen_values.device)
        for i in range(chosen_ids.size(0)):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_value = chosen_values[i]
            rejected_value = rejected_values[i]
            # if torch.all(torch.eq(chosen_id, rejected_id)).item():
            #     c_inds = (chosen_id == self.config.pad_token_id).nonzero()
            #     c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen_id.shape[0]
            #     chosen_mean_scores.append(chosen_value[c_ind - 1])
            #     inference = True
            #     continue
            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen_id == self.config.pad_token_id).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen_id.shape[0]
            r_inds = (rejected_id == self.config.pad_token_id).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected_id.shape[0]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen_id != rejected_id).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_value[divergence_ind:end_ind]
            r_truncated_reward = rejected_value[divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_mean_scores.append(c_truncated_reward[-1])
            rejected_mean_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()

        loss = loss / chosen_ids.size(0)
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return loss,chosen_mean_scores,rejected_mean_scores

    def forward_value(self,input_ids,values):
        bs = values.size(0)
        seq_len = input_ids.shape[1]
        chosen_mean_scores = [
        ]  # we use this name for consistency with the original forwad function
        for i in range(bs):
            input_id = input_ids[i]
            value = values[i]
            c_inds = (input_id == self.config.pad_token_id).nonzero()
            # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
            c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len
            chosen_mean_scores.append(value[c_ind - 1])
        return values,torch.stack(chosen_mean_scores)


    def compute_loss(self, *args,return_value_only=False,**batch) -> tuple:
        value_a = self.forward_reward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if 'input_ids2' in batch:
            value_b = self.forward_reward(input_ids=batch["input_ids2"], attention_mask=batch["attention_mask2"])
            loss,chosen_mean_scores,rejected_mean_scores = self.forward_loss(batch["input_ids"],value_a,batch["input_ids2"],value_b)
            loss_dict = {
                "loss": loss,
                "chosen_mean_scores": chosen_mean_scores.mean(),
                "rejected_mean_scores": rejected_mean_scores.mean()
            }
            if self.training:
                return (loss_dict,)
            return (loss,value_a,value_b)

        values,chosen_mean_scores = self.forward_value(batch["input_ids"],value_a)
        if return_value_only:
            return (values,)
        return (values,chosen_mean_scores)




class MyRewardTransformer(MyRewardModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        super(MyRewardTransformer, self).__init__(*args, **kwargs)
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



class MyPPOTransformer(MyRewardModel,PPOModelBase, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        super(MyPPOTransformer, self).__init__(*args, **kwargs)
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


    def configure_optimizers(self):
        p = self.get_named_parameters(self.backbone)
        opt = configure_optimizers(p, self.training_args,
                                    self.trainer.estimated_stepping_batches)

        o = {}
        if len(opt) == 2:
            o['optimizer'] = opt[0][0]
            o['scheduler'] = opt[1][0]
        else:
            o['optimizer'] = opt[0]

        return (o,)

    def training_step(self, batch):
        outputs = self.compute_loss(**batch)
        return outputs

    def validation_step(self, batch):
        outputs = self.compute_loss(**batch)
        return outputs

    def compute_loss(self, *args, **inputs):
        return self.forward_ppo_loss(*args, **inputs)





def load_reward_model(model_dir) ->MyRewardTransformer:
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(reward_config.train_info_args)
    lora_args = lora_args.config
    config = AutoConfig.from_pretrained(model_dir)
    # 加载权重
    lora_args = LoraArguments.from_pretrained(model_dir)
    pl_module = MyRewardTransformer(lora_args=lora_args,config=config,model_args=model_args,training_args=training_args)
    # 二次加载权重
    pl_module.backbone.from_pretrained(pl_module.backbone.model, pretrained_model_name_or_path=model_dir,lora_config=lora_args)


    pl_module.eval()
    pl_module.requires_grad_(False)
    return pl_module


def load_ref_model(model_dir) ->MyRewardTransformer:
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(reward_config.train_info_args)
    lora_args = lora_args.config
    config = AutoConfig.from_pretrained(model_dir)
    # 加载权重
    lora_args = LoraArguments.from_pretrained(model_dir)
    pl_module = MyRewardTransformer(lora_args=lora_args,config=config,model_args=model_args,training_args=training_args)
    # 二次加载权重
    pl_module.backbone.from_pretrained(pl_module.backbone.model, pretrained_model_name_or_path=model_dir,lora_config=lora_args)

    pl_module.eval()
    pl_module.requires_grad_(False)
    return pl_module