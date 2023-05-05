# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:09
import typing
from typing import List, Tuple, Optional
import torch
from deep_training.nlp.utils import configure_optimizers
from deep_training.nlp.models.lora.v2 import LoraModel, LoraArguments,LoraConfig
from deep_training.nlp.models.transformer import TransformerForCausalLM
from transformers import PreTrainedModel
from deep_training.nlp.rl.ppo.ppo_module import PPOModelBase
from deep_training.nlp.rl.ppo.configuration import PPOConfig,PPOArguments


#如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
load_in_8bit = False


class MyTransformerForCausalLM(TransformerForCausalLM,PPOModelBase):
    def __init__(self, *args, **kwargs):
        # 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
        ppo_args: PPOConfig = kwargs['ppo_args']
        if load_in_8bit:
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        super(MyTransformerForCausalLM, self).__init__(*args, **kwargs)
        self.ppo_config = ppo_args

    def compute_loss(self, *args, **inputs):
        return self.forward_ppo_loss(*args, **inputs)



class MyTransformer(MyTransformerForCausalLM, with_pl=True):
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


    def configure_optimizers(self):
        p = self.get_named_parameters(self.rlhf_engine.actor_model)
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




