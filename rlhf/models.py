# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:09

from typing import List, Tuple, Optional
import torch
from torch import nn
from deep_training.nlp.models.lora.v2 import LoraModel, LoraArguments,LoraConfig
from deep_training.nlp.models.transformer import TransformerForSequenceClassification
from transformers import PreTrainedModel


#如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
load_in_8bit = False


class MyTransformerSequenceClassification(TransformerForSequenceClassification):
    def __init__(self, *args, **kwargs):
        if load_in_8bit:
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        super(MyTransformerSequenceClassification, self).__init__(*args, **kwargs)

    def compute_loss(self, *args, **batch) -> tuple:
        rewards_a = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
        if 'input_ids2' in batch:
            rewards_b = self.model(input_ids=batch["input_ids2"], attention_mask=batch["attention_mask2"])[0]
            loss = -nn.functional.logsigmoid(rewards_a - rewards_b).mean()
            if self.training:
                return (loss,)
            return (loss,rewards_a,rewards_b),
        return (rewards_a,)




class MyTransformer(MyTransformerSequenceClassification, with_pl=True):
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