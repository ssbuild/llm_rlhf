# @Time    : 2023/4/19 23:03
# @Author  : tk
# @FileName: models.py
from typing import List, Tuple
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
        rewards_j = self.model(input_ids=batch["input_ids_j"], attention_mask=batch["attention_mask_j"])[0]
        rewards_k = self.model(input_ids=batch["input_ids_k"], attention_mask=batch["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if self.training:
            return loss
            # return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss




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