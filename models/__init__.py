# -*- coding: utf-8 -*-
# @Time:  11:30
# @Author: tk
import torch

from models.llm_model import *
from models.rrhf_model import *
'''
    模型训练类
'''

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


    def forward_returns(self,*args,**kwargs):
        if self.lora_args is not None and self.lora_args.with_lora:
            model = self.backbone.model
        else:
            model = self.backbone
        return model.forward_returns(*args,**kwargs)

    def load_sft_weight(self,sft_weight_path: str,is_trainable=False,strict=False):
        if self.lora_args is not None and self.lora_args.with_lora:
            #加载lora权重
            self.backbone.from_pretrained(self.backbone.model, pretrained_model_name_or_path=sft_weight_path,
                                          is_trainable=is_trainable)
        else:
            # 加载sft 或者 p-tuning-v2权重
            self.get_llm_model().load_state_dict(torch.load(sft_weight_path), strict=strict)


class MyPPOTransformer(PPOModelForCausalLMWithValueHead, PPOModelLoss, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        ppo_args: PPOConfig = kwargs.pop('ppo_args', None)
        super(MyPPOTransformer, self).__init__(*args, **kwargs)

        self.lora_args = lora_args
        self.ppo_config = ppo_args
        if lora_args is not None and lora_args.with_lora:
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)


    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model

    @torch.no_grad()
    def generate(self,*args,**kwargs):
        return self.get_llm_model().generate(*args,**kwargs)

    def configure_optimizers(self):
        p = self.get_named_parameters(self.backbone)
        training_args = self.training_args
        optimizer = AdamW(p, lr=training_args.learning_rate,
                          eps=training_args.adam_epsilon,
                          betas=training_args.optimizer_betas,
                          weight_decay=training_args.weight_decay)
        return optimizer


    def training_step(self,*args, **inputs):
        outputs = self.compute_loss(*args, **inputs)
        return outputs

    def validation_step(self, batch):
        outputs = self.compute_loss(**batch)
        return outputs

    def compute_loss(self, *args, **inputs):
        return self.forward_ppo_loss(*args, **inputs)


    def forward_logits_values(self,*args,**kwargs):
        return self.model.forward(*args,**kwargs)

    def load_sft_weight(self, sft_weight_path: str, is_trainable=False, strict=False):
        if self.lora_args is not None and self.lora_args.with_lora:
            # 加载lora权重
            self.backbone.from_pretrained(self.backbone.model, pretrained_model_name_or_path=sft_weight_path,
                                          is_trainable=is_trainable)
        else:
            # 加载sft 或者 p-tuning-v2权重
            self.get_llm_model().load_state_dict(torch.load(sft_weight_path), strict=strict)


class MyILQLTransformer(ILQLModelForCausalLMWithILQLHeads, ILQLModelLoss, with_pl=True):
    def __init__(self, *args, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        ilql_args: ILQLConfig = kwargs.pop('ilql_args', None)
        if ilql_args is not None:
            kwargs.update({
                "two_qs": ilql_args.two_qs,
                "alpha": ilql_args.alpha,
            })
        super(MyILQLTransformer, self).__init__(*args, **kwargs)

        self.lora_args = lora_args
        self.ilql_config = ilql_args
        if lora_args is not None and lora_args.with_lora:
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)


    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        return self.backbone.model

    @torch.no_grad()
    def generate(self,*args,**kwargs):
        return self.get_llm_model().generate(*args,**kwargs)

    def configure_optimizers(self):
        p = self.get_named_parameters(self.backbone)
        training_args = self.training_args
        optimizer = AdamW(p, lr=training_args.learning_rate,
                          eps=training_args.adam_epsilon,
                          betas=training_args.optimizer_betas,
                          weight_decay=training_args.weight_decay)
        return optimizer


    def training_step(self,*args, **inputs):
        outputs = self.compute_loss(*args, **inputs)
        return outputs

    def validation_step(self, batch):
        outputs = self.compute_loss(**batch)
        return outputs

    def compute_loss(self, *args, **inputs):
        return self.forward_ilql_loss(*args, **inputs)


    def forward_logits_values(self,*args,**kwargs):
        return self.model.forward(*args,**kwargs)

    def load_sft_weight(self, sft_weight_path: str, is_trainable=False, strict=False):
        if self.lora_args is not None and self.lora_args.with_lora:
            # 加载lora权重
            self.backbone.from_pretrained(self.backbone.model, pretrained_model_name_or_path=sft_weight_path,
                                          is_trainable=is_trainable)
        else:
            # 加载sft 或者 p-tuning-v2权重
            self.get_llm_model().load_state_dict(torch.load(sft_weight_path), strict=strict)


class MyRRHFTransformer(RRHFModelForCausalLM):
    def __init__(self, *args, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        super(MyRRHFTransformer, self).__init__(*args, **kwargs)

        self.lora_args = lora_args
        if lora_args is not None and lora_args.with_lora:
            model = LoraModel(self.backbone, lora_args)
            print('*' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)

    def load_sft_weight(self, sft_weight_path: str, is_trainable=False, strict=False):
        if self.lora_args is not None and self.lora_args.with_lora:
            # 加载lora权重
            self.backbone.from_pretrained(self.backbone.model, pretrained_model_name_or_path=sft_weight_path,
                                          is_trainable=is_trainable)
        else:
            # 加载sft 或者 p-tuning-v2权重
            self.get_llm_model().load_state_dict(torch.load(sft_weight_path), strict=strict)


def load_reward_model(sft_model_dir,sft_weight_path=None) ->MyRewardTransformer:
    '''
        sft_model_dir: 模型配置路径 ， 路径下需存在config.json
        weight_path: 如果是lora 则是lora 权重路径 （）
                     如果是普通 或者 p-tuning-v2 则是权重文件
    '''

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(reward_config.train_info_args)
    lora_args = lora_args.config
    config = AutoConfig.from_pretrained(sft_model_dir)
    # 加载权重
    lora_args = LoraArguments.from_pretrained(sft_model_dir) if lora_args else None
    pl_module = MyRewardTransformer(config=config,model_args=model_args,training_args=training_args,lora_args=lora_args)

    # 加载lora sft 或者 sft 或者 p-tuning-v2 权重
    if lora_args and sft_weight_path is None:
        sft_weight_path = sft_model_dir
    pl_module.load_sft_weight(sft_weight_path)

    pl_module.eval()
    pl_module.requires_grad_(False)
    return pl_module


def load_ref_model(ref_train_info_args,sft_model_dir,sft_weight_path=None) ->MyPPOTransformer:
    '''
        sft_model_dir: 模型配置路径 ， 路径下需存在config.json
        weight_path: 如果是lora 则是lora 权重路径 （）
                     如果是普通 或者 p-tuning-v2 则是权重文件
    '''
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(ref_train_info_args)
    lora_args = lora_args.config
    config = AutoConfig.from_pretrained(sft_model_dir)
    # 加载权重
    lora_args = LoraArguments.from_pretrained(sft_model_dir) if lora_args else None
    pl_module = MyPPOTransformer(config=config,model_args=model_args,training_args=training_args,lora_args=lora_args)

    # 加载lora sft 或者 sft 或者 p-tuning-v2 权重
    if lora_args and sft_weight_path is None:
        sft_weight_path = sft_model_dir
    pl_module.load_sft_weight(sft_weight_path)

    pl_module.eval()
    pl_module.requires_grad_(False)
    return pl_module