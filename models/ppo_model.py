# coding=utf8
# @Time    : 2023/5/12 21:31
# @Author  : tk
# @FileName: llm_model

from typing import List, Tuple, Optional, Union
import torch
from deep_training.nlp.models.rl.modeling_ppo import AutoModelForCausalLMWithValueHead
from deep_training.nlp.rl.ppo.configuration import PPOConfig,PPOArguments
from deep_training.nlp.rl.ppo.ppo_module import PPOModelLoss
from transformers import AdamW
from deep_training.trainer.pl.modelweighter import *
import logging
logger = logging.getLogger(__name__)

class PPOModelForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    def __init__(self,*args,hidden_size=None, up_sampling_score=False,**kwargs):
        config = kwargs.get('config')
        if hidden_size is None:
            hidden_size = config.word_embed_proj_dim if getattr(config, 'word_embed_proj_dim',
                                                                None) else config.hidden_size
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

        super(PPOModelForCausalLMWithValueHead, self).__init__(*args,hidden_size=hidden_size, up_sampling_score=up_sampling_score, **kwargs)



    def enable_input_require_grads(self):
        setattr(self.model, 'model_parallel', True)
        setattr(self.model, 'is_parallelizable', True)
        # self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()








class MyPPOTransformer(PPOModelForCausalLMWithValueHead, PPOModelLoss,ModelWeightMinMax, with_pl=True):
    def __init__(self, *args,new_num_tokens=None, **kwargs):
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        prompt_args: PromptLearningConfig = kwargs.pop('prompt_args', None)
        ppo_args: PPOConfig = kwargs.pop('ppo_args', None)
        super(MyPPOTransformer, self).__init__(*args, **kwargs)

        self.lora_args = lora_args
        self.ppo_config = ppo_args
        self.prompt_args=prompt_args

        self.resize_token_embs(new_num_tokens)

        if lora_args is not None and lora_args.with_lora:
            self.backbone.enable_input_require_grads()
            model: LoraModel = LoraModel(self.backbone, lora_args, auto_prepare_kbit_training=False)
            print('==' * 30, 'lora info')
            model.print_trainable_parameters()
            self.set_model(model, copy_attr=False)
            # for name, module in model.named_modules():
            #     if isinstance(module, LoraLayer):
            #         module = module.to(torch.bfloat16)
            #     if 'norm' in name:
            #         module = module.to(torch.float32)
            #     if 'lm_head' in name or 'embed_tokens' in name:
            #         if hasattr(module, 'weight'):
            #             if module.weight.dtype == torch.float32:
            #                 module = module.to(torch.bfloat16)

    def resize_token_embs(self, new_num_tokens):
        if new_num_tokens is not None:
            logger.info(f"new_num_tokens:{new_num_tokens}")
            model: PreTrainedModel = self.backbone.model
            embedding_size = model.get_input_embeddings().weight.shape[0]
            if new_num_tokens != embedding_size:
                # lora ptv2 二次加载权重需备份原此词表
                if (self.lora_args is not None and self.lora_args.with_lora) or (
                        self.prompt_args is not None and self.prompt_args.with_prompt):
                    config = model.config
                    if config.task_specific_params is None:
                        config.task_specific_params = {}
                    config.task_specific_params['vocab_size'] = config.vocab_size

                logger.info("resize the embedding size by the size of the tokenizer")
                # print('before',self.config)
                model.resize_token_embeddings(new_num_tokens)
                # print('after',self.config)

    def get_model_lr(self, model=None, lr=None):
        # for n, p in self.named_parameters():
        #     print(n, p.requires_grad)
        lr = lr if lr is not None else self.config.task_specific_params['learning_rate']
        if self.lora_args is not None and self.lora_args.with_lora:
            return [(self.backbone, lr)]
        elif self.prompt_args and self.prompt_args.with_prompt:
            return [(self.backbone, lr)]
        return super(MyPPOTransformer, self).get_model_lr(model, lr)

    def get_llm_model(self) -> PreTrainedModel:
        if self.lora_args is not None and self.lora_args.with_lora:
            return self.backbone.model.model
        elif self.prompt_args is not None and self.prompt_args.with_prompt:
            # PromptModel 方法覆盖原来方法
            return self.backbone
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





class Generate:
    @classmethod
    @torch.no_grad()
    def generate(cls,model, tokenizer, query: str, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        # prompt = "Human：" + query + "\nAssistant："
        #自行加模板
        prompt = query
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        return response

    @classmethod
    @torch.no_grad()
    def chat(cls,model, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []

        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        history = history + [(query, response)]
        return response, history