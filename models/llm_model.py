# coding=utf8
# @Time    : 2023/5/12 21:31
# @Author  : tk
# @FileName: llm_model

from typing import List, Tuple, Optional, Union
import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.rl.ppo.configuration import PPOArguments, PPOConfig
from deep_training.nlp.rl.ppo.ppo_module import PPOModelLoss
from deep_training.nlp.rl.ilql.configuration import ILQLArguments, ILQLConfig
from deep_training.nlp.rl.ilql.ilql_module import ILQLModelLoss
from deep_training.nlp.utils import configure_optimizers
from torch import nn
from deep_training.nlp.models.lora.v2 import LoraModel, LoraArguments,LoraConfig
from deep_training.nlp.models.transformer import TransformerForCausalLM
from torch.optim import AdamW
from transformers import PreTrainedModel, HfArgumentParser, AutoConfig
from transformers.utils import ModelOutput
from config import reward_config
from deep_training.nlp.models.rl.modeling_ppo import AutoModelForCausalLMWithValueHead,CausalLMOutputWithValue
from deep_training.nlp.models.rl.modeling_ilql import AutoModelForCausalLMWithILQLHeads

load_in_8bit = False

'''
    reward model
'''

class MyRewardModel(TransformerForCausalLM):
    def __init__(self, *args, **kwargs):
        # 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
        load_in_8bit = kwargs.get('load_in_8bit', False)
        if not load_in_8bit:
            kwargs.pop("device_map", None)
        super(MyRewardModel, self).__init__(*args, **kwargs)

        base_model_prefix = self.base_model_prefix[:-1] if self.base_model_prefix.endswith('_') else self.base_model_prefix
        self.transformer_bone = getattr(self.model,base_model_prefix,None)
        assert self.transformer_bone is not None
        hidden_size = self.config.word_embed_proj_dim if getattr(self.config,'word_embed_proj_dim',None) else self.config.hidden_size
        self.score = nn.Linear(hidden_size, self.config.num_labels)

        if load_in_8bit:
            setattr(self.model, 'model_parallel', True)
            setattr(self.model, 'is_parallelizable', True)
            self.model.enable_input_require_grads()

    def forward_value(self,**batch):
        state = self.transformer_bone(**batch)[0]
        value = self.score(state)
        return value.squeeze(-1)


    def forward_loss(self,chosen_ids: torch.Tensor, chosen_values: torch.Tensor,
                     rejected_ids: torch.Tensor, rejected_values: torch.Tensor):
        chosen_mean_scores = []
        rejected_mean_scores = []
        loss = 0.
        bs = chosen_ids.size(0)
        # pad_id = torch.tensor(self.config.pad_token_id, dtype=chosen_ids.dtype, device=chosen_values.device)
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_value = chosen_values[i]
            rejected_value = rejected_values[i]

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

        loss /= bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return loss,chosen_mean_scores,rejected_mean_scores

    def forward_score(self,input_ids,values):
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
        return torch.stack(chosen_mean_scores)

    def forward_returns(self, **inputs):
        input_ids = inputs['input_ids']
        values = self.forward_value(**inputs)
        ends = torch.argmax((input_ids == self.config.eos_token_id).float(), dim=1).view(-1, 1)
        returns = torch.gather(values, 1, ends).squeeze(-1)
        return returns

    def compute_loss(self, *args, return_value_only=False, **batch) -> tuple:
        input_a, input_b = {}, {}
        for k, v in batch.items():
            i, k = (input_b, k[:-1]) if k.endswith('2') else (input_a, k)
            i[k] = v

        value_a = self.forward_value(**input_a)
        if len(input_b) > 0:
            value_b = self.forward_value(**input_b)
            loss, chosen_mean_scores, rejected_mean_scores = self.forward_loss(input_a["input_ids"], value_a,
                                                                               input_b["input_ids"], value_b)
            loss_dict = {
                "loss": loss,
                "chosen_mean_scores": chosen_mean_scores.mean(),
                "rejected_mean_scores": rejected_mean_scores.mean()
            }
            if self.training:
                return (loss_dict,)
            return (loss, value_a, value_b)


        if return_value_only:
            return (value_a,)
        scores = self.forward_score(batch["input_ids"], value_a)
        return (value_a, scores)



class PPOModelForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    def __init__(self,*args,**kwargs):
        # 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
        load_in_8bit = kwargs.get('load_in_8bit', False)
        if not load_in_8bit:
            kwargs.pop("device_map", None)
        super(PPOModelForCausalLMWithValueHead, self).__init__(*args, **kwargs)

        if load_in_8bit:
            setattr(self.model, 'model_parallel', True)
            setattr(self.model, 'is_parallelizable', True)
            self.model.enable_input_require_grads()


class ILQLModelForCausalLMWithILQLHeads(AutoModelForCausalLMWithILQLHeads):
    def __init__(self, *args, **kwargs):
        load_in_8bit = kwargs.get('load_in_8bit', False)
        if not load_in_8bit:
            kwargs.pop("device_map", None)
        super(ILQLModelForCausalLMWithILQLHeads, self).__init__(*args, **kwargs)

        if load_in_8bit:
            setattr(self.model, 'model_parallel', True)
            setattr(self.model, 'is_parallelizable', True)
            self.model.enable_input_require_grads()




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