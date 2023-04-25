# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:09
import typing
from typing import List, Tuple, Optional
import torch
from deep_training.nlp.models.transformer_base import TransformerLightningModule
from deep_training.nlp.utils import configure_optimizers
from torch import nn
from torch.nn import functional as F
from deep_training.nlp.models.lora.v2 import LoraModel, LoraArguments,LoraConfig
from deep_training.nlp.models.transformer import TransformerForCausalLM
from transformers import PreTrainedModel


#如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
load_in_8bit = False


class MyTransformerLM(TransformerForCausalLM, with_pl=True):
    def __init__(self, *args, **kwargs):
        # 如果显卡支持int8 可以开启 ， 需安装依赖 pip install bitsandbytes
        if load_in_8bit:
            kwargs.update({"load_in_8bit": True, "device_map": "auto"})
        lora_args: LoraConfig = kwargs.pop('lora_args', None)
        super(MyTransformerLM, self).__init__(*args, **kwargs)

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


class RLHF_Engine:
    def __init__(self):
        self.actor_model: typing.Optional[nn.Module] = None
        self.ref_model: typing.Optional[nn.Module] = None
        self.critic_model: typing.Optional[nn.Module] = None
        self.reward_model: typing.Optional[nn.Module] = None

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

class MyTransformerForTraining(TransformerLightningModule):
    def __init__(self, *args, **kwargs):
        super(MyTransformerForTraining, self).__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.rlhf_engine = RLHF_Engine()
        self.actor_model = self.rlhf_engine.actor_model


    def configure_optimizers(self):
        p1 = self.get_named_parameters(self.rlhf_engine.actor_model)
        p2 = self.get_named_parameters(self.rlhf_engine.critic_model)

        optimizer1 = configure_optimizers(p1, self.training_args,
                                    self.trainer.estimated_stepping_batches)

        optimizer2 = configure_optimizers(p2, self.training_args,
                                    self.trainer.estimated_stepping_batches)
        o1,o2 = {},{}
        if len(optimizer1) == 2:
            o1['optimizer'] = optimizer1[0][0]
            o1['scheduler'] = optimizer1[1][0]

        if len(optimizer2) == 2:
            o2['optimizer'] = optimizer2[0][0]
            o2['scheduler'] = optimizer2[1][0]
        return (o1,o2)

    def training_step(self, batch):
        self.compute_loss(**batch)

    def compute_loss(self, *args, **inputs):
        opt1, opt2 = self.optimizers()
        self.toggle_optimizer(opt1)
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :],
                                          inputs['input_ids'][:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.log("actor_loss", actor_loss, prog_bar=True)
        self.manual_backward(actor_loss)
        opt1.step()
        opt1.zero_grad()
        self.untoggle_optimizer(opt1)

        self.toggle_optimizer(opt2)
        value = self.rlhf_engine.critic_model.forward_value(**batch,return_value_only=True,use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,start:],returns, action_mask[:, start:])
        self.log("critic_loss", critic_loss, prog_bar=True)
        self.manual_backward(critic_loss)
        opt2.step()
        opt2.zero_grad()
        self.untoggle_optimizer(opt2)

        return actor_loss, critic_loss

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns





    def _generate_sequence(self, prompts):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        with torch.no_grad():
            seq = self.actor_model.module.generate(prompts,
                                                   max_length=max_min_length,
                                                   min_length=max_min_length)

        # Filter out seq with no asnwers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        ans = seq[:, prompt_length:]
        self.prompt_length = prompt_length
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return out_seq



    def generate_experience(self, prompts):
        self.rlhf_engine.eval()
        seq = self._generate_sequence(prompts)
        self.rlhf_engine.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()

        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            reward_score = self.reward_model(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
            values = self.critic_model(
                seq, attention_mask, return_value_only=True)[0].detach()[:, :-1]

        logits = output.logits
        logits_ref = output_ref.logits

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask
        }
