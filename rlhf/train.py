# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:08

import logging

import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.utils.trainer import SimpleModelCheckpoint

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser

from data_processer import DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN, DEFAULT_BOS_TOKEN
from data_utils import NN_DataHelper, train_info_args, get_deepspeed_config
from models import MyTransformer,LoraArguments,LoraConfig,PPOArguments,PPOConfig


from deep_training.nlp.rl.ppo.ppo_trainer import PPOTrainer


class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        lora_args:LoraConfig= self.external_kwargs['lora_args']
        if lora_args is not None:
            self.weight_file = './best_ckpt'
            self.last_weight_file = './last_ckpt'

    def load_model_from_ckpt(self):
        model_args = self.external_kwargs['model_args']
        training_args = self.external_kwargs['training_args']
        lora_args = LoraArguments.from_pretrained(self.last_weight_file)
        pl_module = MyTransformer(lora_args=lora_args,
                              config=config,
                              model_args=model_args,
                              training_args=training_args)


        pl_module.backbone.from_pretrained(pl_module.backbone.model,self.last_weight_file)
        return pl_module


    def on_save_model(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        lora_args : LoraArguments =  self.external_kwargs['lora_args']
        # 保存权重
        if lora_args is None:
            super(MySimpleModelCheckpoint, self).on_save_model(trainer, pl_module)
        else:
            monitor_candidates = self._monitor_candidates(trainer)
            monitor_candidates.update(self.on_get_metric(trainer, pl_module))
            val = monitor_candidates.get(self.monitor, None)

            #保存loss最小权重
            if self.update_best(val):
                logging.info('epoch {} ,step {} , save best {}, {}\n'.format(monitor_candidates['epoch'],
                                                                             monitor_candidates['step'],
                                                                             self.best[self.monitor],
                                                                             self.weight_file))
                pl_module.backbone.save_pretrained(self.weight_file)
            #保存最新权重
            pl_module.backbone.save_pretrained(self.last_weight_file)
            # # 从最新权重加载模型
            # pl_module = self.load_model_from_ckpt()




if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments, PPOArguments))
    model_args, training_args, data_args, lora_args, ppo_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config
    ppo_args = ppo_args.config

    deepspeed_config = get_deepspeed_config()
    # 保存最小loss模型
    if lora_args is not None:
        assert deepspeed_config is None, ValueError('lora mode does not support deepspeed')
        checkpoint_callback = MySimpleModelCheckpoint(
            # monitor="loss",
            save_weights_only=True,
            every_n_epochs=1,
            every_n_train_steps=2000 // training_args.gradient_accumulation_steps,
            # 模型参数
            model_args=model_args,
            training_args=training_args,
            lora_args=lora_args, )
    else:
        checkpoint_callback = ModelCheckpoint(
            # monitor='loss',
            './best_ckpt',
            save_weights_only=True,
            save_last=True,
            save_top_k=1,
            # every_n_train_steps=1000,
            every_n_epochs=1)

    strategy = 'ddp' if torch.cuda.device_count() > 1 else 'auto'
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config, )

    trainer = PPOTrainer(
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,
        checkpoint_dir=data_args.output_dir,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        strategy=strategy
        # precision=16,#半精度
    )


    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config()
    config.decoder_start_token_id = config.bos_token_id

    if "llama" in model_args.model_name_or_path.lower() and tokenizer.bos_token_id != DEFAULT_BOS_TOKEN:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == -1:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # config.num_hidden_layers = 1

    # 额外参数
    checkpoint_callback.tokenizer = tokenizer
    checkpoint_callback.data_args = data_args

    config.save_pretrained('best_ckpt')

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, mode='test')

    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args, lora_args=lora_args)

    ckpt_path = './best_ckpt/best.pt'
    if not data_args.convert_onnx:
        #  只恢复权重 ， 不恢复步数和优化器 ，
        #  如果想恢复步数， 修改 trainer.fit(pl_model, train_dataloaders=train_datasets，ckpt=ckpt_path)  注lora 当前不支持恢复步数。
        # if os.path.exists(ckpt_path):
        #     if  lora_args is None:
        #         # 加载权重继续训练
        #         pl_model = MyTransformer.load_from_checkpoint(ckpt_path, config=config,model_args=model_args,training_args=training_args,lora_args=lora_args,strict=False)
        #     else:
        #         # 加载lora权重 继续训练  0.0.20版本支持lora 继续训练
        #         pl_model.backbone.from_pretrained(pl_model.backbone.model, pretrained_model_name_or_path=ckpt_path,lora_config=lora_args,is_trainable=True,strict=False)

        train_datasets = dataHelper.load_distributed_random_sampler(
            dataHelper.train_files,
            with_load_memory=True,
            collate_fn=dataHelper.collate_fn,
            batch_size=training_args.train_batch_size,
            drop_last=True,  # 多卡建议扔掉
            num_processes=trainer.world_size, process_index=trainer.global_rank)

        if train_datasets is not None:
            trainer.fit(pl_model, train_loader=train_datasets)

    else:
        if lora_args is None:
            # 加载权重
            pl_model = MyTransformer.load_from_checkpoint(ckpt_path, config=config,
                                                          model_args=model_args,
                                                          training_args=training_args,
                                                          lora_args=lora_args, strict=False)


            model = pl_model.get_glm_model()
            # 保存huggingface model
            model.save_pretrained('huggingface_model', max_shard_size='10GB')
        else:
            # 加载权重
            lora_args = LoraArguments.from_pretrained('./best_ckpt')
            pl_module = MyTransformer(lora_args=lora_args,
                                      config=config,
                                      model_args=model_args,
                                      training_args=training_args)
            # 二次加载权重
            pl_module.backbone.from_pretrained(pl_module.backbone.model, pretrained_model_name_or_path='./best_ckpt',
                                               lora_config=lora_args)

            model = pl_model.get_llm_model()