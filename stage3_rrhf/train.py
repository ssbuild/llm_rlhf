# @Time    : 2023/4/19 23:03
# @Author  : tk
# @FileName: train.py
import logging

import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.nlp.models.lora.v2 import LoraArguments, LoraConfig
from deep_training.utils.trainer import SimpleModelCheckpoint
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser

from data_processer import DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN, DEFAULT_BOS_TOKEN
from data_utils import NN_DataHelper, train_info_args, get_deepspeed_config,global_args
from models import MyRRHFTransformer


class MySimpleModelCheckpoint(SimpleModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(MySimpleModelCheckpoint, self).__init__(*args, **kwargs)
        lora_args:LoraConfig= self.external_kwargs['lora_args']
        if lora_args is not None:
            self.weight_file = './best_ckpt'
            self.last_weight_file = './last_ckpt'

    def on_save_model(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        lora_args : LoraArguments =  self.external_kwargs['lora_args']
        # 保存权重
        if lora_args is None:
            super(MySimpleModelCheckpoint, self).on_save_model(trainer, pl_module)
        else:
            # 保存最新权重
            logging.info('step {} saving model'.format(trainer.global_step))
            # 保存最新权重
            pl_module.backbone.save_pretrained(self.weight_file)


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, lora_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config

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


    trainer = Trainer(
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        strategy=strategy,
        # lora int8 precision='32'
        precision='32' if global_args['load_in_8bit'] else '16',# 可以自行尝试  "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"
    )


    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    config_kwargs = {"torch_dtype": torch.float16}
    if global_args["num_layers"] > 0:
        config_kwargs[global_args["num_layers_key"]] = global_args["num_layers"]
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs=config_kwargs)
    config.decoder_start_token_id = config.bos_token_id

    if "llama" in model_args.model_name_or_path.lower() and tokenizer.bos_token_id != DEFAULT_BOS_TOKEN:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == -1:
            tokenizer.pad_token_id = tokenizer.eos_token_id

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

    pl_model = MyRRHFTransformer(config=config, model_args=model_args, training_args=training_args, lora_args=lora_args,
                                   load_in_8bit=global_args["load_in_8bit"],device_map={"": trainer.local_rank} if trainer.world_size > 1 else "auto")

    # 加载sft权重训练
    # pl_model.load_sft_weight('sft_weight.bin',is_trainable=True)

    # pl_model.bfloat16()
    pl_model.float()



    train_datasets = dataHelper.load_distributed_random_sampler(
        dataHelper.train_files,
        with_load_memory=True,
        collate_fn=dataHelper.collate_fn,
        batch_size=training_args.train_batch_size,
        drop_last=True,  # 多卡建议扔掉
        num_workers=0,  # num_workers for DataLoader
        num_processes=trainer.world_size, process_index=trainer.global_rank)

    if train_datasets is not None:
        trainer.fit(pl_model, train_dataloaders=train_datasets)