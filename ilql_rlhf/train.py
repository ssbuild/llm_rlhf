# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 17:08
import sys
sys.path.append('..')

import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.trainer.pl.modelcheckpoint import FabricModelCheckpoint
from transformers import HfArgumentParser
from data_processer import DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN, DEFAULT_BOS_TOKEN
from data_utils import NN_DataHelper, train_info_args, get_deepspeed_config,global_args
from models import MyILQLTransformer, LoraArguments, LoraConfig, ILQLArguments, ILQLConfig
from deep_training.nlp.rl.ilql.ilql_trainer import ILQLTrainer
from lightning.fabric.strategies import DeepSpeedStrategy


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments, ILQLArguments))
    model_args, training_args, data_args, lora_args, ilql_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config
    ilql_args = ilql_args.config

    output_weight_dir = './best_ckpt'

    dataHelper = NN_DataHelper(model_args, training_args, data_args, ilql_args=ilql_args)
    config_kwargs = {"torch_dtype": torch.float16}
    if global_args["num_layers"] > 0:
        config_kwargs[global_args["num_layers_key"]] = global_args["num_layers"]
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs=config_kwargs)
    dataHelper.preprocess_tokenizer_config()

    config.save_pretrained(output_weight_dir)
    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, mode='test')

    deepspeed_config = get_deepspeed_config()
    strategy = 'ddp' if torch.cuda.device_count() > 1 else 'auto'
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config, )
    checkpoint_callback = FabricModelCheckpoint(
        # monitor="loss",
        dirpath=output_weight_dir,
        save_weights_only=True,
        every_n_epochs=1,
        every_n_train_steps=1000 // training_args.gradient_accumulation_steps,
        save_last=True,
        # 模型参数
        model_args=model_args,
        training_args=training_args,
        lora_args=lora_args, )


    trainer = ILQLTrainer(
        callbacks=[ checkpoint_callback],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,
        checkpoint_dir=data_args.output_dir,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        #max_grad_norm=training_args.max_grad_norm,
        strategy=strategy,
        precision='32',# 可以自行尝试  "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"
    )


    pl_model = MyILQLTransformer(config=config,model_args=model_args,training_args=training_args,lora_args=lora_args,ilql_args=ilql_args,
                                 quantization_config=global_args["quantization_config"],
                                 load_in_8bit=global_args["load_in_8bit"],
                                 device_map={"": trainer.local_rank} if trainer.world_size > 1 else "auto",
                                 torch_dtype=torch.float16, )
    # 加载sft权重训练
    # pl_model.load_sft_weight('sft_weight.bin',is_trainable=True)

    pl_model.float()

    train_datasets = dataHelper.load_distributed_random_sampler(
        dataHelper.train_files,
        with_load_memory=True,
        collate_fn=dataHelper.collate_fn,
        batch_size=training_args.train_batch_size,
        num_workers=0,  # num_workers for DataLoader
        drop_last=True,  # 多卡建议扔掉
        num_processes=trainer.world_size, process_index=trainer.global_rank)


    trainer.fit(pl_model,
                train_loader=train_datasets,
                tokenizer=tokenizer,
                ilql_config=ilql_args,
                stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
                )