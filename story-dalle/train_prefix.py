# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""

import logging
import os, torch
from dalle.models import PrefixTuningDalle
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only
import pororo_dataloader as data
from datetime import datetime
import argparse

console_logger = logging.getLogger(__name__)

class ImageLogger(Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def log_img(self, pl_module, batch, current_epoch, split="train"):
        with torch.no_grad():
            images, labels = batch
            recons = pl_module.stage1(images)
            images = images.cpu()
            recons = recons.cpu()

            grid_org = (torchvision.utils.make_grid(images, nrow=8) + 1.0) / 2.0
            grid_rec = (torchvision.utils.make_grid(recons, nrow=8) + 1.0) / 2.0
            grid_rec = torch.clip(grid_rec, min=0, max=1)

            pl_module.logger.experiment.add_image(f"images_org/{split}", grid_org, global_step=current_epoch)
            pl_module.logger.experiment.add_image(f"images_rec/{split}", grid_rec, global_step=current_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and trainer.current_epoch < 5:
            self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and trainer.current_epoch < 5:
            self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="test")


def get_dataset(args, tokenizer, preprocess: transforms, mode = 'train'):
    dataset = data.ImageDataset(args.data_dir, tokenizer, preprocess, mode=mode)
    return dataset


def setup_callbacks(config, output_dir):
    # Setup callbacks
    now = datetime.now().strftime('%d%m%Y_%H%M%S')
    result_path = os.path.join(output_dir, now)
    ckpt_path = os.path.join(result_path, 'ckpt')
    log_path = os.path.join(result_path, 'log')

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="imagenet-clscond-gen-{epoch:02d}" if config.stage2.use_cls_cond else
                 "imagenet-uncond-gen-{epoch:02d}",
        every_n_epochs=config.experiment.save_ckpt_freq,
        save_weights_only=True,
        save_last=True
    )
    tb_logger = TensorBoardLogger(log_path, name="prefixDALLE")
    logger_img = ImageLogger()
    return checkpoint_callback, tb_logger, logger_img


def main(args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Set seed
    pl.seed_everything(args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # Initiate config and tokenizer
    model, config = PrefixTuningDalle.from_pretrained(args)

    for param in model.stage1.parameters():
        param.requires_grad = False
    for param in model.stage2.parameters():
        param.requires_grad = False

    # print(dir(model))

    # HERE
    # model.resize_token_embeddings(len(tokenizer))

    # if data_args.block_size <= 0:
    #     data_args.block_size = tokenizer.max_len
    #     # Our input block size will be the max possible for the model
    # else:
    #     data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # ADD SPECIAL TOKENS:
    # if (model_args.tuning_mode != 'prefixtune') and ('lowdata' not in training_args.output_dir) and (model_args.tuning_mode != 'adaptertune'):
    #     print(model_args.tuning_mode)
    #     print('adapting the size of the model embedding to include [PAD], [BOS], [EOS].')
    #     print('len(tokenizer) = ', len(tokenizer))
    #     num_added_tokens = tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token':'[BOS]', 'eos_token':'[EOS]'})
    #     embedding_layer = model.resize_token_embeddings(len(tokenizer))
    #     print('len(tokenizer) = ', len(tokenizer))
    # elif data_args.dataless == 'yes':
    #     print(model_args.tuning_mode, 'dataless setting, so no new tokens at all.')
    #     print('We do not add special tokens to the tokenizer, instead, we just finetune on <|endoftext|>')
    #
    #     print(tokenizer.eos_token_id)
    #     print(tokenizer.eos_token)
    #     print(tokenizer.pad_token_id)
    #     tokenizer.pad_token = tokenizer.eos_token
    #     # tokenizer(['he', 'hello w '], padding=True)
    #
    #     # tokenizer.pad_token_id = tokenizer.eos_token_id
    #     # tokenizer.pad_token = tokenizer.eos_token
    #     print(tokenizer.pad_token, tokenizer.pad_token_id)

    ##############################################################
    ################# ADJUST TOKENIZER ###########################
    ##############################################################

    # print(model_args.tuning_mode)
    # print('adapting the size of the model embedding to include [PAD]')
    # print('len(tokenizer) = ', len(tokenizer))
    # num_added_tokens = tokenizer.add_special_tokens(
    #     {'pad_token': '[PAD]'})
    # embedding_layer = model.resize_token_embeddings(len(tokenizer))
    # print('len(tokenizer) = ', len(tokenizer))
    # print(tokenizer.eos_token, tokenizer.eos_token_id)
    # print(tokenizer.bos_token, tokenizer.bos_token_id)


    ##############################################################
    #################LOADING DATASETS ###########################
    ##############################################################

    train_transform = transforms.Compose(
        [transforms.Resize(config.dataset.image_resolution),
         transforms.RandomCrop(config.dataset.image_resolution),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    valid_transform = transforms.Compose(
        [transforms.Resize(config.dataset.image_resolution),
         transforms.CenterCrop(config.dataset.image_resolution),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    train_dataset = (
        get_dataset(args, model.tokenizer, train_transform, mode='train')  # if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(args, model.tokenizer, valid_transform, mode='val')
    )

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_gpu_train_batch_size * args.n_gpu,
        drop_last=True,
        shuffle=True,
        num_workers=int(args.dataloader_num_workers))

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size * args.n_gpu,
        drop_last=True,
        shuffle=False,
        num_workers=int(args.dataloader_num_workers))

    # Setup callbacks
    ckpt_callback, logger, logger_img = setup_callbacks(config, args.output_dir)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    console_logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    args.n_gpus = torch.cuda.device_count()
    # Calculate how many batches are accumulated
    assert config.experiment.total_batch_size % (config.experiment.local_batch_size * args.n_gpus) == 0
    grad_accm_steps = config.experiment.total_batch_size // (config.experiment.local_batch_size * args.n_gpus)
    config.optimizer.max_steps = len(train_dataset) // config.experiment.total_batch_size * config.experiment.epochs

    # Build trainer
    trainer = pl.Trainer(max_epochs=config.experiment.epochs,
                         accumulate_grad_batches=grad_accm_steps,
                         gradient_clip_val=config.optimizer.grad_clip_norm,
                         precision=16 if config.experiment.use_amp else 32,
                         callbacks=[ckpt_callback, logger_img],
                         accelerator="gpu",
                         devices=args.n_gpus,
                         strategy="ddp",
                         logger=logger)

    # Training & Evaluation
    trainer.fit(model, train_loader, eval_loader)
    return


# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arguments for training/evaluating prefix-tuning DALLE')

    # Model Arguments
    parser.add_argument('--model_name_or_path', type=str, default=None, help='The model checkpoint for weights initialization.')
    parser.add_argument('--prefix_model_name_or_path', type=str, default=None, help='The prefix model checkpoint for weights initialization.')
    parser.add_argument('--prefix_mode', type=str, default='activation', help='activation or embedding')
    parser.add_argument('--preseqlen', type=int, default=0, help='how many tokens of prefix should we include.')
    parser.add_argument('--optim_prefix', action="store_true", help='set to True if optimizing prefix directly; no if through amortized function')
    parser.add_argument('--tuning_mode', type=str, default='prefixtune', help='prefixtune or finetune')
    parser.add_argument('--top_k_layers', type=int, default=2, help='In finetuning setting, if we only tune the top k layers.')
    parser.add_argument('--parameterize_mode', type=str, default='mlp', help="mlp or emb to parametrize when we optimize for the embeddings.")
    parser.add_argument('--prefix_dropout', type=float, default=0.0, help='dropout rate for the prefix tuning model.')
    parser.add_argument('--teacher_dropout', type=float, default=0.0, help='dropout rate for the teacher model.')
    parser.add_argument('--init_random', action="store_true", help="set True if initializing random embeddings")
    parser.add_argument('--init_shallow', action="store_true", help="set True if not using reparameterization")
    parser.add_argument('--init_shallow_word', type=bool, default=False, help="set True if init_shallow and specify words")
    parser.add_argument('--replay_buffer', action="store_true", help="set True if using replay buffer in training")
    parser.add_argument('--gumbel', action="store_true", help="set True if using the gumbel softmax in training")
    parser.add_argument('--hidden_dim_prefix', type=float, default=512, help="hidden dim of MLP for generating prefix?")

    # Data Arguments
    parser.add_argument('--data_dir', type=str, default=None, help="Path to data directory")
    parser.add_argument('--lowdata_token', type=str, default='story', help="The token to be prepended at initialization time.")
    parser.add_argument('--use_lowdata_token', type=bool, default=True, help="Whether we should use the lowdata token for prefix-tuning")
    parser.add_argument('--train_embeddings', action="store_true", help="Whether to train word embeddings")
    parser.add_argument('--train_max_target_length', type=int, default=100, help='the max target length for training data.')
    parser.add_argument('--val_max_target_length', type=int, default=100, help='the max target length for dev data.')
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='number of workers when loading data')


    # Training Arguments
    parser.add_argument('--output_dir', type=str, default=None, help="Path to data directory")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test.")
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    args = parser.parse_args()

    main(args)