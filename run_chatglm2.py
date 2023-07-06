# -*- coding: utf-8 -*-
import logging
import os
from typing import Union

import torch
from fsspec.core import url_to_fs
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training, TaskType
from peft.tuners.lora import LoraLayer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding, \
    get_constant_schedule_with_warmup

from utils import load_jsonl, prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CPU_DEVICES = os.cpu_count() // 2


class HfModelCheckpoint(ModelCheckpoint):
    """ https://github.com/Lightning-AI/lightning/issues/3096#issuecomment-1441278197 """

    def _save_checkpoint(self, trainer: 'pl.Trainer', filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        hf_save_dir = filepath + '_hf'
        if trainer.is_global_zero:
            trainer.lightning_module.model.save_pretrained(hf_save_dir)
            trainer.lightning_module.tokenizer.save_pretrained(hf_save_dir)

    # https://github.com/Lightning-AI/lightning/pull/16067
    def _remove_checkpoint(self, trainer: 'pl.Trainer', filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        hf_save_dir = filepath + '_hf'
        if trainer.is_global_zero:
            fs, _ = url_to_fs(hf_save_dir)
            if fs.exists(hf_save_dir):
                fs.rm(hf_save_dir, recursive=True)


class LitCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments('model.pretrained_model_name_or_path', 'data.pretrained_model_name_or_path')

        parser.add_argument('--init_from', default='')

        parser.add_lightning_class_args(HfModelCheckpoint, 'model_checkpoint')
        parser.set_defaults(
            {
                'model_checkpoint.filename': '{epoch:02d}-{step:06d}-{val_loss:.4f}-{train_loss:.4f}',
                'model_checkpoint.monitor': 'val_loss',
                'model_checkpoint.mode': 'min',
                'model_checkpoint.save_top_k': 5,
            }
        )

    def fit(self, model, **kwargs):
        if self.config.fit.init_from:
            checkpoint = torch.load(self.config.fit.init_from)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info(f"Load weight from {self.config.fit.init_from}")

        self.trainer.fit(model, **kwargs)


class Seq2SeqDataset(Dataset):

    def __init__(self, data, tokenizer, max_source_length=512, max_target_length=512):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data[index]

        source_ids = self.tokenizer(
            (
                prompt['prompt_input'].format(instruction=record['instruction'], input=record['input'])
                if record['input']
                else prompt['prompt_no_input'].format(instruction=record['instruction'])
            ),
            add_special_tokens=False,
        ).input_ids[:self.max_source_length - 2]

        target_ids = None
        if 'output' in record:
            target_ids = self.tokenizer(
                record['output'],
                add_special_tokens=False  # ignore all special tokens
            ).input_ids[:self.max_target_length - 1]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)

        if target_ids:
            length = len(source_ids) + 2
            labels = [-100] * length + input_ids[length:]
            return {'input_ids': input_ids, 'labels': labels, **record}
        else:
            return {'input_ids': input_ids, **record}


class LitDataCollator(DataCollatorWithPadding):
    r"""
    Data collator for ChatGLM. It is capable of dynamically padding for batched data.
    """
    HF_INPUTS_KEY = ['input_ids', 'attention_mask', 'labels', 'decoder_input_ids']

    def __init__(
            self,
            tokenizer,
            config,
            label_pad_token_id=-100
    ):
        super().__init__(tokenizer, padding=True)
        self.config = config
        self.label_pad_token_id = label_pad_token_id

    def get_attention_masks(self, input_ids, device):
        batch_size, seq_length = input_ids.size()
        attention_mask = torch.ones((batch_size, seq_length), device=device)
        for i, seq in enumerate(input_ids):
            attention_mask[i, :(seq == self.tokenizer.get_command('sop')).nonzero()[0].item()] = 0  # context
        return attention_mask

    def get_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)[None, ...]
        position_ids = position_ids.expand(batch_size, -1)
        return position_ids

    def __call__(self, features, return_tensor=None):
        r"""
        Pads batched data to the longest sequence in the batch.
        """
        hf_features, examples = [], []
        for feature in features:
            hf_features.append({k: v for k, v in feature.items() if k in self.HF_INPUTS_KEY})
            examples.append({k: v for k, v in feature.items() if k not in self.HF_INPUTS_KEY})

        input_ids = [torch.tensor(feature['input_ids']) for feature in hf_features]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        hf_features_batch = {
            'input_ids': input_ids,
            'attention_mask': self.get_attention_masks(input_ids, device=input_ids.device),
            'position_ids': self.get_position_ids(input_ids, device=input_ids.device)
        }
        if 'labels' in hf_features[0]:
            labels = [torch.tensor(feature['labels']) for feature in hf_features]
            hf_features_batch['labels'] = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=self.label_pad_token_id
            )

        return {'hf_inputs': hf_features_batch, 'examples': examples}


class LitDataModule(LightningDataModule):

    def __init__(
            self,
            pretrained_model_name_or_path: str,

            train_data_path: str = None,
            val_data_path: str = None,
            test_data_path: str = None,

            num_workers: int = NUM_CPU_DEVICES,
            train_batch_size: int = 8,
            eval_batch_size: int = 8,

            max_source_length: int = 512,
            max_target_length: int = 512,

            label_pad_token_id: int = -100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        self.tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True

        self.data = {'train': [], 'val': [], 'test': []}

        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        self.data_collator = LitDataCollator(
            tokenizer=self.tokenizer,
            config=config,
            label_pad_token_id=label_pad_token_id,
        )

    def prepare_data(self):

        logger.info('Preparing Data ...')

        if self.hparams.train_data_path:
            logger.info(f"Loading Train Data from {self.hparams.train_data_path}")
            for record in load_jsonl(self.hparams.train_data_path):
                self.data['train'].append(record)

        if self.hparams.val_data_path:
            logger.info(f"Loading Val Data from {self.hparams.val_data_path}")
            for record in load_jsonl(self.hparams.val_data_path):
                self.data['val'].append(record)

        if self.hparams.test_data_path:
            logger.info(f"Loading Test Data from {self.hparams.test_data_path}")
            for record in load_jsonl(self.hparams.test_data_path):
                self.data['test'].append(record)

        logger.info({k: len(v) for k, v in self.data.items()})

    def setup(self, stage):

        if stage == 'fit':
            self.train_dataset = Seq2SeqDataset(
                self.data['train'],
                self.tokenizer,
                self.hparams.max_source_length,
                self.hparams.max_target_length
            )

        if stage in {'fit', 'validate'}:
            self.val_dataset = Seq2SeqDataset(
                self.data['val'],
                self.tokenizer,
                self.hparams.max_source_length,
                self.hparams.max_target_length
            )

        if stage == 'test':
            self.test_dataset = Seq2SeqDataset(
                self.data['test'],
                self.tokenizer,
                self.hparams.max_source_length,
                self.hparams.max_target_length
            )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):

        for k in batch['hf_inputs']:
            batch['hf_inputs'][k] = batch['hf_inputs'][k].to(device)

        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.hparams.num_workers)


class LitModule(LightningModule):

    def __init__(
            self,
            pretrained_model_name_or_path: str,
            lora_model_name_or_path: str = None,

            lora_rank: int = 8,
            lora_alpha: int = 32,
            lora_dropout: float = 0.1,
            lora_target_modules=None,

            learning_rate: float = 2e-5,
            warmup_ratio: float = 0.02,

            bits: Union[int] = 4,
            use_gradient_checkpointing: bool = True,

    ):
        super().__init__()
        self.save_hyperparameters()

        # for ChatGLM
        if lora_target_modules is None:
            lora_target_modules = ['query_key_value']

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True
        )

        # Prepare model
        model_args = {}
        if bits in [4, 8]:
            model_args['load_in_4bit'] = bits == 4
            model_args['load_in_8bit'] = bits == 8
            model_args['torch_dtype'] = torch.bfloat16
            model_args['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            **model_args
        )

        if bits in [4, 8]:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Setup peft
        if lora_model_name_or_path:
            logger.info(f"Load LoRA from {lora_model_name_or_path}")
            model = PeftModel.from_pretrained(
                model,
                lora_model_name_or_path,
                is_trainable=True
            )

        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                target_modules=lora_target_modules,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            model = get_peft_model(model, peft_config)

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if module.weight.dtype == torch.float32:
                        module.to(torch.bfloat16)

        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        if torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        # `use_cache=True` is incompatible with gradient checkpointing
        if use_gradient_checkpointing:
            model.config.use_cache = False

        model.print_trainable_parameters()

        self.model = model
        self.tokenizer = tokenizer
        self.outputs = []

    def state_dict(self, *args, **kwargs):

        state_dict = super().state_dict(*args, **kwargs)

        # only save the trainable parameters
        filtered_state_dict = {}
        for k, v in self.named_parameters():
            if v.requires_grad:
                filtered_state_dict[k] = state_dict[k].cpu().clone().detach()

        return filtered_state_dict

    def forward(self, *args, **kwargs):
        return self.model(
            *args, **kwargs
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.learning_rate
        )

        if self.hparams.warmup_ratio:
            num_warmup_steps = int(self.hparams.warmup_ratio * len(self.trainer.datamodule.train_dataloader()))
            assert num_warmup_steps != 0
        else:
            num_warmup_steps = 0

        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps),

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
            },
        }

    def training_step(self, batch, batch_idx):
        output = self(**batch['hf_inputs'])
        self.log_dict({'loss': output.loss.item()}, prog_bar=True)
        self.outputs.append(output.loss.item())
        return output.loss

    def on_validation_epoch_start(self):

        loss = sum(self.outputs) / (len(self.outputs) + 1e-6)
        self.outputs.clear()
        self.log_dict({'train_loss': loss}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        output = self(**batch['hf_inputs'])
        self.outputs.append(output.loss.item())

    def on_validation_epoch_end(self):

        loss = sum(self.outputs) / (len(self.outputs) + 1e-6)
        self.outputs.clear()
        self.log_dict({'val_loss': loss}, prog_bar=True)


def main():
    _ = LitCLI(LitModule, LitDataModule, seed_everything_default=42)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main()
