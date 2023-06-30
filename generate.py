# -*- coding: utf-8 -*-
import logging
import random

import fire
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import dump_jsonl, load_jsonl, prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = 2048


def escape_f_string(text):
    return text.replace('{', '{{').replace('}', '}}')


class RandomizedKShotSampler:

    def __init__(self, examples, tokenizer):

        for example in examples:
            example['instruction'] = escape_f_string(example['instruction'])
            example['input'] = escape_f_string(example['input'])

        self.examples = examples
        self.tokenizer = tokenizer

    def __call__(self, inputs, k=5):

        inputs['instruction'] = escape_f_string(inputs['instruction'])
        inputs['input'] = escape_f_string(inputs['input'])

        examples = random.sample(self.examples, k=k)

        text = ''
        last_text = None
        for i, example in enumerate(sorted(examples, key=lambda x: len(x['instruction'] + x['input'] + x['output']))):

            if i == 0:
                text += prompt['prompt_input'].format(
                    instruction=example['instruction'], input=example['input']
                )
            else:
                text += prompt['prompt_no_description'].format(
                    instruction=example['instruction'], input=example['input']
                )
            text += example['output']

            text_with_input = text + prompt['prompt_no_description'].format(
                instruction=inputs['instruction'], input=inputs['input']
            )
            if len(self.tokenizer(text_with_input).input_ids) > MAX_LENGTH:
                break

            last_text = text_with_input

        if not last_text:
            last_text = prompt['prompt_input'].format(
                instruction=inputs['instruction'], input=inputs['input']
            )

        return last_text


@torch.no_grad()
def main(
        llm_model_file: str = 'THUDM/chatglm2-6b',
        peft_model_file: str = '',
        inp_file: str = '',
        out_file: str = '',
        ref_file: str = '',
        k_shot: int = 0,
        device: str = 'cuda'
):
    if not inp_file:
        return

    logger.info(f"Load examples file from {inp_file}")
    examples = load_jsonl(inp_file)

    tokenizer = AutoTokenizer.from_pretrained(
        llm_model_file,
        trust_remote_code=True,
    )

    # Prepare model
    model_args = {'device_map': device, 'torch_dtype': torch.float if device == 'cpu' else torch.float16}
    model = AutoModel.from_pretrained(
        llm_model_file,
        trust_remote_code=True,
        **model_args
    )

    if peft_model_file:
        # Setup peft
        model = PeftModel.from_pretrained(
            model,
            peft_model_file,
        )

    model.eval()
    model.to(device)
    model = torch.compile(model)

    tokenizer_kwargs = {'return_tensors': 'pt', 'max_length': MAX_LENGTH, 'truncation': True}
    """
    greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False
    contrastive search by calling contrastive_search() if penalty_alpha>0. and top_k>1
    multinomial sampling by calling sample() if num_beams=1 and do_sample=True
    beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False
    beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True
    diverse beam-search decoding by calling group_beam_search(), if num_beams>1 and num_beam_groups>1
    constrained beam-search decoding by calling constrained_beam_search(), if constraints!=None or force_words_ids!=None
    """
    generate_kwargs = {
        'max_new_tokens': 512,
        'num_beams': 1,
        'temperature': 0.7,
        'top_k': 40,
        'top_p': 0.95,
        'num_return_sequences': 1
    }
    decode_kwargs = {'skip_special_tokens': False}

    ref_examples = load_jsonl(ref_file) if ref_file else []
    sampler = RandomizedKShotSampler(ref_examples, tokenizer)

    for example in tqdm(examples):

        text = sampler(example, k_shot)
        logger.info(f"Input : {text}")

        model_inputs = tokenizer(
            [text],
            **tokenizer_kwargs
        ).to(device)

        outputs = model.generate(**model_inputs, **generate_kwargs)
        outputs = tokenizer.batch_decode(outputs, **decode_kwargs)

        for i, output in enumerate(outputs):
            record = {
                'instruction': example['instruction'],
                'input': example['input'],
                'output': output.split(prompt['prompt_split'])[-1],
            }
            logger.info(f"{i} Output    : {record['output']}")
            dump_jsonl([record], out_file, 'a+')

    logger.info(f"Output file is saved to {out_file}")


if __name__ == '__main__':
    fire.Fire(main)
