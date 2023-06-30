# -*- coding: utf-8 -*-
import json

import rouge_chinese

rouge_metric = rouge_chinese.Rouge()

prompt = {
    "prompt_input": "以下是一条描述任务的指令。请根据问题，编写一个回答，以适当地完成指令。\n\n### 指令:\n{instruction}\n\n### 问题:\n{input}\n\n### 回答:\n",
    "prompt_no_input": "以下是一条描述任务的指令。编写一个回答，以适当地完成指令。\n\n### 指令:\n{instruction}\n\n### 回答:\n",
    "prompt_no_description": "\n\n### 指令:\n{instruction}\n\n### 问题:\n{input}\n\n### 回答:\n",
    'prompt_split': '### 回答:\n'
}


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def dump_jsonl(objs, path, mode='w'):
    with open(path, mode=mode, encoding='utf-8') as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
