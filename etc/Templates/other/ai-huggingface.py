#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import TextDataset
from transformers import DataCollator, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

model_name = 'gpt2'
tokenizer_dir = 'tok'
file_path = 'input.txt'
block_size = 256

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model     = GPT2LMHeadModel.from_pretrained(model_name)

os.makedirs(tokenizer_dir, exist_ok=True)
dataset = TextDataset(
    tokenizer,
    file_path,
    block_size=block_size,
    overwrite_cache=False,
    cache_dir=tokenizer_dir,
)

# dataset affordances
# ===================

## whole file text datasets
assert len(dataset.examples) == 4723
assert len(dataset) == 4723
assert len(dataset.examples[0]) == block_size
assert len(dataset[0]) == block_size # this is a torch tensor
tokenizer.decode(dataset.examples[0])

text = open(file_path).read()
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)
not_special = ids[:10]
yes_special = tokenizer.build_inputs_with_special_tokens(not_special)

## line by line text datasets
from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(tokenizer, file_path, block_size=256)

## custom text datasets
from torch.utils.data import Dataset

class LearnableMethodsDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        raw_examples = open(file_path).read().split('\n\n')
        examples = []
        for raw_example in raw_examples:
            examples.append(tokenizer.encode(raw_example))
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

dataset = LearnableMethodsDataset(tokenizer, file_path='raw.txt')

# datacollator affordances
# ========================
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
#tokenizer.pad_token = '[PAD]'
#collated = datacollator([dataset[k] for k in range(10)])
#input_ids = collated['input_ids']
#labels = collated['labels']

# training argument affordances
# =============================
training_args = TrainingArguments(
    output_dir='output',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=5,
    optim='adamw_torch',
    report_to='none',
    xpu_backend='ccl',
)
trainer = Trainer(
    model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    #model_init,
    #compute_metrics,
    #callbacks,
    #preprocess_logits_for_metrics,
)

trainer.train()
