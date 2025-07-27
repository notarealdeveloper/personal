#!/usr/bin/env python3

import os
import re
import sys
import glob
import argparse
import datetime
import requests
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_DROPOUT = 0.2


class Head(nn.Module):

    def __init__(self, embed_dim, head_size, context_size, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.WQ = nn.Linear(embed_dim, head_size, bias=False)
        self.WK = nn.Linear(embed_dim, head_size, bias=False)
        self.WV = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.WQ(x)  # (B,T,hs)
        k = self.WK(x)  # (B,T,hs)
        v = self.WV(x)  # (B,T,hs)

        w = q @ k.transpose(-2, -1) * k.shape[-1]**(-0.5) # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T), now causal
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        out = w @ v     # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, head_size, context_size, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_dim, head_size, context_size) for _ in range(num_heads)])
        self.project = nn.Linear(num_heads*head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outs = [head(x) for head in self.heads]
        out = torch.cat(outs, dim=-1)
        return self.dropout(self.project(out))


class FeedForward(nn.Module):

    def __init__(self, embed_dim, hidden_dim, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    """ Transformer Block: Communication followed by computation. """

    def __init__(self, embed_dim, num_heads, context_size, dropout=DEFAULT_DROPOUT):
        super().__init__()

        # parameters
        head_size = embed_dim // num_heads

        # network
        self.attn = MultiHeadAttention(embed_dim, num_heads, head_size, context_size, dropout=dropout)
        self.ffwd = FeedForward(embed_dim, hidden_dim=4*embed_dim, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, num_tokens, embed_dim=512, num_heads=8, context_size=256, num_layers=8, dropout=DEFAULT_DROPOUT):

        super().__init__()

        # network
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.position_embedding = nn.Embedding(context_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, context_size, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, num_tokens)

        # better init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensors of integers
        te = self.token_embedding(idx)
        pe = self.position_embedding(torch.arange(T, device=device)) # (T,C)
        x = te + pe  # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.linear(x)  # (B,T,num_tokens)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            if DATA_PARALLEL:
                loss = loss.view([1])

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is a (B,T) tensor of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits, loss = self(idx_cond) # logits is (B,T,C)
            logits = logits[:, -1, :] # becomes (B,C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def speak(self, text=None, max_new_tokens=500):
        if text is None:
            context = torch.zeros(1, dtype=torch.long, device=device)
        else:
            context = torch.tensor(encode(text), dtype=torch.long, device=device)
        return decode(self.generate(context.view(1, -1), max_new_tokens)[0].tolist())



DATASETS = {
    'shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    'bible': 'http://www.gutenberg.org/ebooks/10.txt.utf-8',
}

def ensure_have_dataset(dataset, output_path=None):
    url = DATASETS[dataset]
    if output_path is None:
        output_path = f"{dataset}.txt"
    if not os.path.exists(output_path):
        text = requests.get(url).content.decode()
        with open(output_path, 'w') as fp:
            fp.write(text)
        return text
    else:
        with open(output_path, 'r') as fp:
            return fp.read()


dataset = 'shakespeare' # options: 'shakespeare', 'bible'
text = ensure_have_dataset(dataset)
chars = sorted(set(text))
num_tokens = len(chars)

c2i = {c:i for i,c in enumerate(chars)}
i2c = {i:c for i,c in enumerate(chars)}

def encode(s):
    return [c2i[c] for c in s]

def decode(l):
    return ''.join([i2c[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
eval_data = data[n:]


# gpt parameters
embed_dim = 256
context_size = 256
batch_size = 64
num_heads = 4
num_layers = 4
learning_rate = 3e-4
max_iters = 5000
save_every = 1e6 # 1000
eval_iters = 5
eval_every = 1 # 50
speak_every = 2000
save_dir = 'gpt-checkpoints'
DATA_PARALLEL = False
load_latest = False

os.makedirs(save_dir, exist_ok=True)

@torch.no_grad()
def estimate_loss(eval_iters):
    losses = {}
    model.eval()
    for split in ('train', 'eval'):
        estimates = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            if DATA_PARALLEL:
                loss = loss.mean()
            estimates[iter] = loss.item()
        losses[split] = estimates.mean()
    model.train()
    return losses

def get_batch(split):
    if split not in ('train', 'eval'):
        raise ValueError(f"split must be one of: 'train', 'eval'")
    data = train_data if split == 'train' else eval_data
    idxs = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in idxs])
    y = torch.stack([data[i+1:i+context_size+1] for i in idxs])
    x,y = x.to(device), y.to(device)
    return x, y

# model
saved_models = sorted(glob.glob(f"{save_dir}/*.pth"))
if len(saved_models) > 0 and load_latest:
    latest_model = saved_models[-1]
    model = torch.load(latest_model)
    match = re.search('iter-(?P<num>\d+)[.]pth', os.path.basename(latest_model))
    init_iter = int(match.groupdict()['num']) + 1
else:
    model = GPT(
        num_tokens = num_tokens,
        embed_dim = embed_dim,
        num_heads = num_heads,
        context_size = context_size,
        num_layers = num_layers,
    )
    init_iter = 0

#if device == 'cpu':
#    model = torch.compile(model)

if DATA_PARALLEL:
    model = nn.DataParallel(model)
    module = model.module # for calling methods on the wrapped GPT instance
else:
    module = model
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'million parameters')

# training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start = datetime.datetime.now()

#print(model)

for iter in range(init_iter, max_iters):

    try:
        if (iter % eval_every == 0) or (iter == max_iters - 1):
            losses = estimate_loss(eval_iters=eval_iters)
            now = datetime.datetime.now()
            duration = now - start
            minutes = round(int(duration.total_seconds())/60, 1)
            print(f"{now} ({minutes} mins in): step {iter}: train loss {losses['train']:.4f}, val loss {losses['eval']:.4f}")

        if (iter) and (iter % speak_every == 0) or (iter == max_iters - 1):
            print(module.speak(max_new_tokens=500))
            print('='*100)

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        if DATA_PARALLEL:
            loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter > 0 and iter % save_every == 0) or (iter == max_iters - 1):
            output_path = f"{save_dir}/iter-{iter:05d}.pth"
            print(f"Saving model to {output_path}.")
            torch.save(model, output_path)
    except KeyboardInterrupt:
        output_path = f"{save_dir}/iter-{iter:05d}.pth"
        print(f"User requested exit. Saving model to {output_path}.")
        torch.save(model, output_path)
        break
