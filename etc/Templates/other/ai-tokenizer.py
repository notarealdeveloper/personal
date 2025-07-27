#!/usr/bin/env python3

vocab_size = 50_000
files = [] # put files here

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, ByteLevel
from transformers import PreTrainedTokenizerFast

fast_tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
fast_tokenizer.pre_tokenizer = Whitespace()
fast_tokenizer.pre_tokenizer = Punctuation()
fast_tokenizer.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=100)
fast_tokenizer.train(files, trainer)

tokenizer = PreTrainedTokenizerFast(tokenizer_object=fast_tokenizer)


#sorted(fast_tokenizer.get_vocab(), key=len)[-100:]

sorted(tokenizer.vocab, key=len)[-1000:]

