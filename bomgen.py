#!/usr/bin/env python3
# pip3 install tokenizer tensorflow==1.15 gpt-2-simple requests

import gpt_2_simple as gpt2
import os
import requests
import random
import sys

model_name = "355M"
if not os.path.isdir(os.path.join("models", model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/355M/


file_name = "bom.txt"
if not os.path.isfile(file_name):
    url = "https://www.gutenberg.org/cache/epub/17/pg17.txt"
    data = requests.get(url)
    with open(file_name, 'w') as f:
        f.write(data.text)

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,
              file_name,
              model_name=model_name,
              steps=1000)   # steps is max number of training steps
#gpt2.load_gpt2(sess)

seed = random.randint(0, 2**32 - 1)
out_file = f"bom.{seed}.txt"
prefix = ''
max_tokens = 500000
temp = 0.8
with open(out_file, 'w') as f:
    for step in range(int(max_tokens / 512)):
        text = gpt2.generate(sess, prefix=prefix, include_prefix=False, seed=seed, return_as_list=True, temperature=temp)
        print(''.join(text), end='', flush=True)
        print(''.join(text), end='', file=f, flush=True)
        if len(text) > 1:
            prefix = ''.join(text[int(len(text)/2):])
        else:
            text = ''.join(text)
            prefix = text[int(len(text)/2):]
