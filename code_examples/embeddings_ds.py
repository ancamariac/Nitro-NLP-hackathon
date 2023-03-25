import numpy as np
import os
import tensorflow as tf
from transformers import RobertaModel, RobertaTokenizer
import re
import pickle
import torch

CLEANR = re.compile('<.*?>') 

# remove html tags
def clean_html(text):
  cleantext = re.sub(CLEANR, '', text)
  return cleantext

# dataset paths
main_path = os.getcwd()
train_path = os.path.join(main_path, "train")
pos_path = os.path.join(train_path, "pos")
neg_path = os.path.join(train_path, "neg")
test_path = os.path.join(main_path, "test")
test_pos_path = os.path.join(test_path, "pos")
test_neg_path = os.path.join(test_path, "neg")

# initiate a corpus for each label
corpus_pos = []
corpus_neg = []
corpus_pos_test = []
corpus_neg_test = []

print("Loading corpus...")

# fill the positive reviews corpus
for filename in os.listdir(pos_path):
    f = os.path.join(pos_path, filename)
    if os.path.isfile(f):
        with open(f, "r", encoding="utf-8") as fr:
            text = fr.read()
            corpus_pos.append(clean_html(text))

# fill the negative reviews corpus
for filename in os.listdir(neg_path):
    f = os.path.join(neg_path, filename)
    if os.path.isfile(f):
        with open(f, "r", encoding="utf-8") as fr:
            text = fr.read()
            corpus_neg.append(clean_html(text))

# fill the test corpus
for filename in os.listdir(test_pos_path):
    f = os.path.join(test_pos_path, filename)
    if os.path.isfile(f):
        with open(f, "r", encoding="utf-8") as fr:
            text = fr.read()
            corpus_pos_test.append(clean_html(text))

for filename in os.listdir(test_neg_path):
    f = os.path.join(test_neg_path, filename)
    if os.path.isfile(f):
        with open(f, "r", encoding="utf-8") as fr:
            text = fr.read()
            corpus_neg_test.append(clean_html(text))
print("Generating embeddings")

# load pretrained roberta
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

i = 0
emb_pos = []
emb_neg = []
emb_pos_test = []
emb_neg_test = []

# apply roberta on the corpus
for text in corpus_pos:
    input_ids = torch.tensor(tokenizer.encode(text, truncation=True)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    emb_np = last_hidden_states.detach().numpy()[0][0]
    emb_pos.append(emb_np)
    if i % 50 == 0:
        print(i)
    i += 1

for text in corpus_neg:
    input_ids = torch.tensor(tokenizer.encode(text, truncation=True)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    emb_np = last_hidden_states.detach().numpy()[0][0]
    emb_neg.append(emb_np)
    if i % 50 == 0:
        print(i)
    i += 1

for text in corpus_pos_test:
    input_ids = torch.tensor(tokenizer.encode(text, truncation=True)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    emb_np = last_hidden_states.detach().numpy()[0][0]
    emb_pos_test.append(emb_np)
    if i % 50 == 0:
        print(i)
    i += 1

for text in corpus_neg_test:
    input_ids = torch.tensor(tokenizer.encode(text, truncation=True)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    emb_np = last_hidden_states.detach().numpy()[0][0]
    emb_neg_test.append(emb_np)
    if i % 50 == 0:
        print(i)
    i += 1

# save the embeddings
with open(os.path.join(main_path,'embeddings_neg.pk'), 'wb') as f:
  pickle.dump(emb_neg, f)

with open(os.path.join(main_path,'embeddings_neg.pk'), 'wb') as f:
  pickle.dump(emb_neg, f)

with open(os.path.join(main_path,'embeddings_pos_test.pk'), 'wb') as f:
  pickle.dump(emb_pos_test, f)

with open(os.path.join(main_path,'embeddings_neg_test.pk'), 'wb') as f:
  pickle.dump(emb_neg_test, f)