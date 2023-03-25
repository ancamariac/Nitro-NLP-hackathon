import pandas as pd
import numpy as np
import tensorflow as tf
import re
import pickle
from transformers import AutoTokenizer, TFAutoModel

df = pd.read_csv("data//train_data.csv", encoding="utf-8")

# https://huggingface.co/readerbench/RoBERT-large
tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-large")
model = TFAutoModel.from_pretrained("readerbench/RoBERT-large")

# x_data - embeddings, y_data - labels, id_data - ids
x_data = []
y_data = []
id_data = []
i = 0

for index, row in df.iterrows():
    # transform labelul intr-un numar
    if row[1] == 'offensive':
        y_data.append(0)
    elif row[1] == 'non-offensive':
        y_data.append(1)
    elif row[1] == 'direct':
        y_data.append(2)
    elif row[1] == 'descriptive':
        y_data.append(3)
    elif row[1] == 'reporting':
        y_data.append(4)
    else:
        continue
    id_data.append(row[2])

    # replaced twitter accounts with USERNAME
    clean = re.sub(r"\B@\w+", "USERNAME", row[0])
    
    # replaced emails with EMAIL
    clean = re.sub('([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})', "EMAIL", clean)
    
    # removed emojis
    # https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    clean = emoji_pattern.sub(r'', clean)
    
    input_ids = tf.constant(tokenizer.encode(clean, max_length=512, truncation=True))[None, :]
    outputs = model(input_ids)
    
    if i % 50 == 0:
        print(i)
    i += 1

    # using the last layer of RoBERT
    # default reshaping to 1024
    x_data.append(np.array(outputs["pooler_output"]).reshape(1024))
    
    if i > 39007:
        break

# saving the pickle file 
with open('embeddings.pk', 'wb') as f:
    pickle.dump([x_data, y_data, id_data], f)

