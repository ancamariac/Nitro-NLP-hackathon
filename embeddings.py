import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# BERT model transforma propozitiile intr un vector de numere

df = pd.read_csv("data//train_data.csv", encoding="ISO-8859-1")

print(df.head(5))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = TFBertModel.from_pretrained("bert-base-uncased")

# liste pentru noul dataset (x_data-embeddings, y_data-labels)
x_data = []
y_data = []
i = 0
for index, row in df.iterrows():
    # transform labelul intr-un numar
    if row[0] == 'neutral':
        y_data.append(1)
    elif row[0] == 'positive':
        y_data.append(2)
    elif row[0] == 'negative':
        y_data.append(0)
    else:
        continue

    # apelez modelul BERT pentru fiecare propozitie
    input_ids = tf.constant(tokenizer.encode(row[1], max_length=512, truncation=True))[None, :]
    outputs = model(input_ids)
    if i % 10 == 0:
        print(i)
    i += 1

    # folosesc doar ultimul strat returnat de modelul BERT
    x_data.append(np.array(outputs["pooler_output"]).reshape(768))
    if i > 4840:
        break

# salvam setul de date pe care il vom folosi la antrenarea modelului nostru
import pickle

with open('embeddings.pk', 'wb') as f:
    pickle.dump([x_data, y_data], f)



"""
model:
    strat dense input, activare ReLU
    strat overfitting
    straturi dense hidden (testam sa vedem cate facem si cat de mari sa fie fiecare), activare ReLU
    strat output, 3 neuroni, activare softmax

"""