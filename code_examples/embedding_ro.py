import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle


df = pd.read_json("train.json")
print(df.head())

tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

x_bert = []
y_bert = []

i = 0
for index, row in df.iterrows():
    input_ids = tokenizer.encode(row["text"], add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(input_ids)
    pooler_output = outputs["pooler_output"]
    x_bert.append(pooler_output.detach().numpy().reshape(768))
    y_bert.append(row["rating"])
    if i % 50 == 0:
        print(i)
    i += 1

x_bert = np.array(x_bert)
y_bert = np.array(y_bert)

with open('trainembeddings.pickle', 'wb') as f:
    pickle.dump([x_bert, y_bert], f)
