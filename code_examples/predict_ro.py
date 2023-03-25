import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import keras
import pickle

df = pd.read_json("test_wor.json")
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

x_bert = []
i = 0
for index, row in df.iterrows():
    input_ids = tokenizer.encode(row["text"], add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(input_ids)
    pooler_output = outputs["pooler_output"]
    x_bert.append(pooler_output.detach().numpy().reshape(768))
    if i % 50 == 0:
        print(i)
    i += 1

x_bert = np.array(x_bert)

predict_model = keras.models.load_model("model.h5")

y_pred = predict_model.predict(x_bert)
y = np.argmax(y_pred, axis=1)

with open('ratings.pickle', 'wb') as f:
    pickle.dump(y, f)
