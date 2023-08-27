import pandas as pd
import numpy as np
import keras
import pickle
import os

from sklearn.metrics import balanced_accuracy_score
def balanced_accuracy(y_true, y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    y_true_np = np.argmax(y_true_np, axis=1)
    y_pred_np = np.argmax(y_pred_np, axis=1)
    return balanced_accuracy_score(y_true_np, y_pred_np)

df = pd.read_csv("data//test_data.csv", encoding="utf-8")

path = os.getcwd()
emb_path = os.path.join(path, "embeddings_test.pk")

with open(emb_path, 'rb') as f:
    X, IDS = pickle.load(f)

X = np.array(X, dtype='f')

predict_model = keras.models.load_model("model.h5", custom_objects={"balanced_accuracy": balanced_accuracy})

y_pred = predict_model.predict(X)
y = np.argmax(y_pred, axis=1)

y_text = []
index_to_label = {
    0: "offensive",
    1: "non-offensive",
    2: "direct",
    3: "descriptive",
    4: "reporting"
}
for elem in y:
    y_text.append(index_to_label[elem])

df['Label'] = y_text
df = df.drop(columns=['Text'])
df.to_csv('test_data.csv',index=False)
print(df.head(5))