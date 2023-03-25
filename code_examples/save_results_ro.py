import pickle
import pandas as pd

df = pd.read_json("test_wor.json")

with open('ratings.pickle', 'rb') as f:
    ratings = pickle.load(f)


for i in range(len(ratings)):
    ratings[i] += 1

df["pred_rating"] = ratings
i = 0

df = df.drop(columns=["rating"])
df = df.rename(columns={"pred_rating": "rating"})

df.to_json("test.json", orient="records")