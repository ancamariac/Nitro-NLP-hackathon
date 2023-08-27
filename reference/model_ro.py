import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import  KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score, accuracy_score

with open('trainembeddings.pickle', 'rb') as f:
    X, Y = pickle.load(f)

def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")

def base_model():
    model = Sequential()
    # add the layers
    model.add(Dense(200, input_shape=(768, ), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # compile, train and evaluate
    model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer='adam')
    return model

num_folds = 10
kfold = KFold(n_splits=num_folds)

w_f1 = []

estimator = KerasClassifier(build_fn=base_model, epochs=200, batch_size=512)
"""
for train, test in kfold.split(X, Y):
    estimator.fit(X[train], Y[train])
    Y_pred = estimator.predict(X[test])

    print(accuracy_score(Y[test], Y_pred))
    w_f1.append(weighted_f1(Y[test], Y_pred))

print(w_f1)
"""
estimator.fit(X, Y)

estimator.model.save("model.h5")