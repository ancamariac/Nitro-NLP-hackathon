import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import itertools

# paths to roberta embeddings of the dataset
path = os.getcwd()
emb_path = os.path.join(path, "embeddings.pk")


# load dataset
with open(emb_path, 'rb') as f:
    X, Y_aux, IDS = pickle.load(f)

Y = []
cnt_Y = [0,0,0,0,0]

cnt_non_off = 0
for elem in Y_aux:
    aux = [0.0,0.0,0.0,0.0,0.0]
    aux[elem] = 1.0
    cnt_Y[elem] += 1 
    Y.append(aux)

class_weights = {}
for i in range(5):
    class_weights[i] = 39007 / cnt_Y[i]

X = np.array(X, dtype='f')
Y = np.array(Y, dtype='f')

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.8, random_state=33)

checkpoint_filepath = os.path.join(os.getcwd(), 'tmp', 'checkpoint')

# grid search dict
dct_grid_space = {
    'layer1' : [
        1024,
        512
    ],
    'layer2' : [
        1024,
        512
    ],
    'layer3' : [
        1024,
        512
    ],
    'dropout1' : [
        0.1,
        0.3
    ],
    'dropout2' : [
        0.1,
        0.3
    ],
    'opt_class' : [
        'sgd',
        'adam'
    ],
    'lr' : [
        0.001,
        0.01,
    ]
}


# initialize the model
def create_model(l1, l2, l3, d1, d2, opt):
    model = Sequential()
    model.add(l1)
    model.add(d1)
    model.add(l2)
    model.add(d2)
    model.add(l3)
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model

# generate all combinations for grid search
grid_params = []
grid_values = []
for k in dct_grid_space:
    grid_params.append(k)
    grid_values.append(dct_grid_space[k])
grid_combs = list(itertools.product(*grid_values))

best_comb = grid_combs[0]
best_acc = 0
counter = 1

# try all combinations and save the one with the best accuracy
for combination in grid_combs:
    print("Combination ", counter)
    print(combination)
    counter += 1

    # callbacks to save the best epoch
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # combination = [l1_num, l2_num, d1, act1, act2,  opt, lr]
    l1 = Dense(combination[0], input_shape=(1024, ), activation='relu')
    l2 = Dense(combination[1], activation='relu')
    l3 = Dense(combination[2], activation='relu')
    d1 = Dropout(combination[3])
    d2 = Dropout(combination[4])
    if combination[5] == 'adam':
        opt = keras.optimizers.Adam(learning_rate=combination[6])
    else:
        opt = keras.optimizers.SGD(learning_rate=combination[6])

    model = create_model(l1, l2, l3, d1, d2, opt)
    model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_val, Y_val), callbacks=[model_checkpoint_callback], class_weight=class_weights)
    model.load_weights(checkpoint_filepath)
    scores = model.evaluate(X_val, Y_val)

    acc = scores[2]
    print(f'{model.metrics_names[2]} of {acc*100}%;')
    if acc > best_acc:
        best_comb = combination
        best_acc = acc
    exit(0)

# create the model for the best combination
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

l1 = Dense(best_comb[0], input_shape=(1024, ), activation="relu")
l2 = Dense(best_comb[1], activation="relu")
l3 = Dense(best_comb[2], activation="relu")
d1 = Dropout(best_comb[3])
d2 = Dropout(best_comb[4])
if best_comb[5] == 'adam':
    opt = keras.optimizers.Adam(learning_rate=best_comb[6])
else:
    opt = keras.optimizers.SGD(learning_rate=best_comb[6])

model = create_model(l1, l2, l3, d1, d2, opt)
history = model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_val, Y_val), callbacks=[model_checkpoint_callback], class_weight=class_weights)
model.load_weights(checkpoint_filepath)
scores = model.evaluate(X_val, Y_val)


# metrics
print(f'Score:\n{model.metrics_names[0]} of {scores[0]};')
print(f'{model.metrics_names[1]} of {scores[1]*100}%;')

print("Model grid:", best_comb)

# save the model
model.save(os.path.join(path, 'model_bin.h5'))

# plots

# validation loss
plt.clf()
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (len(history_dict['loss']) + 1))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# validation accuracy
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, (len(history_dict['accuracy']) + 1))
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""
Score:
loss of 0.21578842401504517;
recall_432 of 91.01141691207886%;
accuracy of 91.13600254058838%;
precision_432 of 91.38809442520142%;

Best combination:
(256, 128, 0.1, 'relu', 'relu', 'adam', 0.001)

"""
