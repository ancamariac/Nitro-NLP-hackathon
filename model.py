import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import itertools

# paths to roberta embeddings of the dataset
path = os.getcwd()
pos_path = os.path.join(path, "embeddings_pos.pk")
neg_path = os.path.join(path, "embeddings_neg.pk")
test_pos_path = os.path.join(path, "embeddings_pos_test.pk")
test_neg_path = os.path.join(path, "embeddings_neg_test.pk")

# load dataset
with open(pos_path, 'rb') as f:
    X_pos_train = pickle.load(f)
    Y_pos_train = np.array([1.0] * len(X_pos_train))
with open(neg_path, 'rb') as f:
    X_neg_train = pickle.load(f)
    Y_neg_train = np.array([0.0] * len(X_neg_train))
with open(test_pos_path, 'rb') as f:
    X_pos_aux = pickle.load(f)
    Y_pos_aux = np.array([1.0] * len(X_pos_aux))
with open(test_neg_path, 'rb') as f:
    X_neg_aux = pickle.load(f)
    Y_neg_aux = np.array([0.0] * len(X_neg_aux))

X_train = np.concatenate((X_pos_train, X_neg_train), axis=0)
Y_train = np.concatenate((Y_pos_train, Y_neg_train), axis=0)
X_aux = np.concatenate((X_pos_aux, X_neg_aux), axis=0)
Y_aux = np.concatenate((Y_pos_aux, Y_neg_aux), axis=0)

X_val, X_test, Y_val, Y_test = train_test_split(X_aux, Y_aux, test_size=0.5, random_state=42)

checkpoint_filepath = os.path.join(os.getcwd(), 'tmp', 'checkpoint')

# grid search dict
dct_grid_space = {
    'layer1' : [
        256,
        128,
        64
    ],
    'layer2' : [
        256,
        128,
        64
    ],
    'dropout1' : [
        0.1,
        0.2,
        0.3
    ],
    'activation1' : [
        'elu',
        'relu'
    ],
    'activation2' : [
        'relu',
        'elu'
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
def create_model(l1, l2, d1, opt):
    model = Sequential()
    model.add(l1)
    model.add(d1)
    model.add(l2)
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall(), 'accuracy', tf.keras.metrics.Precision()], optimizer=opt)
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
    l1 = Dense(combination[0], input_shape=(768, ), activation=combination[3])
    l2 = Dense(combination[1], activation=combination[4])
    d1 = Dropout(combination[2])
    if combination[5] == 'adam':
        opt = keras.optimizers.Adam(learning_rate=combination[6])
    else:
        opt = keras.optimizers.SGD(learning_rate=combination[6])

    model = create_model(l1, l2, d1, opt)
    model.fit(X_train, Y_train, batch_size=128, epochs=20, validation_data=(X_val, Y_val), callbacks=[model_checkpoint_callback], verbose=0)
    model.load_weights(checkpoint_filepath)
    scores = model.evaluate(X_test, Y_test)

    acc = scores[2]
    print(f'{model.metrics_names[2]} of {acc*100}%;')
    if acc > best_acc:
        best_comb = combination
        best_acc = acc

# create the model for the best combination
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

l1 = Dense(best_comb[0], input_shape=(768, ), activation=best_comb[3])
l2 = Dense(best_comb[1], activation=best_comb[4])
d1 = Dropout(best_comb[2])
if best_comb[5] == 'adam':
    opt = keras.optimizers.Adam(learning_rate=best_comb[6])
else:
    opt = keras.optimizers.SGD(learning_rate=best_comb[6])

model = create_model(l1, l2, d1, opt)
history = model.fit(X_train, Y_train, batch_size=128, epochs=20, validation_data=(X_val, Y_val), callbacks=[model_checkpoint_callback])
model.load_weights(checkpoint_filepath)
scores = model.evaluate(X_test, Y_test)


# metrics
scores = model.evaluate(X_test, Y_test)
print(f'Score:\n{model.metrics_names[0]} of {scores[0]};')
print(f'{model.metrics_names[1]} of {scores[1]*100}%;')
print(f'{model.metrics_names[2]} of {scores[2]*100}%;')
print(f'{model.metrics_names[3]} of {scores[3]*100}%;')

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
