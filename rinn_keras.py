import tensorflow as tf
from tensorflow import keras
from functools import partial
import pickle

# Change to directory where your data is located.
data_stem = '/home/me/Code/python/rinn/data/'

# Get data
train_file_name = data_stem+'train_dataset_simdata1_mm.pkl'
test_file_name = data_stem+'test_dataset_simdata1_mm.pkl'
f = open(train_file_name, 'rb')
train_dataset = pickle.load(f, encoding='latin1')
f.close()
g = open(test_file_name, 'rb')
test_dataset = pickle.load(g, encoding='latin1')
g.close()

# Hyperparams
layer_size = 10 
rr = 0.0005  #regularization rate
lr = 0.0001  #learning rate
e = 200
bs = 100

# Model
RegularizedDense = partial(keras.layers.Dense, 
	                       units=layer_size, 
	                       activation='relu', 
	                       kernel_initializer='he_normal', 
	                       kernel_regularizer=keras.regularizers.l1(rr))

input_ = keras.layers.Input(shape=(16,))
h1 = RegularizedDense()(input_)
concat1 = keras.layers.Concatenate()([h1, input_])
h2 = RegularizedDense()(concat1)
concat2 = keras.layers.Concatenate()([h2, input_])
h3 = RegularizedDense()(concat2)
concat3 = keras.layers.Concatenate()([h3, input_])
h4 = RegularizedDense()(concat3)
concat4 = keras.layers.Concatenate()([h4, input_])
h5 = RegularizedDense()(concat4)
concat5 = keras.layers.Concatenate()([h5, input_])
h6 = RegularizedDense()(concat5)
concat6 = keras.layers.Concatenate()([h6, input_])
h7 = RegularizedDense()(concat6)
concat7 = keras.layers.Concatenate()([h7, input_])
h8 = RegularizedDense()(concat7)
output = RegularizedDense(units=16, activation=None)(h8)
model = keras.Model(inputs=[input_], outputs=[output])

model.compile(loss='mse', optimizer='adam')
history = model.fit(train_dataset.features, train_dataset.labels, 
	                epochs=e, batch_size=bs,
	                validation_data=(test_dataset.features, test_dataset.labels))