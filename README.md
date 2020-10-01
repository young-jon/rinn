# rinn
Some of the code and all of the simulated data used in my Redundant Input Neural Network (RINN) work. 

See rinn_example.py and constrained_GA_example.py for simple examples of RINN and the constrained GA (i.e., ES-C). 
RINN was written with older versions of TensorFlow and works here (with warnings) with TensorFlow 1.14. It has not been updated
for later versions of TensorFlow, but works with some earlier versions :). 

To run from the command-line (recommend using a conda environment with the requirements installed):
```
python rinn_example.py
```

```
python constrained_GA_example.py
```

See our RINN paper presented at the 2020 KDD Workshop on Causal Discovery:  http://proceedings.mlr.press/v127/young20a.html

For running on GPU server:
```
CUDA_VISIBLE_DEVICES=0 nohup python -u model_selection_example.py &> model_selection.out
```

### For a basic Keras version of the RINN, see rinn_keras.py.  Here is the Keras model code without data or import statements:
```python
# Hyperparameters
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
```
