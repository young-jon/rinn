import random
import time
import csv
import pickle
import tensorflow as tf
import numpy as np
from utils_res import rmse, get_sparsity
from rinn import RINN

### SET SEEDS
### Comment out these 2 lines to get differnt results for each run of RINN.
tf.set_random_seed(18)
np.random.seed(18)

### Change to directory where your data is located.
data_stem = '/home/me/Code/python/rinn/data/'

### Get data
train_file_name = data_stem+'train_dataset_simdata1_mm.pkl'
test_file_name = data_stem+'test_dataset_simdata1_mm.pkl'
f = open(train_file_name, 'rb')
train_dataset = pickle.load(f, encoding='latin1')
f.close()
g = open(test_file_name, 'rb')
test_dataset = pickle.load(g, encoding='latin1')
g.close()

### save path for when you want to save errors, weights, etc. 
save_path = '/home/me/Output/test/'  

### TRAIN 
start = time.time()

### Setup hyperparameters
seed = 13192399
l = 10
rr = 0.0005
lr = 0.0001
bs = 128
te = 100 #2398
activ = tf.nn.relu
cost_function = rmse
output_activation = tf.identity
optimizer = tf.compat.v1.train.AdamOptimizer
initializer = 'xavier_custom' 

tf.compat.v1.set_random_seed(seed)
np.random.seed(seed)
config = {'save_path': save_path, 'hidden_layers': [l,l,l,l,l,l,l,l], 
    'activation': activ, 'cost_function': cost_function, 
    'optimizer': optimizer,
    'regularizer': [tf.contrib.layers.l1_regularizer, rr], 
    'initializer': initializer,
    'learning_rate': lr, 
    'training_epochs': te,
    'batch_size': bs,
    'display_step': 10, 'save_costs_to_csv': False, 
    'improvement_threshold': 0.99995, 'count_threshold': 3000}

### Buid, train, and save model
sess = tf.compat.v1.InteractiveSession()
rinn = RINN(sess, config, train_dataset, test_dataset)  # init config and build graph 
best_val_cost, best_val_error, b_c_epoch, b_e_epoch, early, final_epoch, best_weights, best_biases = rinn.train()
f_c, f_e, f_a, f_a_row = rinn.get_validation_cost_and_accuracy(test_dataset, multilabel=True, output_layer_activation=output_activation) 

### Print 'sparsity' of graph
ae, sp = get_sparsity(rinn.weights,0.1)
print('active edges 0.1: ', ae)
print('sparsity 0.1: ', sp)
ae2, sp2 = get_sparsity(rinn.weights,0.01)
print('active edges 0.01: ', ae2)
print('sparsity 0.01: ', sp2)
total = 0
for w in rinn.weights:
    total += np.sum(np.abs(w.eval()))
print('sum of weights: ', total)
print()

### Comment out line below if want to stay in Tensorflow Interactive Session
sess.close(); tf.reset_default_graph();
