import random
import time
import csv
import pickle
import tensorflow as tf
import numpy as np
from utils_res import read_data_file, sep_data_train_test_val, get_image_dims, xavier_init, rmse, get_sparsity
from dataset_res import DataSet
from rinn import RINN

### SET SEEDS
tf.set_random_seed(18) 
np.random.seed(18)

f = read_data_file('/home/me/Code/python/rinn/data/y_simdata1_mm.csv')  
g = read_data_file('/home/me/Code/python/rinn/data/x_simdata1_mm.csv') 

total_instances = 5000 

save_name = '/home/me/Output/test/test.csv'  

train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.1

### Create 2 validation and train sets
data = sep_data_train_test_val(g, train_ratio, test_ratio, val_ratio, f)
_train_dataset = data['train']
_validation_dataset = data['validation']
test_dataset = data['test']

trf = np.concatenate((_train_dataset.features, _validation_dataset.features), axis=0)
trl = np.concatenate((_train_dataset.labels, _validation_dataset.labels), axis=0)
print(trf.shape,trl.shape)

##training data
train = DataSet(trf, trl, to_one_hot=False)

random_index = np.random.choice(trf.shape[0], trf.shape[0], replace=False)
print(random_index.shape)

train_index1 = random_index[0:int(total_instances*train_ratio)]
validation_index1 = random_index[int(total_instances*train_ratio):]
print(train_index1.shape)
print(validation_index1.shape)

train_index2 = random_index[int(total_instances*val_ratio):]
validation_index2 = random_index[0:int(total_instances*val_ratio)]
print(train_index2.shape)
print(validation_index2.shape)

train1_features = trf[train_index1,:]
val1_features = trf[validation_index1,:]
train2_features = trf[train_index2,:]
val2_features = trf[validation_index2,:]
train1_labels = trl[train_index1,:]
val1_labels = trl[validation_index1,:]
train2_labels = trl[train_index2,:]
val2_labels = trl[validation_index2,:]

train_dataset_1 = DataSet(train1_features, train1_labels,to_one_hot=False)
validation_dataset_1 = DataSet(val1_features, val1_labels,to_one_hot=False)
train_dataset_2 = DataSet(train2_features, train2_labels,to_one_hot=False)
validation_dataset_2 = DataSet(val2_features, val2_labels,to_one_hot=False)

data_1 = (train_dataset_1, validation_dataset_1, 1)
data_2 = (train_dataset_2, validation_dataset_2, 2)

start = time.time()

yit = 0

### intitializers  
### Xavier:  tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
### He:      tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
### Use 'xavier_custom' to get the custom xavier function I wrote
all_initializers = [
        tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False),
        tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
        'xavier_custom']
all_optimizers = [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer, tf.train.GradientDescentOptimizer]

### RANDOM SEARCH
for _ in range(2):  ### 200
    seed = random.randint(0,50000000)
    l = random.choice([8,10,20])
    rr = 10**-random.uniform(2,5.3)
    lr = 10**-random.uniform(1,4.5)
    bs = random.choice([64,128,256])
    activ = random.choice([tf.nn.relu,tf.nn.sigmoid,tf.nn.softplus])
    initial = random.choice(all_initializers)
    optimizer = random.choice(all_optimizers)

    best_val_error_list = []
    avg_val_error = 999
    for split in [data_1,data_2]:
        if yit > 0:
            sess.close(); tf.reset_default_graph(); 
        yit += 1   

        tf.set_random_seed(seed)
        np.random.seed(seed)

        # cost_function = tf.nn.sigmoid_cross_entropy_with_logits
        # output_layer_activation=tf.nn.sigmoid
        ### config for continuous data (see above for changes necessary for binary)
        config = {'save_path': '/home/me/Output/test', 'hidden_layers': [l,l,l,l,l,l,l,l], 
            'activation': activ, 'cost_function': rmse,
            'optimizer': optimizer,
            'regularizer': [tf.contrib.layers.l1_regularizer, rr], 
            'initializer': initial,
            'learning_rate': lr, 
            'training_epochs': 3000
            'batch_size': bs,
            'display_step': 10, 'save_costs_to_csv': False,
            'improvement_threshold': 0.99995, 'count_threshold': 50}
        
        ### Buid, train, and save model
        ### train_dataset = split[0], validation_dataset = split[1]
        sess = tf.InteractiveSession()
        rinn = RINN(sess, config, split[0], split[1])  # init config and build graph
        best_val_cost, best_val_error, b_c_epoch, b_e_epoch, early, final_epoch, best_weights, best_biases = rinn.train() 
        best_val_error_list.append(best_val_error) 
        f_c, f_e, f_a, f_a_row = rinn.get_validation_cost_and_accuracy(split[1], multilabel=True) ### CHANGE!!!

        ae, sp = get_sparsity(rinn.weights,0.1)

        if split[2] == 2:
            avg_val_error = np.sum(np.array(best_val_error_list))/2.0
        print('active edges 0.1: ', ae)
        print('sparsity 0.1: ', sp)
        ae2, sp2 = get_sparsity(rinn.weights,0.01)
        print('active edges 0.01: ', ae2)
        print('sparsity 0.01: ', sp2)
        total = 0
        for ii in rinn.weights:
            total += np.sum(np.abs(ii.eval()))
        print('sum of weights: ', total)
        print()
        print(seed)
        print()
        new_row = [split[2],avg_val_error,best_val_cost,best_val_error,b_c_epoch,b_e_epoch,early,final_epoch,seed,l,rr,lr,bs,activ,initial,optimizer,ae,sp,ae2,sp2,total] 
        print(new_row)
        with open(save_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)

### GRID SEARCH
bs = 128
for initial in ['xavier_custom', tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)]:
    for l in [8, 16]:
        for lr in [0.01, 0.001, 0.0001]:
            for rr in [0.05, 0.01, 0.005, 0.001, 0.0005]:
                for activ in [tf.nn.relu,tf.nn.sigmoid]:
                    for optimizer in [tf.train.AdamOptimizer, tf.train.RMSPropOptimizer]:
                        seed = random.randint(0,50000000)

                        best_val_error_list = []
                        avg_val_error = 999
                        for split in [data_1,data_2]:
                            if yit > 0:
                                sess.close(); tf.reset_default_graph(); 
                            yit += 1   

                            tf.set_random_seed(seed)
                            np.random.seed(seed)

                            # cost_function = tf.nn.sigmoid_cross_entropy_with_logits
                            # output_layer_activation=tf.nn.sigmoid
                            ### config for continuous data (see above for changes necessary for binary)
                            config = {'save_path': '/home/me/Output/test', 'hidden_layers': [l,l,l,l,l,l,l,l], 
                                'activation': activ, 'cost_function': rmse,
                                'optimizer': optimizer,
                                'regularizer': [tf.contrib.layers.l1_regularizer, rr], 
                                'initializer': initial,
                                'learning_rate': lr, 
                                'training_epochs': 3000,
                                'batch_size': bs,
                                'display_step': 10, 'save_costs_to_csv': False,
                                'improvement_threshold': 0.99995, 'count_threshold': 50}
                            
                            ### Buid, train, and save model
                            ### train_dataset = split[0], validation_dataset = split[1]
                            sess = tf.InteractiveSession()
                            rinn = RINN(sess, config, split[0], split[1])  # init config and build graph
                            best_val_cost, best_val_error, b_c_epoch, b_e_epoch, early, final_epoch, best_weights, best_biases = rinn.train()  
                            best_val_error_list.append(best_val_error) ### FORGOT TO ADD THIS IN Version i actually ran!!
                            f_c, f_e, f_a, f_a_row = rinn.get_validation_cost_and_accuracy(split[1], multilabel=True) ### CHANGE!!!

                            ae, sp = get_sparsity(rinn.weights,0.1)

                            if split[2] == 2:
                                avg_val_error = np.sum(np.array(best_val_error_list))/2.0
                            print('active edges 0.1: ', ae)
                            print('sparsity 0.1: ', sp)
                            ae2, sp2 = get_sparsity(rinn.weights,0.01)
                            print('active edges 0.01: ', ae2)
                            print('sparsity 0.01: ', sp2)
                            total = 0
                            for ii in rinn.weights:
                                total += np.sum(np.abs(ii.eval()))
                            print('sum of weights: ', total)
                            print()
                            print(seed)
                            print()
                            new_row = [split[2],avg_val_error,best_val_cost,best_val_error,b_c_epoch,b_e_epoch,early,final_epoch,seed,l,rr,lr,bs,activ,initial,optimizer,ae,sp,ae2,sp2,total] 
                            print(new_row)
                            with open(save_name, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow(new_row)

end = time.time()
print(end - start)