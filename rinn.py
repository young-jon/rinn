from __future__ import division, print_function, absolute_import
import time
import csv
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from utils_res import xavier_init

class RINN(object):
    '''
    A Redundant Input Neural Network built from a Deep Neural Network 
    (Multilayer Perceptron) implementation using the TensorFlow library. See 
    __main__ for example usage.

    Args:

    config (dict): file of hyperparameters
    train_dataset (DataSet): Training data in the form of a DataSet object from 
        dataset.py. Labels should be one-hot-vector.
    validation_dataset (DataSet): same as train_dataset

    '''

    def __init__(self, sess, config, train_dataset, validation_dataset,
    				pretrain_weights=None, pretrain_biases=None):
        self.sess = sess
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.pretrain_weights = pretrain_weights
        self.pretrain_biases = pretrain_biases
        self.save_path = config['save_path']
        self.hidden_layers = config['hidden_layers']
        self.activation = config['activation']
        self.cost_function = config['cost_function']
        self.optimizer = config['optimizer']
        self.regularizer = config['regularizer']
        self.initializer = config['initializer']
        self.learning_rate = config['learning_rate']
        self.training_epochs = config['training_epochs']
        self.batch_size = config['batch_size']
        self.display_step = config['display_step']
        self.save_costs_to_csv = config['save_costs_to_csv']
        self.improvement_threshold = config['improvement_threshold']
        self.count_threshold = config['count_threshold']

        if self.save_costs_to_csv:
            self._complete_save_path = self.save_path + 'out_' + time.strftime("%m%d%Y_%H:%M:%S") + '/'
            if not os.path.exists(self._complete_save_path):
                os.makedirs(self._complete_save_path)

        self._build_graph()

    def _build_graph(self):
        '''Builds the RINN graph. This function is intended to be called by 
        __init__. It builds a symbolic deep neural network graph based on the 
        config hyperparameters.'''

        print('Building Graph...')
        self.n_input = self.train_dataset.features.shape[1] # determine from train dataset
        self.n_classes = self.train_dataset.labels.shape[1] # to_one_hot = True

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Store layer weights & biases (initialized using random_normal)
        all_layers = [self.n_input] + self.hidden_layers + [self.n_classes]
        print('Network Architecture: ', all_layers)
        self.weights=[]
        self.biases=[]
        if self.pretrain_weights and self.pretrain_biases:
            print('Using pretrained weights and biases.')
            for i in range(len(self.pretrain_weights)):
                self.weights.append(tf.Variable(self.pretrain_weights[i]))
                self.biases.append(tf.Variable(self.pretrain_biases[i]))
            self.weights.append(tf.Variable(tf.random_normal([all_layers[i+1], all_layers[i+2]])))
            self.biases.append(tf.Variable(tf.random_normal([all_layers[i+2]])))
        elif self.regularizer[0] == tf.contrib.layers.l2_regularizer or self.regularizer[0] == tf.contrib.layers.l1_regularizer:
            ### updated 9_18_18 by removing bias and adding tf.contrib initializers
            ### Code below is for an 8 hidden layer RINN.
            if self.initializer == 'xavier_custom':
                w1 = tf.get_variable(name='w1', initializer=xavier_init([all_layers[0],all_layers[1]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b1 = tf.get_variable(name='b1', initializer=xavier_init([all_layers[1]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w2 = tf.get_variable(name='w2', initializer=xavier_init([all_layers[0]+all_layers[1],all_layers[2]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b2 = tf.get_variable(name='b2', initializer=xavier_init([all_layers[2]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w3 = tf.get_variable(name='w3', initializer=xavier_init([all_layers[0]+all_layers[2],all_layers[3]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b3 = tf.get_variable(name='b3', initializer=xavier_init([all_layers[3]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w4 = tf.get_variable(name='w4', initializer=xavier_init([all_layers[0]+all_layers[3],all_layers[4]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b4 = tf.get_variable(name='b4', initializer=xavier_init([all_layers[4]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w5 = tf.get_variable(name='w5', initializer=xavier_init([all_layers[0]+all_layers[4],all_layers[5]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b5 = tf.get_variable(name='b5', initializer=xavier_init([all_layers[5]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w6 = tf.get_variable(name='w6', initializer=xavier_init([all_layers[0]+all_layers[5],all_layers[6]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b6 = tf.get_variable(name='b6', initializer=xavier_init([all_layers[6]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w7 = tf.get_variable(name='w7', initializer=xavier_init([all_layers[0]+all_layers[6],all_layers[7]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b7 = tf.get_variable(name='b7', initializer=xavier_init([all_layers[7]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w8 = tf.get_variable(name='w8', initializer=xavier_init([all_layers[0]+all_layers[7],all_layers[8]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b8 = tf.get_variable(name='b8', initializer=xavier_init([all_layers[8]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w9 = tf.get_variable(name='w9', initializer=xavier_init([all_layers[8],all_layers[9]]),
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b9 = tf.get_variable(name='b9', initializer=xavier_init([all_layers[9]]))
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
            else:
                print("Using tf.contrib initializers.")
                w1 = tf.get_variable(name='w1', shape=[all_layers[0],all_layers[1]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b1 = tf.get_variable(name='b1', shape=[all_layers[1]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w2 = tf.get_variable(name='w2', shape=[all_layers[0]+all_layers[1],all_layers[2]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b2 = tf.get_variable(name='b2', shape=[all_layers[2]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w3 = tf.get_variable(name='w3', shape=[all_layers[0]+all_layers[2],all_layers[3]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b3 = tf.get_variable(name='b3', shape=[all_layers[3]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w4 = tf.get_variable(name='w4', shape=[all_layers[0]+all_layers[3],all_layers[4]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b4 = tf.get_variable(name='b4', shape=[all_layers[4]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w5 = tf.get_variable(name='w5', shape=[all_layers[0]+all_layers[4],all_layers[5]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b5 = tf.get_variable(name='b5', shape=[all_layers[5]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w6 = tf.get_variable(name='w6', shape=[all_layers[0]+all_layers[5],all_layers[6]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b6 = tf.get_variable(name='b6', shape=[all_layers[6]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w7 = tf.get_variable(name='w7', shape=[all_layers[0]+all_layers[6],all_layers[7]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b7 = tf.get_variable(name='b7', shape=[all_layers[7]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
                w8 = tf.get_variable(name='w8', shape=[all_layers[0]+all_layers[7],all_layers[8]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b8 = tf.get_variable(name='b8', shape=[all_layers[8]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1])) 
                w9 = tf.get_variable(name='w9', shape=[all_layers[8],all_layers[9]], initializer=self.initializer,
                                     regularizer=self.regularizer[0](self.regularizer[1]))
                b9 = tf.get_variable(name='b9', shape=[all_layers[9]], initializer=self.initializer)
                                     #regularizer=self.regularizer[0](self.regularizer[1]))
            
            self.weights = [w1,w2,w3,w4,w5,w6,w7,w8,w9]
            self.biases = [b1,b2,b3,b4,b5,b6,b7,b8,b9]
            
        else:
            for i in range(len(all_layers)-1):
                self.weights.append(tf.Variable(xavier_init([all_layers[i], all_layers[i+1]])))
                # self.weights.append(tf.Variable(tf.random_normal([all_layers[i], all_layers[i+1]])))
                self.biases.append(tf.Variable(tf.random_normal([all_layers[i+1]])))

        # CREATE MODEL
        # create hidden layer 1
        self.model = []
        self.model.append(self.activation(tf.add(tf.matmul(self.x, self.weights[0]), self.biases[0])))
        # create remaining hidden layers
        for j in range(len(self.hidden_layers))[1:]:
            ### new code 5_8_17 ###
            self.model.append(self.activation(tf.add(tf.matmul(tf.concat([self.model[j-1],self.x],1), self.weights[j]), self.biases[j])))
            ### end new code
            
            # self.model.append(self.activation(tf.add(tf.matmul(self.model[j-1], self.weights[j]), self.biases[j])))
            # self.model.append(self.activation(tf.add(tf.matmul(self.model[j-1], tf.nn.sigmoid(self.weights[j])), self.biases[j])))
        #create output layer
        self.model.append(tf.add(tf.matmul(self.model[-1], self.weights[-1]), self.biases[-1]))
        # self.model.append(tf.add(tf.matmul(self.model[-1], tf.nn.sigmoid(self.weights[-1])), self.biases[-1]))
            

 
        # Construct model
        self.logits = self.model[-1]  ### output layer logits

        ### NOTES ### 
        # the output of tf.nn.softmax_cross_entropy_with_logits(logits, y) is an array of the 
        # size of the minibatch (256). each entry in the array is the cross-entropy (scalar value) 
        # for the corresponding image. tf.reduce_mean calculates the mean of this array. Therefore, 
        # the cost variable below (and the cost calculated by sess.run is a scalar value), i.e., the 
        # average cost for a minibatch). see tf.nn.softmax_cross_entropy_with_logits??

        # Define cost (objective function) and optimizer
        self.cost = tf.reduce_mean(self.cost_function(logits = self.logits, labels = self.y)) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.train_step = self.optimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.rmse = tf.reduce_mean(self.cost_function(logits = self.logits, labels = self.y))
        print('Finished Building RINN Graph')

    def train(self):
        print('Training RINN...')
        # initialize containers for writing results to file
        self.train_cost = []; self.validation_cost = [];

        # Launch the graph
        # with tf.Session() as self.sess:
        # self.sess.run(init)
        self.sess.run(tf.global_variables_initializer())

        # Training cycle
        ### new 1_29_18 for early stopping, updated 9_20_18
        best_validation_cost = np.inf
        best_validation_cost_epoch = 0
        best_validation_rmse = np.inf
        best_validation_rmse_epoch = 0
        count = 0
        early_stop = False
        ###
        for epoch in range(self.training_epochs):
            total_cost = 0.
            total_batch = int(self.train_dataset.num_examples/self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = self.train_dataset.next_batch(self.batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.train_step, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})
        
                # Collect cost for each batch
                total_cost += c

            # Compute average loss for each epoch
            avg_cost = total_cost/total_batch

            #compute validation set average cost for each epoch given current state of weights
            validation_avg_cost = self.cost.eval({self.x: self.validation_dataset.features, 
                                                    self.y: self.validation_dataset.labels})

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "train cost=", \
                    "{:.9f}".format(avg_cost), "validation cost=", \
                    "{:.9f}".format(validation_avg_cost))

            #collect costs to save to file
            self.train_cost.append(avg_cost)
            self.validation_cost.append(validation_avg_cost)
            
            ### new 1_29_18. updated 9_20_18 ###
            if validation_avg_cost < best_validation_cost * self.improvement_threshold:
                best_validation_cost = validation_avg_cost
                best_validation_cost_epoch = epoch
                count = 0
                validation_set_rmse = self.rmse.eval({self.x: self.validation_dataset.features, 
                                                self.y: self.validation_dataset.labels},
                                                session = self.sess)
                if validation_set_rmse < best_validation_rmse:
                    best_validation_rmse = validation_set_rmse
                    best_validation_rmse_epoch = epoch

                ### new 2_20_18
                ### keep track of weights
                # initialize containers for best weights and biases
                best_weights_list = []
                best_biases_list = []
                for jo in self.weights:
                    best_weights_list.append(jo.eval())
                for ko in self.biases:
                    best_biases_list.append(ko.eval())

            else:
                count += 1
            if count == self.count_threshold:
                print("Stopped due to early stopping!!!")
                early_stop = True
                break
            ### end new
                
        print("Optimization Finished!")

        if self.save_costs_to_csv:
            self.save_train_and_validation_cost()
            
        print('Best Validation Set RMSE: ', best_validation_rmse)
        return(best_validation_cost, best_validation_rmse, best_validation_cost_epoch, best_validation_rmse_epoch, 
            early_stop, epoch, best_weights_list, best_biases_list)

    def save_train_and_validation_cost(self):
        '''Saves average train and validation set costs to .csv, all with unique filenames'''
        # write validation_cost to its own separate file
        name = 'validation_costs_'
        file_path = self._complete_save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(self.validation_cost)
        # all error measures in one file
        df_to_disk = pd.DataFrame([self.train_cost, self.validation_cost],
                                    index=[[self.hidden_layers,self.learning_rate,
                                            self.training_epochs,self.batch_size], ''])
        df_to_disk['error_type'] = ['train_cost', 'validation_cost']
        # create file name and save as .csv
        name = 'all_costs_'
        file_path = self._complete_save_path + name + time.strftime("%m%d%Y_%H;%M;%S") + '.csv'
        df_to_disk.to_csv(file_path)
        print("Train and validation costs saved in file: %s" % file_path)

    ### ADDED 9_20_18###
    def get_validation_cost_and_accuracy(self, validation_dataset, output_layer_activation=tf.identity, multilabel=False):
        '''
        Calculate cost and accuracy for a validation dataset, given RINN weights at time of calling this function.

        ARGS:
        validation_dataset (DataSet): Validating data in the form of a DataSet object from 
        dataset.py. Labels should be one-hot-vector if binary.

        RETURNS:
        Validation set accuracy and average validation set cost over entire validation dataset
        '''
        #calculate validation sets cost

        validation_set_cost = self.cost.eval({self.x: validation_dataset.features, 
                                        self.y: validation_dataset.labels}, 
                                        session = self.sess)

        print('Final validation set cost: ', validation_set_cost)
        #calculate accuracy
        if multilabel:
            # correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(self.logits)), self.y)
            correct_prediction = tf.equal(tf.round(output_layer_activation(self.logits)), tf.round(self.y))
            accuracy_row = tf.reduce_mean(tf.cast(correct_prediction, "float"), axis = 0)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            validation_set_rmse = self.rmse.eval({self.x: validation_dataset.features, 
                                                self.y: validation_dataset.labels},
                                                session = self.sess)
            validation_set_accuracy = accuracy.eval({self.x: validation_dataset.features, 
                                                self.y: validation_dataset.labels},
                                                session = self.sess)
            validation_set_accuracy_row = accuracy_row.eval({self.x: validation_dataset.features, 
                                                self.y: validation_dataset.labels},
                                                session = self.sess)


        else:
            print('ERROR! multilabel must be True.')

        print("Final validation set error:", validation_set_rmse)
        print("Final validation set accuracy:", validation_set_accuracy)
        return(validation_set_cost, validation_set_rmse, validation_set_accuracy, validation_set_accuracy_row)

    def get_test_cost_and_accuracy(self, test_dataset, output_layer_activation=tf.identity, multilabel=False):
        '''
        Calculate cost and accuracy for a test dataset, given RINN weights at time of calling this function.

        ARGS:
        test_dataset (DataSet): Testing data in the form of a DataSet object from 
        dataset.py. Labels should be one-hot-vector.

        RETURNS:
        Test set accuracy and average test set cost over entire test dataset
        '''
        #calculate test and validation sets cost

        ### UPDATED 9_18_18 to add validation set cost and accuracy###
        test_set_cost = self.cost.eval({self.x: test_dataset.features, 
                                        self.y: test_dataset.labels}, 
                                        session = self.sess)

        print('Final test set cost: ', test_set_cost)
        #calculate accuracy
        if multilabel:
            # correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(self.logits)), self.y)
            correct_prediction = tf.equal(tf.round(output_layer_activation(self.logits)), tf.round(self.y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_set_accuracy = accuracy.eval({self.x: test_dataset.features, 
                                                self.y: test_dataset.labels},
                                                session = self.sess)
            test_set_rmse = self.rmse.eval({self.x: test_dataset.features, 
                                                self.y: test_dataset.labels},
                                                session = self.sess)

        else:
            print('ERROR! multilabel must be True.')
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_set_accuracy = accuracy.eval({self.x: test_dataset.features, 
                                                self.y: test_dataset.labels},
                                                session = self.sess)
        print("Final test set error:", test_set_rmse)
        print("Final test set accuracy:", test_set_accuracy)
        return(test_set_cost, test_set_rmse, test_set_accuracy)
        
    """def test_model(self, train_dataset, validation_dataset, test_dataset): 
        ### TESTS TO MAKE SURE THAT COSTS ARE COMPUTED AS EXPECTED USING CURRENT TRAIN_VALIDATION DATASET
        print("Testing model with default parameters!")
        ### DEFAULT NEURAL NETWORK HYPERPARAMETERS
        config = {
            'save_path': '/home/me/Output/biotensorflow/',
            'hidden_layers': [150,60],
            'activation': tf.nn.relu,
            'cost_function': tf.nn.softmax_cross_entropy_with_logits,
            'optimizer': tf.train.AdamOptimizer,
            #'regularizer': [None],
            'regularizer':[tf.nn.batch_normalization,1e-5],#batch_normalization_epsilon
            #'regularizer': [tf.nn.dropout,[0.6,1]],#dropout_rate
            'learning_rate': 0.001,
            'training_epochs': 3,
            'batch_size': 100,
            'display_step': 4, #training_epochs + 1, don't want to display anything
            'save_costs_to_csv': False
        }

        ### GENERATE RANDOM SEEDS
        tensorflow_seed = np.array([0, 42, 1234, 1776, 1729])
        numpy_seed = np.array([1729, 1776, 1234, 42, 0])

        test_costs = [None] * 5 

        for count in range(0,5): 
            #print(count)
            tf.set_random_seed(tensorflow_seed[count])
            np.random.seed(numpy_seed[count])

            with tf.Session() as sess: 
                #use subprocesses to silent 

                dnn = DNN(sess, config, train_dataset, validation_dataset)
                dnn.train()

                #evaluate model on a test set
                c, a = dnn.get_test_cost_and_accuracy(test_dataset)
                test_costs[count] = c

        print("#####")
        print("Average cost", sum(test_costs)/5)"""



if __name__ == '__main__':
    ### RUN
    import random
    import time
    import csv
    import pickle
    import tensorflow as tf
    import numpy as np
    from scipy.special import expit
    from sklearn import metrics
    from utils_res import read_data_file,sep_data_train_test_val,get_image_dims,xavier_init,rmse,forward_prop,softplus,nprelu,get_sparsity
    from dataset_res import DataSet

    ### SET SEEDS
    ### Comment out these 2 lines to get differnt results for each run of RINN.
    tf.set_random_seed(18)
    np.random.seed(18)

    all_initializers = {'He_contrib': tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
    'Xavier_contrib': tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False),
    'Xavier_custom': 'xavier_custom'}

    # Change to directory where your data is located.
    data_stem = '/home/me/Code/python/rinn/data/'

    train_file_name = data_stem+'train_dataset_simdata1_mm.pkl'
    test_file_name = data_stem+'test_dataset_simdata1_mm.pkl'
    f = open(train_file_name, 'rb')
    train_dataset = pickle.load(f, encoding='latin1')
    f.close()
    g = open(test_file_name, 'rb')
    test_dataset = pickle.load(g, encoding='latin1')
    g.close()

    ### save path
    save_path = '/home/me/Output/test/'  ###CHANGE

    ### TRAIN 
    start = time.time()

    seed = 13192399
    l = 10
    rr = 0.0005
    lr = 0.0001
    bs = 128
    te = 100 #2398, 100 for testing
    activ = tf.nn.relu
    activ_python = nprelu
    cost_function = rmse
    output_activation = tf.identity
    optimizer = tf.train.AdamOptimizer
    initializer = 'xavier_custom' 

    tf.set_random_seed(seed)
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
    sess = tf.InteractiveSession()
    rinn = RINN(sess, config, train_dataset, test_dataset)  # init config and build graph 
    best_val_cost, best_val_error, b_c_epoch, b_e_epoch, early, final_epoch, best_weights, best_biases = rinn.train()
    f_c, f_e, f_a, f_a_row = rinn.get_validation_cost_and_accuracy(test_dataset, multilabel=True, output_layer_activation=output_activation) ### CHANGE!!!

    ae, sp = get_sparsity(rinn.weights,0.1)
    print('active edges 0.1: ', ae)
    print('sparsity 0.1: ', sp)
    ae2, sp2 = get_sparsity(rinn.weights,0.01)
    print('active edges 0.01: ', ae2)
    print('sparsity 0.01: ', sp2)
    total = 0
    for su in rinn.weights:
        total += np.sum(np.abs(su.eval()))
    print('sum of weights: ', total)
    print()
    
    ### Comment out line below if want to stay in Tensorflow Interactive Session
    sess.close(); tf.reset_default_graph();
