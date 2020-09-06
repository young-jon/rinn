import numpy as np
import copy
import matplotlib.pyplot as plt



### IMAGING UTILS
def get_image_dims(n_input):
    ''' converts 1-D integer to 2-D representation for plotting as an image

    Args
    n_input (int):  number of features in the data that you want to plot as an image

    Returns
    A 2-D tuple (of dimensions) for plotting

    Usage
    print(get_image_dims(784))
    # (28, 28)
    print(get_image_dims(500))
    # (25, 20)

    '''
    ### check for perfect square
    if not (np.sqrt(n_input) - int(np.sqrt(n_input))):
        dimensions = (int(np.sqrt(n_input)), int(np.sqrt(n_input)))
    ### if not perfect square
    else:
        dim1 =[]
        dim2=[]
        mid = int(np.floor(np.sqrt(n_input)))
        for i in range(mid):
            if (n_input % (i+1)) == 0:
                # print(i+1)
                dim1.append(i+1)
        for i in range(mid,n_input):
            if (n_input % (i+1)) == 0:
                # print(i+1)
                dim2.append(i+1)
        dimensions = (min(dim2), max(dim1))
        if 1 in dimensions:
            print('prime number of features')
    return dimensions


### INITIALIZATION

### Goodfellow book recommends treating weight initialization as a hyperparameter.
### E.g. could try xavier_init, xavier_init2, tf.truncated_normal_initializer(stddev=stddev)) stdev=0.02 or other,
### tf.truncated_normal(shape, stddev=0.1). truncated_normal_initializer used by Improved_gan and Info_Gan code.
### For biases could use tf.constant(0.1, shape=shape) --- from tensorflow tutorial, zeros, or random_normal
def xavier_init(size):
    print('Using Xavier initialization')
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def xavier_init2(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)






def forward_prop(sga,w,b,activation,output_activation):
    '''forward propagation through a RINN. returns output activations. This assumes output is binary 
    (ie using sigmoid output function).
    sga = (numpy ndarray) input vector
    w = list of  weight matrices (each matrix is numpy ndarray) (8 hidden layer network), 8 hid means 9 weight matrices.
    b = list of bias vectors (each vector is numpy ndarray)
    activation = activation function for hidden layers (a python function)
    output_activation = activation function for output layer (e.g. sigmoid -- for binary, or identity -- for regression)
    '''
    # from scipy.special import expit  #sigmoid/logistic function
    
    da = np.asarray(sga)     
    h1 = activation(np.dot(da, w[0]) + b[0])
    h2 = activation(np.dot(np.concatenate((h1, da)), w[1]) + b[1])
    h3 = activation(np.dot(np.concatenate((h2, da)), w[2]) + b[2])
    h4 = activation(np.dot(np.concatenate((h3, da)), w[3]) + b[3])
    h5 = activation(np.dot(np.concatenate((h4, da)), w[4]) + b[4])
    h6 = activation(np.dot(np.concatenate((h5, da)), w[5]) + b[5])
    h7 = activation(np.dot(np.concatenate((h6, da)), w[6]) + b[6])
    h8 = activation(np.dot(np.concatenate((h7, da)), w[7]) + b[7])
    deg = output_activation(np.dot(h8, w[8]) + b[8])

    return deg

def forward_prop_dnn(sga,w,b,activation,output_activation):
    '''forward propagation through a DNN!!! returns output activations. This assumes output is binary 
    (ie using sigmoid output function).
    sga = (numpy ndarray) input vector
    w = list of  weight matrices (each matrix is numpy ndarray) (if 8 hid layers, then 9 weight vectors)
    b = list of bias vectors (each vector is numpy ndarray)
    activation = activation function for hidden layers (a python function)
    output_activation = activation function for output layer (e.g. sigmoid -- for binary, or identity -- for regression)
    '''
    # from scipy.special import expit  #sigmoid/logistic function
    
    h = np.asarray(sga)  
    for i in range(len(w)-1):
        h = activation(np.dot(h, w[i]) + b[i])
    deg = output_activation(np.dot(h, w[-1]) + b[-1])

    return deg

def softplus(x):
    return np.log(1.0 + np.exp(x))


def get_sparsity(weights, threshold): #weights is list of tensorflow variables 
    all_matrices = []
    for i in weights:
        all_matrices.append(np.where(np.abs(i.eval()) > threshold, 1, 0))
    active_edges = 0
    possible_edges = 0
    for j in all_matrices:
        active_edges += np.sum(j)
        possible_edges += (j.shape[0]*j.shape[1])
    return active_edges, active_edges/possible_edges



#### NEW FOR EVO STRATEGIES  ####


def relu(x):
    ### faster than return x * (x >0 )
    return np.maximum(x,0)

# def relu(x):
#     return x * (x > 0)

def identity(x):
    return x

def forward_prop_rinn(x, weights, biases, activ_hid=relu, activ_out=identity):
    '''
    Gets y_pred for forward propagation through a RINN.
    x = 2D numpy ndarray of input (rows are samples, columns are features)
    weights = list of numpy matrices representing the weights of a RINN
    biases = list of numpy vectors representing the biases of a RINN
    activation = python function on a numpy matrix
    '''
    out = activ_hid(x.dot(weights[0]) + biases[0])
    for ind in range(len(weights) - 2):  ### dont run this loop on 1st or last weight matrix (ie -2)
        out = np.concatenate((out, x), axis=1) # Add redundant input
        out = activ_hid(out.dot(weights[ind+1]) + biases[ind+1])
    y_pred = activ_out(out.dot(weights[-1]) + biases[-1])
    return y_pred

def flatten_params(weights, biases):
    ### returns NPARAMS and model_shapes ([weights]+[biases])
    params = weights + biases
    orig_params = copy.deepcopy(params)
    # get number of weight params and flatten
    flat_params = []
    model_shapes = []
    for p in orig_params:
        print(p.shape, p.flatten().shape)
        model_shapes.append(p.shape)
        flat_params.append(p.flatten())
    orig_params_flat = np.concatenate(flat_params)
    return len(orig_params_flat), model_shapes

def unflatten_and_update_params(flat_params, model_shapes, num_weight_matrices):  
    '''
    flat_params: new parameters that are flattened
    '''
    #num_weight_matrices = len(biases)
    weights = []
    biases = []
    idx = 0
    i = 0
    
    for j in range(len(model_shapes)):
        delta = np.product(model_shapes[i])
        block = flat_params[idx:idx+delta]
        block = np.reshape(block, model_shapes[i])
        if i < num_weight_matrices:
            weights.append(block)
        else:
            biases.append(block)

        i += 1
        idx += delta
        
    return weights, biases

    