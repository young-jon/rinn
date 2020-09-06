import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import copy
import pickle
import csv
import random
import es
from utils_evo import relu, identity, flatten_params, forward_prop_rinn, unflatten_and_update_params


def mse_fxn(y_pred, y, average=True):
    ### this function is not the same as how i calculated rmse with my rinn tensorflow scripts
    ### this fxn calcs error/instance (average=True) vs. error/deg (what tf scripts do)
    if average:
        n = y.shape[0]
        return (np.square(y_pred - y).sum())/n
    else:
        return np.square(y_pred - y).sum()

    
save_name = '/home/me/Output/test/test_ga.csv'  ###CHANGE
save_name_params = '/home/me/Output/test/test_ga_params.csv'  ###CHANGE

### Get Data
df_x = pd.read_csv('/home/me/Code/python/rinn/data/x_simdata1_mm.csv', delimiter=',', header=None) ###CHANGE
x = df_x.values
df_y = pd.read_csv('/home/me/Code/python/rinn/data/y_simdata1_mm.csv', delimiter=',', header=None) ###CHANGE
y = df_y.values

### smaller data to run faster!!!
x = x[0:1200]   
y = y[0:1200]   

seed = 19 ### random state
np.random.seed(seed)
features_train, features_test, targets_train, targets_test = train_test_split(x,y,test_size=.333333333,random_state=seed)
### really hard to overfit while adding a regularization penalty
### could train on 800 and test on 400

seed = 29258803 ### random state for GA
np.random.seed(seed)

### random search
elite_ratio = 0.15
mutation_rate = 0.002
L1_rate = 9.2096467608
legal_param_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4, -1.6, -1.8, -2.0]
hid_layers = [8,8,8,8,8,8,8,8]
epochs = 100 #Change to 7001 when running for real
NPOPULATION = 201
d_in = x.shape[1]
d_out = y.shape[1]

# Get NPARAMS by creating temporary containers for biases and weights (redundant included)
weights = []
biases = []
weights.append(np.zeros((d_in, hid_layers[0])))
biases.append(np.zeros(hid_layers[0]))
for ind in range(len(hid_layers) - 1):
    weights.append(np.zeros((hid_layers[ind] + d_in, hid_layers[ind+1])))
    biases.append(np.zeros(hid_layers[ind+1]))
weights.append(np.zeros((hid_layers[-1], d_out)))
biases.append(np.zeros(d_out))

NPARAMS, model_shapes = flatten_params(weights, biases)
history_epoch = []
history_epoch_test = []


solver = es.ConstrainedSimpleGA(NPARAMS,       # number of model parameters
              popsize=NPOPULATION,  # number of individuals in population (num of complete parameter sets)
              forget_best=False,    # forget the historical best elites
              sigma_init=0.1,       # initial standard deviation
              sigma_limit=0.01,
              sigma_decay=0.999,
              elite_ratio=elite_ratio,      # percentage of the elites to keep each generation
              weight_decay=0.00,    # L2 Coefficient. could easily change to L1 in es.py
              weight_decay_l1=L1_rate,   # 100.0 seems to work well here, 20 seems good, 50 seems good
              legal_param_values=legal_param_values,
              mutation_rate=mutation_rate
             )

start = time.time()

for e in range(epochs):
    fitness_list = np.zeros(solver.popsize) # vector of zeros
    
    ### RUN EVOLUTIONARY STRATEGY ON EACH MEMBER OF POPULATION
    solutions = solver.ask()
    for s in range(solver.popsize):
        weights, biases = unflatten_and_update_params(solutions[s], model_shapes, len(biases))
        ### does not use RINN class, but a RINN without regularization using numpy (forward_prop_rinn)
        fitness_list[s] = -mse_fxn(forward_prop_rinn(features_train, weights ,biases), 
                                   targets_train, average=True)
    

    ### get average fitness for the epoch (ie. across all batches)

    solver.tell(fitness_list)
    result = solver.result()
    # return best params so far, along with historically best reward, curr reward, sigma
    # self.best_reward (i.e. result[1]): is the reward (plus regularization penalty) for only the best set 
    # of params
    history_epoch.append(result[1])
    if e % 100 == 0:
        w_val, b_val = unflatten_and_update_params(result[0], model_shapes, len(biases))
        history_epoch_test.append(-mse_fxn(forward_prop_rinn(features_test, w_val, b_val), targets_test, average=True))
        print('average epoch fitness for epoch '+str(e+1)+'  train: '+str(result[1])+'  '+
              str(result[2])+'   test: '+str(history_epoch_test[-1]))
        if e == 200:
            if history_epoch_test[-1] < -6.5:
                print("Didn't make -6.5 cutoff at epoch 200! Stopping!")
                break
            else:
                print("Made -6.5 cutoff!")
        if e % 1000 == 0:
            if e >= 2000:
                if history_epoch_test[-1] <= history_epoch_test[-20]:
                    print('Test MSE has not improved for 2000 epochs! Stopping!')
                    break
                else:
                    print('Still improving!')
              
    ### check to make sure fitness is improving
    if (e+1) % 550 == 0:
        difference = history_epoch[e] - history_epoch[e-499]
        print(e, history_epoch[e], history_epoch[e-499], difference, history_epoch_test[-1])
        if difference < 0.3:
            if history_epoch_test[-1] >= -2.8:  ### UPDATE
                print("Test fitness is getting good! Keep going!")
                pass
            else:
                print("Stopped due to early stopping!!!")
                break

### if really good, train some more
if history_epoch_test[-1] >= -1.2:
    e_old = e
    e_old += 1
    for e in range(e_old, e_old+13000):
        fitness_list = np.zeros(solver.popsize) # vector of zeros
        
        ### RUN EVOLUTIONARY STRATEGY ON EACH MEMBER OF POPULATION
        solutions = solver.ask()
        for s in range(solver.popsize):
            weights, biases = unflatten_and_update_params(solutions[s], model_shapes, len(biases))
            fitness_list[s] = -mse_fxn(forward_prop_rinn(features_train, weights ,biases), 
                                       targets_train, average=True)
        ### get average fitness for the epoch (ie. across all batches)
        solver.tell(fitness_list)
        result = solver.result()
        # return best params so far, along with historically best reward, curr reward, sigma
        # self.best_reward (i.e. result[1]): is the reward (plus regularization penalty) for only the best set 
        # of params
        history_epoch.append(result[1])
        if e % 100 == 0:
            w_val, b_val = unflatten_and_update_params(result[0], model_shapes, len(biases))
            history_epoch_test.append(-mse_fxn(forward_prop_rinn(features_test, w_val, b_val), targets_test, average=True))
            print('average epoch fitness for epoch '+str(e+1)+'  train: '+str(result[1])+'  '+
                  str(result[2])+'   test: '+str(history_epoch_test[-1]))
                  
        ### check to make sure fitness is improving
        if (e+1) % 550 == 0:
            difference = history_epoch[e] - history_epoch[e-499]
            print(e, history_epoch[e], history_epoch[e-499], difference, history_epoch_test[-1])
            if difference < 0.3:
                if history_epoch_test[-1] >= -1.3:  ### UPDATE
                    print("Test fitness is getting good! Keep going!")
                    pass
                else:
                    print("Stopped due to early stopping!!!")
                    break

weights, biases = unflatten_and_update_params(result[0], model_shapes, len(biases))
history_epoch_test.append(-mse_fxn(forward_prop_rinn(features_test, weights ,biases), targets_test, average=True))
### WILL NEED TO CHANGE CALCULATION BELOW IF USE LARGE DATASET
train_mse = -mse_fxn(forward_prop_rinn(features_train, weights ,biases), targets_train, average=True)
print('FINAL fitness e '+str(e+1)+' train: '+str(train_mse)+' reg: '+str(result[1])+' '+
      str(result[2])+' test: '+ str(history_epoch_test[-1]))
output_list = [history_epoch_test[-1], train_mse, result[1], result[2], seed, elite_ratio, mutation_rate, 
               L1_rate, e, legal_param_values]
print('output: ', output_list)

end = time.time()
print('Run Time in Minutes: ', (end-start)/60) ## print num minutes
print()

### SAVE errors and hyperparameters
# with open(save_name, 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(output_list)

### SAVE weights and biases
# with open(save_name_params, 'a') as f:
#     writer = csv.writer(f)
#     writer.writerow(result[0])