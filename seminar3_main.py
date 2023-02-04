import numpy as np
import copy
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# import argparse
import pandas as pd
import matplotlib.pyplot as plt
import neural_network_regressor as nn_reg

from sklearn.linear_model import Ridge
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import auc

from tabulate import tabulate

## much of this code is taken and/or adapted from:
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html#sphx-glr-download-auto-examples-linear-model-plot-poisson-regression-non-normal-loss-py    


###################################################################
### functions for computing quantities assessing quality of predictions
####################################################################
def lorenz_curve(f_gt, f_pred, exposure):

    #order samples by increasing predicted risk:
    ranking = np.argsort(f_pred.ravel())
    
    ranked_frequencies = f_gt[ranking]
    ranked_exposure = exposure[ranking]
    
    cumulated_claims = np.cumsum(ranked_frequencies * ranked_exposure)
    cumulated_claims /= cumulated_claims[-1]
    
    cumulated_exposure = np.cumsum(ranked_exposure)    
    cumulated_exposure /= cumulated_exposure[-1]
    
    return cumulated_exposure, cumulated_claims


# assume the frquency rate is being predicted
def ScoreEstimator(f_pred, target_freq, target_time):

    tt = np.sum(target_time)

    d = (f_pred - target_freq);

    est_scores = {}
    
    est_scores['MAE'] = np.sum(np.abs(d) * target_time) / tt;
    est_scores['MSE'] = np.sum(d*d * target_time) / tt;


    mask = (f_pred > 0)
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print("WARNING: Estimator yields invalid, non-positive predictions "
            f" for {n_masked} samples out of {n_samples}. These predictions "
            "are ignored when computing the Poisson deviance."
        )
        
    a = 2*f_pred[mask]

    tf = target_freq[mask]
    pf = f_pred[mask]
    ind = (target_freq[mask] > 0).nonzero()
    
    a[ind[0]] = 2 * (pf[ind[0]] + tf[ind[0]] * np.log(tf[ind[0]] / pf[ind[0]]) - tf[ind[0]])
    
    tta = np.sum(target_time[mask])    
    est_scores['mean Poisson deviance'] = np.sum(a * target_time[mask]) / tta;

    # put in other potential metrics later
    # Brier
    # Ranked prob scores

    return est_scores


###############################################

#######################################################
## Functions for printing out formatted results & info
#######################################################
    
def PrintOutMetrics(est_name, est_scores):

    # print('---------------------------------------')
    # print('---------------------------------------')
    print('Metrics for the ' +  est_name + ' estimator:')    
    names = est_scores.keys()
    for name in names:
        print(name + f' score: {est_scores[name]:.4f}')
    print('---------------------------------------')
    print('---------------------------------------')        

def print_table_of_results(est_names, est_scores, msg):

    table_to_display = []
    
    table_headers = ['Method']
    kk = est_scores[est_names[0]].keys()
    for k in kk:
        table_headers.append(k)
    
    for name in est_names:
        table_row = [name]
        kk = est_scores[name].keys()
        for k in kk:
            table_row.append(est_scores[name][k])
        table_to_display.append(table_row)

    print(msg)
    print(' ')
    print(tabulate(table_to_display, headers = table_headers))

def print_out_sample_inputs(data, id):

    n = data['inputs'].shape[1] 
    for i in range(n):
        print(data['input names'][i], data['inputs'][id, i])
        

#######################################################    


#############################################
### functions for plotting various quantities
#############################################

def display_lorenz_curves(est_names, freq, f_pred, data, lorenz_ax):

    for idx in range(len(est_names)):
        cum_exposure, cum_claims = lorenz_curve(freq, f_pred[est_names[idx]], data['outputs'][:, 1:2])
        gini = 1 - 2 * auc(cum_exposure, cum_claims)
        label = "{} (Gini: {:.3f})".format(est_names[idx], gini)
        lorenz_ax.plot(cum_exposure, cum_claims, linestyle="-", label=label)
        
    cum_exposure, cum_claims = lorenz_curve(freq, freq, data['outputs'][:, 1:])
    gini = 1 - 2 * auc(cum_exposure, cum_claims)
    label = "{} (Gini: {:.2f})".format("Oracle", gini)
    lorenz_ax.plot(cum_exposure, cum_claims, linestyle="-.", color= "black", label=label)

    lorenz_ax.plot([0, 1], [0, 1], linestyle="--", color="green", label="Random baseline")
    lorenz_ax.set(
        title="Lorenz curves by model",
        xlabel="Cumulative proportion of exposure (from predicted safest to riskiest)",
        ylabel="Cumulative proportion of claims",
    )    
    lorenz_ax.legend(loc="upper left")

def DisplayRawOutput(outputs, axs):

    frequency = outputs[:, 0:1] / outputs[:, 1:]

    n_bins = 5
    hh = np.histogram(outputs[:, 0:1], range=(0,4), bins=n_bins)[0]
    bin_edges = np.linspace(-.5, 4.5, n_bins + 1)
    widths = np.diff(bin_edges)
    centers = bin_edges[:-1] + widths / 2

    ## plot histograms of the output data
    axs[0].set_title("Number of claims")
    axs[0].set_yscale('log')
    axs[0].bar(centers, hh, width=widths)

    axs[1].set_title("Exposure in years")
    axs[1].set_yscale('log')
    axs[1].hist(outputs[:, 1:], bins=20)

    axs[2].set_title("Rate of claims / year")
    axs[2].set_yscale('log')
 #   axs[2].hist(frequency, bins=30)

    n_bins = 30
    axs[2].hist(frequency, range=(-.5,5), bins=n_bins)
#    axs[2].set_title(name)
#    axs[2].set_yscale('log')
    # axs[2].bar(centers, hh, width=widths)

def make_disp_hist(preds, names, axs):

    #bins=np.linspace(-1, 4, n_bins)

    #names = preds.keys()    

    n_bins = 20
    bin_edges = np.linspace(-.5, 4.5, n_bins + 1)
    widths = np.diff(bin_edges)
    centers = bin_edges[:-1] + widths / 2
    
    for idx, name in enumerate(names):
        hh = np.histogram(preds[name], range=(0,5), bins=n_bins)[0]
        axs[idx].set_title(name)
        axs[idx].set_yscale('log')
        axs[idx].bar(centers, hh, width=widths)    

################################################

#############################################
### functions for reading in raw data
#############################################

def ReadInCSVData(fname):
    
    df = pd.read_csv(fname)
    XX = df.values.T.tolist()

    names = df.columns

    data = {}
    data['outputs']   = np.transpose(np.array(XX[1:3]))
    data['output names'] = [names[1], names[2]]
    
    data['inputs']    = np.transpose(np.array(XX[3:]))
    data['input names'] = names[3:]
    
    data['id number'] = np.transpose(np.array(XX[0]))

    ## data['input names'] has the high level names of the columns of data['inputs']
    ## 0 = VehPower   (int)
    ## 1 = VehAge     (int)
    ## 2 = DrivAge    (int)
    ## 3 = BonusMalus (int between 50 and 230)
    ## 4 = VehBrand  (cat, 11 levels)
    ## 5 = VehGas    (cat, binary)
    ## 6 = Area      (cat, 6 levels)
    ## 7 = Density   (int)
    ## 8 = Region    (cat, 22 levels)

    ## data['input names'] contains columns names of data['outputs']
    ## 0 = Number of claims   (int)  (made during the insured period)
    ## 1 = Exposure     (float)      (duration of the insured period)    
    

    return data


################################################

#############################################
### functions for pre-processing data
#############################################
        
def OneHotCatData(X_inputs, num_cats):    

    one_hot_X = np.zeros((X_inputs.shape[0], np.sum(num_cats)))
    
    inds =  np.arange(X_inputs.shape[0])
    st = 0
    for idx, ii in enumerate(range(X_inputs.shape[1])):
        # column indices 
        jj = st + X_inputs[:, ii]

        one_hot_X[inds, jj.astype(int)] = 1
        st += num_cats[idx]
        
    return one_hot_X

def NormalizeData(inputs, mean_input="", std_input=""):
    
    if len(mean_input) < 1:
        mean_input = np.mean(inputs, axis=0)
        std_input = np.std(inputs, axis=0)    
        normalized_inputs = (inputs - mean_input) / std_input
        return normalized_inputs, mean_input, std_input
    
    else:
        normalized_inputs = (inputs - mean_input) / std_input
        return normalized_inputs

################################################    
    

    

if __name__ == "__main__":
    


    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--display", default="True", help="Set to True to make plots of training data and results")
    parser.add_argument("-n", "--nn", default="False", help="Set to True to train the neural networks")
    parser.add_argument("-dir", "--data_dir", default="", help="Give the directory name of where the data resides")
    args = vars(parser.parse_args())

    # Set up parameters
    display_results = args["display"]
    do_train_nn = args["nn"] #set to True to train the neural networks. This training is a little slow
    data_dir = args["data_dir"] #set to True to train the neural networks. This training is a little slow    

    #######################################################
    ## Read in the data
    #######################################################
    
    train_data = ReadInCSVData(data_dir + '/num_car_train_data.csv')
    val_data   = ReadInCSVData(data_dir + '/num_car_val_data.csv')
    test_data  = ReadInCSVData(data_dir + '/num_car_test_data.csv')

    train_freq = train_data['outputs'][:, 0:1] / train_data['outputs'][:, 1:]
    val_freq   = val_data['outputs'][:, 0:1] / val_data['outputs'][:, 1:]
    test_freq  = test_data['outputs'][:, 0:1] / test_data['outputs'][:, 1:]


    #######################################################
    ## Basic summary and visualization of the output data
    #######################################################
    
    ## percentage of no claims    
    n_zero = np.count_nonzero(train_data['outputs'][:, 0:1] == 0)
    p_zero = n_zero / train_data['outputs'][:, 0:1].shape[0]

    ind = (train_data['outputs'][:, 0:1] == 0).nonzero()
    p_zero0 = np.sum(train_data['outputs'][ind[0], 1:]) / np.sum(train_data['outputs'][:, 1:])

    ## mean frequency
    mean_freq = np.mean(train_freq)
    ## weighted mean frequency
    mean_freq0 = np.sum(train_freq * train_data['outputs'][:, 1:]) / np.sum(train_data['outputs'][:, 1:])

    print('SUMMARY OF TRAINING DATA:')
    print(f'Fraction of training examples with zero claims: {p_zero:.4f}.')
    print(f'Fraction of exposure with zero claims = {p_zero0:.4f}.')
    print(f'Mean frequency: {mean_freq:.4f}')
    print(f'Average # of claims weighted by exposure: {mean_freq0:.4f}')
    print('---------------------------------------')
    print('---------------------------------------')

    ## display the output for the training
    if display_results:
        output_fig, output_axs = plt.subplots(1, 3, tight_layout=True)
        DisplayRawOutput(train_data['outputs'], output_axs)    
        output_fig.suptitle('Histogram of output of training data', fontsize=16)

    
    #######################################################
    ## list of the classifiers fit to the training data
    #######################################################

    #set this to True if you want to train the neural networks which is a little slow    
    if do_train_nn:
        est_names = ['baseline', 'linear', 'poisson glm', 'poisson gbrt', 'nn', 'balanced nn']
    else:
        est_names = ['baseline', 'linear', 'poisson glm']
    val_scores = {}
    val_fpred = {}
    test_fpred = {}    
    
    
    #######################################################
    ## Simple mean baseline
    ######################################################

    ## the baseline mean prediction
    val_fpred[est_names[0]] = mean_freq0 * np.ones((val_data['outputs'].shape[0], 1))
    
    val_scores[est_names[0]] = ScoreEstimator(val_fpred[est_names[0]], val_freq, val_data['outputs'][:, 1:])
    PrintOutMetrics(est_names[0], val_scores[est_names[0]])


    #######################################################    
    ## Pre-process the input data
    ####################################################

    # obtain 1-hot encoding of catgorical entries
    #print('Convert the categorical entries to 1-hot encoding')
    categorical_ind  = [4, 5, 6, 8]
    num_cats = [11, 2, 6, 22]

    train_inputs_onehot = OneHotCatData(train_data['inputs'][:, categorical_ind], num_cats)
    val_inputs_onehot = OneHotCatData(val_data['inputs'][:, categorical_ind], num_cats)
    test_inputs_onehot = OneHotCatData(test_data['inputs'][:, categorical_ind], num_cats)

    # normalize the non-categorical data
    train_normalized = np.c_[copy.deepcopy(train_data['inputs'][:, 0:4]), np.log(train_data['inputs'][:, 7:8])]
    
    [train_normalized, mean_input, std_input] = NormalizeData(train_normalized)

    val_normalized = np.c_[copy.deepcopy(val_data['inputs'][:, 0:4]), np.log(val_data['inputs'][:, 7:8])]
    val_normalized = NormalizeData(val_normalized, mean_input, std_input)
    
    test_normalized = np.c_[copy.deepcopy(test_data['inputs'][:, 0:4]), np.log(test_data['inputs'][:, 7:8])]
    test_normalized = NormalizeData(test_normalized, mean_input, std_input)        

    ####################################################
    ## form of input used for each predictor type
    ####################################################
    
    ## pre-processed inputs used for the linear and glm predictors
    train_lin_inputs = np.c_[train_normalized, train_inputs_onehot]
    val_lin_inputs   = np.c_[val_normalized, val_inputs_onehot]
    test_lin_inputs  = np.c_[test_normalized, test_inputs_onehot]

    ## pre-processed inputs used for the gbrt and neural network predictor
    train_gbrt_inputs = np.c_[train_normalized, train_data['inputs'][:, categorical_ind]]
    val_gbrt_inputs   = np.c_[val_normalized, val_data['inputs'][:, categorical_ind]]
    test_gbrt_inputs  = np.c_[test_normalized, test_data['inputs'][:, categorical_ind]]            


    #######################################################    
    ## Linear predictor
    ####################################################
    
    ## fit the linear model via ridge regression
    print('Fitting the LINEAR model')
    ridge_lm = Ridge(alpha=1e-6)
    
    ridge_lm.fit(train_lin_inputs, train_freq, sample_weight = train_data['outputs'][:, 1])
    
    val_fpred[est_names[1]] = ridge_lm.predict(val_lin_inputs);
    val_scores[est_names[1]] = ScoreEstimator(val_fpred[est_names[1]], val_freq, val_data['outputs'][:, 1:])
    
    PrintOutMetrics(est_names[1], val_scores[est_names[1]])
    

    #######################################################    
    ## GLM predictor
    ####################################################
    
    ## fit the general linear model by optimizing the poisson deviance
    print('Fitting the GLM by minimizing the poisson deviance')
    #poisson_glm = PoissonRegressor(alpha=1e-6, solver="newton-cholesky") #using this solver is much slow

    poisson_glm = PoissonRegressor(alpha=1e-6, solver="lbfgs")
    poisson_glm.fit(train_lin_inputs, train_freq.ravel(), sample_weight = train_data['outputs'][:, 1])
    
    val_fpred[est_names[2]] = np.reshape(poisson_glm.predict(val_lin_inputs), (-1, 1));

    val_scores[est_names[2]] = ScoreEstimator(val_fpred[est_names[2]], val_freq, val_data['outputs'][:, 1:])
    PrintOutMetrics(est_names[2], val_scores[est_names[2]])
    
    
    #######################################################
    ## Gradient boosted ensemble predictor
    #######################################################


    
    print('Fitting the Gradient Boosted Ensemble')
    poisson_gbrt = HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes = 32, max_iter = 150)
    poisson_gbrt.fit(train_gbrt_inputs, train_freq.ravel(), sample_weight = train_data['outputs'][:, 1])
    
    val_fpred[est_names[3]] = np.reshape(poisson_gbrt.predict(val_gbrt_inputs), (-1, 1));

    val_scores[est_names[3]] = ScoreEstimator(val_fpred[est_names[3]], val_freq, val_data['outputs'][:, 1:])
    PrintOutMetrics(est_names[3], val_scores[est_names[3]])
    

    #######################################################
    ## Neural network regressor
    ####################################################

    if do_train_nn:    
        print('Training the Neural Network')
    
        nn_train_outputs  = copy.deepcopy(train_data['outputs'])
        nn_train_outputs[:, 0:1]  = train_data['outputs'][:, 0:1] / train_data['outputs'][:, 1:2]

        nn_val_outputs  = copy.deepcopy(val_data['outputs'])
        nn_val_outputs[:, 0:1]  = val_data['outputs'][:, 0:1] / val_data['outputs'][:, 1:2]

    
        # parameters embeddings for the categorical and non-categorical 
        embed_params = {'dim': 8, 'cat_ind': [5,6,7,8], 'num_cats': num_cats, 'dis_ind': [0,1,2,3,4]}
    
        arch_params = {'n_hidden': 1024} # have just one hidden layer after the embedding layer
    
        opt_params = {'nb': 10192, 'lr': .01, 'n_epochs': 30, 'use_balanced_sampling': False, 'optimizer': 'adamw', 'weight_decay': 1e-4, 'balance ratio': [.95, .05]}

    
        [val_fpred['nn'], nn_model] =  nn_reg.train_network(embed_params, arch_params, opt_params,
                                                            {'inputs':  train_gbrt_inputs, 'outputs': nn_train_outputs},
                                                            {'inputs':  val_gbrt_inputs, 'outputs': nn_val_outputs})
        
        val_scores[est_names[4]] = ScoreEstimator(val_fpred[est_names[4]], val_freq, val_data['outputs'][:, 1:2])

        ## use to make predictions for the trained network
        # nn_reg.make_predictions(nn_model, val_gbrt_inputs)

        print('')
        PrintOutMetrics(est_names[4], val_scores[est_names[4]])

        print('Training the Balanced Neural Network')
        
        # My far from optimal solution to balancing:
        # use random sampling of training based on a weight for each example  
        # Set these weights via the "balance ratio" vector which has two numbers:
        # first = the total percentage weight given to the samples with zero claims (in the training set it is ~.93)
        # second = the total percentage weight to all the other samples (in the training set it is ~.07)        
        # This balancing reduces overall performance especially on the zero claims examples but does make the network more predict higher frequency rates
        
        opt_params = {'nb': 10192, 'lr': .01, 'n_epochs': 30, 'use_balanced_sampling': True, 'optimizer': 'adamw', 'weight_decay': 1e-4, 'balance ratio': [.9, .1]}
        [val_fpred['balanced nn'], bal_nn_model] =  nn_reg.train_network(embed_params, arch_params, opt_params,
                                                                     {'inputs': train_gbrt_inputs, 'outputs': nn_train_outputs},
                                                                     {'inputs': val_gbrt_inputs, 'outputs': nn_val_outputs})

        ## To make predictions for the trained network use
        # nn_reg.make_predictions(bal_nn_model, val_gbrt_inputs)
    
        val_scores[est_names[5]] = ScoreEstimator(val_fpred[est_names[5]], val_freq, val_data['outputs'][:, 1:2])

        print('')
        PrintOutMetrics(est_names[5], val_scores[est_names[5]])     
        
    

    
    
    #######################################################
    ## Print and visualize results on validation set
    #######################################################

    ## print the metrics in tabular form
    print_table_of_results(est_names, val_scores, 'Summary of results on validation data')
    print(' ')
    
    if display_results:
        
        ## display histogram of predictions (want to make sure not just estimating 0 for everything)
        pred_fig, pred_axs = plt.subplots(1, len(est_names)-1, sharey=True, tight_layout=True)
        ## don't bother showing the simple baseline predictor
        make_disp_hist(val_fpred, est_names[1:], pred_axs)

        ## compute and display lorenz curve
        lorenz_fig, lorenz_ax = plt.subplots(figsize=(8, 8))
        display_lorenz_curves(est_names, val_freq, val_fpred, val_data, lorenz_ax)    

    
    plt.show()
