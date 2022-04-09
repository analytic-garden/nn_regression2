#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
rnn_regression.py - a simple recurrent neural net rergression using PyTorch
author: Bill Thompson
license: GPL 3
copyright: 2022-04-04
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import copy

import torch
import torch.utils.data as Data

from RNet import RNet

def GetArgs():
    """
    GetArgs - return arguments from command line. 
    Use -h to see all arguments and their defaults.

    Returns
    -------
    argparse object
        parser object
    """    
    def ParseArgs(parser):
        class Parser(argparse.ArgumentParser):
            def error(self, message):
                sys.stderr.write('error: %s\n' % message)
                self.print_help()
                sys.exit(2)

        parser = Parser(description='A simple neural net rergression using PyTorch')

        parser.add_argument('input_file1',
                            help="""
                            Input CSV file 1 (required).
                            Must contain columns Year and Temperature_Anomaly
                            """)
        parser.add_argument('input_file2',
                            help="""
                            Input CSV file 2 (required).
                            Must contain columns Year and CO2_concentrations
                            """)
        parser.add_argument('-o', '--output_path',
                            help="Output file csv file. (optional)",
                            required = False,
                            type = str)
        parser.add_argument('-b', '--batch_size',
                            required = False,
                            default = 32,
                            type=int,
                            help='Batch size (default = 32)')
        parser.add_argument('-e', '--epochs',
                            required=False,
                            type=int,
                            default = 200,
                            help='Number epochs Default=200')
        parser.add_argument('-l', '--layers',
                            required=False,
                            type=int,
                            default = 1,
                            help='Number of RNN layers. Default=1')
        parser.add_argument('-n', '--hidden_size',
                            required=False,
                            type=int,
                            default = 10,
                            help='Size of RNN hidden layer.  Default=10')
        parser.add_argument('-p', '--predict_size',
                            required=False,
                            type=int,
                            default = 10,
                            help='Number of input rows left out for prediction.  Default=10')
        parser.add_argument('-r', '--learning_rate',
                            required=False,
                            type=float,
                            default = 0.01,
                            help='Learning rate Default = 0.01')
        parser.add_argument('-w', '--workers',
                            required=False,
                            type=int,
                            default = 2,
                            help='Number of processes for Dataloader Default = 2')
        parser.add_argument('-s', '--rand_seed',
                            required=False,
                            type=int,
                            help='Random mumber generator seed.')
    
        return parser.parse_args()

    parser = argparse.ArgumentParser(description='A simple neural net rergression using PyTorch')
    args = ParseArgs(parser)

    return args

def read_data(input_file1, input_file2, predict_size = 10):
    """
    read_data - read data for regression

    Parameters
    ----------
    input_file1 : str
        path to input csv file
    input_file2 : str
        path to input csv file
    predict_size : int
        number of data points to reserve for prediction

    Returns
    -------
    tuple
        x, y : tensors 
            data for regression from input_file1 and input_file2
        x_predict, y_predict
            data for prediction        
        X_data, Y1_data, Y2_data : numpy arrays
            original data 

    Requires
    --------
    input_file1 must contain columns Year and Temperature_Anomaly
    input_file2 must contain columns Year and CO2.concentrations
    df1.shape[0] == df2.shape[0]
    predict_size > 0
    """    
    df1 = pd.read_csv(input_file1)
    df1 = df1.sort_values(by = 'Year')

    df2 = pd.read_csv(input_file2)
    df2 = df2.sort_values(by = 'Year')

    if df1.shape[0] != df2.shape[0]:
        raise ValueError(f'{input_file1} and {input_file2} must have the same number of rows.')

    # scale data and get tensors
    X_data = df1['Year']
    Y1_data = df1['Temperature_Anomaly']
    Y2_data = df2['CO2.concentrations']

    X_data_scale = (X_data - np.mean(X_data)) / np.std(X_data)
    Y1_data_scale = (Y1_data - np.mean(Y1_data)) / np.std(Y1_data)
    Y2_data_scale = (Y2_data - np.mean(Y2_data)) / np.std(Y2_data)

    x = torch.Tensor(np.array(X_data_scale[:-predict_size])).unsqueeze(dim = 1)
    y = torch.Tensor(np.array([Y1_data_scale[:-predict_size], Y2_data_scale[:-predict_size]])).t()

    x_predict = torch.Tensor(np.array(X_data_scale[-predict_size:])).unsqueeze(dim = 1)
    y_predict = torch.Tensor(np.array([Y1_data_scale[-predict_size:], Y2_data_scale[-predict_size:]])).t()

    return x, y, x_predict, y_predict, X_data, Y1_data, Y2_data

def unscale_predictions(prediction, future_prediction, Y1, Y2):
    """
    unscale_predictions - rescale data back to original dimensions

    Parameters
    ----------
    prediction : Tensor
        Y1 predictions
    future_prediction : Tensor
        Y2 predictions
    Y1 : numpy array
        original data
    Y2 : numpy array
        original data

    Returns
    -------
    tuple
        prediction1, prediction2, future_prediction1, future_prediction2 : numpy arrays
            rescaled predictions of original data and future data
    """
    prediction1 = prediction.data.numpy()[:, 0] * np.std(Y1) + np.mean(Y1)
    future_prediction1 = future_prediction.data.numpy()[:, 0] * np.std(Y1) + np.mean(Y1)

    prediction2 = prediction.data.numpy()[:, 1] * np.std(Y2) + np.mean(Y2)
    future_prediction2 = future_prediction.data.numpy()[:, 1] * np.std(Y2) + np.mean(Y2)

    return prediction1, prediction2, future_prediction1, future_prediction2

def plot_results(X, Y1, Y2,
                 predict_size,
                 prediction, future_prediction,
                 output_path):
    """
    plot_results - show some plots

    Parameters
    ----------
    X : numpy array
        original data
    Y1 : numpy array
         original data
    Y2 : numpy array
         original data
    predict_size : int
        number of prediction steps
    prediction : numpy array
        prediction for X[:-predict_size]
    future_prediction : numpy arrray
        prediction for X[-predict_size:]
    output_path : str
        directory for output
    """
    prediction1, prediction2, future_prediction1, future_prediction2 = \
            unscale_predictions(prediction, future_prediction, Y1, Y2)

    fig, ax = plt.subplots()
    ax.set_title('Neural Net Regression Analysis')
    ax.set_xlabel('X1')
    ax.set_ylabel('Y1')
    ax.scatter(X, Y1, s = 2, label = 'Data1')
    ax.plot(X, np.concatenate((prediction1, future_prediction1)))  # to get rid of gap
    ax.plot(X[:-predict_size], prediction1, label = 'Regression')
    ax.plot(X[-predict_size:], future_prediction1, label = 'Prediction')
    plt.legend()
    if output_path is not None:
        fig.savefig(output_path + 'figure1.png')
    plt.show(block = False)

    fig, ax = plt.subplots()
    ax.set_title('Neural Net Regression Analysis')
    ax.set_xlabel('X2')
    ax.set_ylabel('Y2')
    ax.scatter(X, Y2, s = 2, label = 'Data2')
    ax.plot(X, np.concatenate((prediction2, future_prediction2)))
    ax.plot(X[:-predict_size], prediction2, label = 'Regression')
    ax.plot(X[-predict_size:], future_prediction2, label = 'Prediction')
    plt.legend()
    if output_path is not None:
        fig.savefig(output_path + 'figure2.png')
    plt.show(block = False)

def write_settings(input_file1, input_file2,
                   predict_size,
                   output_path,
                   batch_size, epochs,
                   learning_rate, 
                   hidden_size,
                   workers,
                   rand_seed,
                   r2, min_loss):
    """
    write_setting - output setting to file or stdout

    Parameters
    ----------
    input_file1 : str
        data file
    input_file2 : str
        data file
    predict_size : int
        number of steps to predict
    output_path : str
        path to save data
    batch_size : int
        size of batches for regression
    epochs : int
        number of epochs
    learning_rate : float
        learing rate for NN
    hidden_size : int
        size of RNN hidden layer
    workers : int
        number of parallel processes for regression
    rand_seed : int
        rng seed
    r2 : float
        R^2 of best model
    min_loss : float
        loss value of best model
    """                  
    if output_path is None:
        f = sys.stdout
    else:     
        out_file = output_path + 'settings.txt'
        f = open(out_file, 'w')

    print('input_file1 =', os.path.abspath(input_file1), file = f)
    print('input_file2 =', os.path.abspath(input_file2), file = f)
    if output_path is None:
        print('output_path =', 'None', file = f)
    else:
        print('output_path =', os.path.abspath(output_path), file = f)
    print('predict_size =', predict_size, file = f)
    print('batch_size =', batch_size, file = f)
    print('epochs =', epochs, file = f)
    print('learning_rate =', learning_rate, file = f)
    print('hidden_size =', hidden_size, file = f)
    print('workers =', workers, file = f)
    print('rand_seed =', rand_seed, file = f)
    print('R^2 =', r2, file = f)
    print('Best model loss =', min_loss, file = f)

    if output_path is not None:
        f.close()

def main():
    # get command line args
    args = GetArgs()
    input_file1 = args.input_file1
    input_file2 = args.input_file2
    output_path = args.output_path
    batch_size = args.batch_size
    epochs = args.epochs
    num_layers = args.layers
    lr = args.learning_rate
    hidden_size = args.hidden_size
    workers = args.workers
    predict_size = args.predict_size
    rand_seed = args.rand_seed

    if output_path is not None:
        if output_path != '/':
            output_path += '/'

    if rand_seed is None:
        rand_seed = int(time.time())
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    # read some data
    x, y, x_predict, y_predict, X_data, Y1_data, Y2_data = read_data(input_file1, input_file2, 
                                                                     predict_size = predict_size)
    # construct the model
    net = RNet(hidden_size = hidden_size, num_layers = num_layers)

    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    loss_func = torch.nn.SmoothL1Loss()  # Smooth loss - less ensitive to outliers

    # set up data for loading
    dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset = dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = workers,)

    # start training
    best_model = None
    min_loss = sys.maxsize
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
            prediction = net(batch_x)

            loss = loss_func(prediction, batch_y)
            print('epoch:', epoch, 'step:', step, 'loss:', loss.data.numpy())
            if loss.data.numpy() < min_loss:
                min_loss = loss.data.numpy()
                best_model = copy.deepcopy(net)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

    # get a regression fit
    best_model.eval()
    prediction = best_model(x)

    r2 = r2_score(y.data.numpy(), prediction.data.numpy())

    # predict the future
    future_predict = best_model(torch.cat((x, x_predict)))[-predict_size:, :]

    # save thye settings
    write_settings(input_file1, input_file2,
                   predict_size,
                   output_path,
                   batch_size, epochs,
                   lr, 
                   hidden_size,
                   workers,
                   rand_seed,
                   r2, min_loss)

    plot_results(X_data, 
                 Y1_data, Y2_data,
                 predict_size,
                 prediction, future_predict, 
                 output_path)
                   
    if output_path is not None:
        prediction1, prediction2, future_prediction1, future_prediction2 = \
            unscale_predictions(prediction, future_predict, Y1_data, Y2_data)

        df = pd.DataFrame({'x': x.data[:, 0].tolist(),
                           'y': y.data[:, 0].tolist(),
                           'y': y.data[:, 1].tolist(),
                           'Prediction1': [p[0] for p in prediction.data.numpy()],
                           'Prediction2': [p[1] for p in prediction.data.numpy()],
                           'prediction_rescale1': prediction1,
                           'prediction_rescale2': prediction2,
                           'X': X_data[:-predict_size],
                           'Y1': Y1_data[:-predict_size],
                           'Y2': Y2_data[:-predict_size]
                           })
        df.to_csv(output_path + 'results.csv', index = False)

        if x_predict is not None:
            df2 = pd.DataFrame({'X': x_predict.data[:, 0].tolist(),
                                'Y': y_predict.data[:, 0].tolist(),
                                'Prediction1': [p[0] for p in future_predict.data.numpy()],
                                'Prediction2': [p[1] for p in future_predict.data.numpy()],
                                'prediction_rescale1': future_prediction1,
                                'prediction_rescale2': future_prediction2,
                                'X': X_data[-predict_size:],
                                'Y1': Y1_data[-predict_size:],
                                'Y2': Y2_data[-predict_size:]
                                })
            df2.to_csv(output_path + 'results2.csv', index = False)

        # save the best_model
        torch.save(best_model.state_dict(), output_path + 'model.trc')
    
    input('Press <Enter> to continue')
        
if __name__ == "__main__":
    main()
