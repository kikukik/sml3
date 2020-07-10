# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:07:00 2020

@author: No-Pa
"""


import numpy as np
from matplotlib import pyplot as plt

lin_reg_train = np.loadtxt("lin_reg_train.txt")


lin_reg_test = np.loadtxt("lin_reg_test.txt")


# evaluate gaussian function
def gaussian_density_function(x, mu, sigma):
    x = np.array(x)
    c1 = 1 / np.sqrt((2*np.pi)**2 * np.linalg.det(sigma))
    c2 = np.exp(-0.5 * np.matmul((x - mu), np.matmul(np.linalg.inv(sigma),(x - mu)).T))
    y = c1 * c2
    return y.item()

def preprocess_X(X):
    ones_vector = np.ones(len(X))
    X_new = np.column_stack((X, ones_vector))
    return X_new.T

def lin_ridge_regression(X_train, y_train, lam):
    X_train_new = preprocess_X(X_train)
    x_x_T = np.matmul(X_train_new, X_train_new.T)
    I = np.eye(len(x_x_T))
    lambda_I = lam * I
    invers = np.linalg.inv(x_x_T + lambda_I)
    w = np.matmul(np.matmul(invers, X_train_new), y_train)
    return w

def solution_value(X, w):
    X_new = preprocess_X(X)
    return np.matmul(X_new.T,w)
    
    
def root_mean_squared_error(X_test ,w ,y_test):
    y_pred = solution_value(X_test, w)
    difference = np.linalg.norm(y_pred - y_test)
    return difference

def stack(X_1, degree):
    X = X_1
    for i in range(2,degree+1):
        X = np.column_stack((X,np.power(X_1,i)))
    return X


def five_fold_cross_validation(train_data, test_data):
    X_1 = train_data[0:10]
    X_2 = train_data[10:20]
    X_3 = train_data[20:30]
    X_4 = train_data[30:40]
    X_5 = train_data[40:50]


    # Use subsets 1 - 4 to train your model with polynomial features of degrees 2, 3 and 4.
    degrees = [2,3,4]
    subset_5 = [np.concatenate((X_1,X_2,X_3,X_4), axis = 0), X_5]
    subset_4 = [np.concatenate((X_1,X_2,X_3,X_5), axis = 0), X_4]
    subset_1 = [np.concatenate((X_2,X_3,X_4,X_5), axis = 0), X_1]
    subset_2 = [np.concatenate((X_3,X_4,X_5,X_1), axis = 0), X_2]
    subset_3 = [np.concatenate((X_4,X_5,X_1,X_2), axis = 0), X_3]
    
    
    
    
    subsets = [subset_1, subset_2, subset_3, subset_4, subset_5]
    

    for degree in degrees:
        RMSE_test = 0
        cross_RMSE_train = 0
        cross_RMSE_test = 0
        for subset in subsets:
            # define train and test set
            cross_validation_X_train = subset[0][:,0]
            cross_validation_y_train = subset[0][:,1]
            cross_validation_X_test = subset[1][:,0]
            cross_validation_y_test = subset[1][:,1]
            lam = 0.01
        
            
            cross_validation_X_train_new = stack(cross_validation_X_train, degree)
            
            w = lin_ridge_regression(cross_validation_X_train_new, cross_validation_y_train, lam)
            
            # RMSE for cross_validation
            cross_RMSE_train = cross_RMSE_train + root_mean_squared_error(cross_validation_X_train_new, w, cross_validation_y_train)
            
            cross_validation_X_test_new = stack(cross_validation_X_test, degree)
            
            cross_RMSE_test = cross_RMSE_test + root_mean_squared_error(cross_validation_X_test_new, w, cross_validation_y_test)
            
            #RMSE for test data
            X_test = test_data[:,0]
            y_test = test_data[:,1]
            
            X_test_new = stack(X_test, degree)
            
            RMSE_test = RMSE_test + root_mean_squared_error(X_test_new, w, y_test)
        print("degree: ", degree, "Average Train RMSE: ", cross_RMSE_train/5, "Average Validation RMSE:", cross_RMSE_test/5, "Average Test RMSE: ", RMSE_test/5)
        
def bayesian_linear_ridge_regression_mu(X,x,y):
    alpha = 100
    beta = 100
    
    X_new = preprocess_X(X)
    x_x_T = np.matmul(X_new, X_new.T)
    I = np.eye(len(x_x_T))
    lambda_I = (alpha/beta) * I
    invers = np.linalg.inv(x_x_T + lambda_I)
    w = np.matmul(np.matmul(invers, X_new), y_train)
    if type(x) != int:
        x_new = preprocess_X(x)
    else:
        x_new = np.asarray([x, 1])
    mu = np.matmul(x_new.T,w)
    return mu

def rmse_bayesian(X_train,y_train, data_set_x, data_set_y):
    
    y_pred = bayesian_linear_ridge_regression_mu(X_train, data_set_x, y_train)
    difference = np.linalg.norm(y_pred - data_set_y)
    return difference

def log_likehood_bayesian(X_train, y_train, data_set_x, data_set_y):
    alpha = 100
    beta = 100
    
    mu = bayesian_linear_ridge_regression_mu(X_train, data_set_x, y_train)
    
    X_new = preprocess_X(X_train)
    x_x_T = np.matmul(X_new, X_new.T)
    print(x_x_T.shape)
    print(x_x_T)
    I = np.eye(len(x_x_T))
    alpha_I = alpha * I
    beta_x_x_T = beta * x_x_T
    invers = np.linalg.inv(alpha_I + beta_x_x_T)
    if type(data_set_x) != int:
        data_set_x_new = preprocess_X(data_set_x)
    else:
        data_set_x_new = np.asarray([data_set_x, 1])
    print(invers.T)
    print(np.matmul(invers, data_set_x_new.T))
    #sigma_pre = np.matmul(np.matmul(data_set_x_new.T, invers),data_set_x_new)
    #sigma = 1/beta + sigma_pre
    #print(sigma)
        
    
        
    
if __name__ == "__main__":
    # 1a
    w = lin_ridge_regression(lin_reg_train[:,0], lin_reg_train[:,1], 0.01)
    print(w)
    y_pred = solution_value(lin_reg_test[:,0], w)
    difference_1a = root_mean_squared_error(lin_reg_test[:,0], w, lin_reg_test[:,1])
    
    #plot
    plt.scatter(lin_reg_train[:,0], lin_reg_train[:,1], c = "black")
    
    x = np.linspace(-1,1,100) # 100 linearly spaced numbers
    y_pred = solution_value(x,w)
    
    plt.plot(x,y_pred, c="blue")
    
    
    # bias wird in linear_regression ergänzt
    #1b
    X_train = lin_reg_train[:,0]
    y_train = lin_reg_train[:,1]
    X_test = lin_reg_test[:,0]
    y_test = lin_reg_test[:,1]
    degree = 4
    
    X_train_new = stack(X_train, degree)
    w1 = lin_ridge_regression(X_train_new, y_train, 0.01)
    print("w for polynomial case: ", w1)
    difference_1b_train = root_mean_squared_error(X_train_new, w1, y_train)
    print("Root mean squared error Train", difference_1b_train)
    
    X_test_new = stack(X_test, degree)
    difference_1b_test = root_mean_squared_error(X_test_new, w1, y_test)
    print("Root squared mean error Test", difference_1b_test)
    
    x = np.linspace(-1,1,100) # 100 linearly spaced numbers
    X_1 = stack(x, degree)
    y_pred = solution_value(X_1, w1)
    
    plt.scatter(lin_reg_train[:,0], lin_reg_train[:,1], c = "black")
    plt.plot(x,y_pred, c="blue")
    
    # Model is called linear regression, since we still do linear regression but in an higher dimensional space
    
    #1c
    
    five_fold_cross_validation(lin_reg_train, lin_reg_test)
    
    #1e
    
    X_train = lin_reg_train[:,0]
    y_train = lin_reg_train[:,1]
    X_test = lin_reg_test[:,0]
    y_test = lin_reg_test[:,1]
    
    bayesian_linear_ridge_regression_mu(X_train,[1, 2],y_train)
    
    # Report the RMSE of the train and test data under your Bayesian model (use the predictive mean)
    
    print("RMSE Bayesian_training: ", rmse_bayesian(X_train,X_train, X_train, y_train))
    
    print("RMSE Bayesian Test: ",rmse_bayesian(X_train, y_train, X_test, y_test))
    
    # # Report the average log-likelihood of the train and test data under your Bayesian model.
    
    # print(log_likehood_bayesian(X_train, y_train, [1, 2], y_test))
    
    
    
    
    