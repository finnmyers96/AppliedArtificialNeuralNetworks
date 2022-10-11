### CE 359 - Backpropagation Script ###
# Import packages
import time
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# check how long code takes to run
start_time = time.time()

# Set random seed
np.random.seed(44)

# Set working directory to CE359 folder (take out for submission)
os.chdir(r'C:\Users\ghmye\OneDrive\Documents\UVM\Courses\CE359\HW1')

# Read in training and target data
training_data = pd.read_csv("TrainingData.csv", header=None)
target_data = pd.read_csv("TargetData.csv", header=None)

# Concatenate data column wise (resulting df is of dimensions: (nrows, ncols x 2))
data = pd.concat([training_data, target_data], axis=1)

# Shuffle rows of data randomly so they feed into backpropagration function randomly
data = data.sample(frac=1).reset_index(drop=True)

# Split data back into training and target data
training_data = data.iloc[:, :4]
target_data = data.iloc[:, 4:]

# Create set of random weights from 0 to 1
wts = np.random.rand(len(training_data) + 2, len(training_data) + 1)
wts_jk = wts.transpose()

def activation(S):
    """
    Sigmoidal activation function
    :param S: dot product of weights and input pattern
    :return: return f(S)
    """
    f_s = 1/(1+math.exp(-S))
    return f_s

def activation_deriv(S):
    """
    :param S: Activation function
    :return: derivative of activation function
    """
    f_s_prime = (1/(1+math.exp(-S)))*(1- (1/(1+math.exp(-S))))
    return f_s_prime

def backpropagation(train, target, wts_ij, wts_jk, threshold, learning_coef=0.5, max_iter=100, cal=False):
    """
    Executes the backpropagation artificial neural network on testing data to try and identify patterns and predict the
    corresponding testing data. Uses Root-Mean-Squared Error as the evaluation metric.
    :param train: Testing data. Dataframe, array, or similar object
    :param target: Target data. Dataframe, array, or similar object
    :param wts_ij: Random matrix of weights of shape (n_hidden_nodes, n_inputs).
    :param threshold: Target threshold RMSE value. ANN will execute until this value is reached.
    :param learning_coef: A constant, set at 0.5 for this HW.
    :return: A plot of RMSE, a plot the derivative of the activation function multiplied by the difference between the
    predicted value and target value, and the final set of weights.
    """
    # Create list iterations to count the number of iterations needed to achieve the threshold
    iterations = [0]

    # The whole function is run inside a while loop that runs until the RMSE threshold is met
    RMSE = [0.5]

    while RMSE[-1] >= threshold:
        # Create an empty list for storing output and target values
        outputs = []
        targets = []
        # Create a new a_i for each row in the training data and loop through the process with each row
        for i in range(len(train)):
            # Set your inputs pattern and append your targets to the targets list used to calculate delta values
            a_i = train.loc[i]
            t = target.loc[i]
            targets.append(t)

            # Calculate S_j by computing the dot product of a_i with its respective weights in wts_ij
            S_j = np.array(list(np.dot(wts_ij[i], a_i) for i in range(len(wts_ij))))

            # Calculate a_j by passing the respective S_j values through the activation function
            a_j = np.array(list(activation(i) for i in S_j))

            # Calculate S_k by computing the dot product of a_j with its respective weights in wts_jk
            S_k = np.array(list(np.dot(wts_jk[i], a_j) for i in range(len(wts_jk))))

            # Calculate a_k by passing the respective S_k values through the activation function
            a_k = np.array(list(activation(i) for i in S_k))
            outputs.append(a_k)

            # Calculate delta values for back-error propagation
            # Delta_k calculation
            delta_k = np.array(list(((t[i] - a_k[i])*activation_deriv(S_k[i]) for i in range(len(t)))))

            # Delta_j calculation
            #Transpose wts_jk for multiplication
            wts_jk = wts_jk.transpose()
            delta_j = np.array(list((np.dot(delta_k, wts_jk[i]) * activation_deriv(S_j[i])) for i in range(len(wts_jk))))

            # Calculate the delta_wts_jk and update the wts_jk value
            # Transform delta_k and define a_j's shape explicitly
            delta_k = np.reshape(delta_k, (len(delta_k), 1))
            a_j = np.reshape(a_j, (1, len(a_j)))

            # Untranspose wts_jk for next loop and for matrix addition
            wts_jk = wts_jk.transpose()

            # Calculate delta_wts_jk and add to original wts_jk matrix
            delta_wts_jk = learning_coef * np.dot(delta_k, a_j)
            wts_jk = np.add(wts_jk, delta_wts_jk)

            # Calculate the delta_wts_ij and update the wts_ij value
            # Reshape delta_j and define a_i's shape explicitly for dot product calculation
            delta_j = np.reshape(delta_j, (len(delta_j), 1))
            a_i = np.array(a_i).reshape((1, len(a_i)))
            delta_wts_ij = learning_coef * np.dot(delta_j, a_i)
            wts_ij = np.add(wts_ij, delta_wts_ij)

        # Calculate RMSE
        # Calculate (t - o)^2
        t_minus_o_squared = list(((targets[i] - outputs[i])**2) for i in range(len(targets)))
        RMSE.append(math.sqrt(np.sum(t_minus_o_squared))/(len(t_minus_o_squared)**2))

        # Add to the total number of iterations
        iterations.append(iterations[-1]+1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(iterations, RMSE)
    ax.scatter(iterations, RMSE, c="red")
    ax.set(title="RMSE vs. Iterations",
           xlabel="RMSE",
           ylabel="Number of Iterations")
    # for i in range(len(iterations)):
    #     plt.annotate(iterations[i], (iterations[i], RMSE[i] + 0.01))
    plt.tight_layout(pad=2.0)
    # plt.show()
    plt.savefig(r'C:\Users\ghmye\OneDrive\Documents\UVM\Courses\CE359\HW1\RMSE_initial.png')
    return({"N_Iterations": iterations[-1],
            "Final_RMSE": round(RMSE[-1], 6),
            "Weights_IJ": wts_ij,
            "Weights_JK": wts_jk})

def backpropagation_bm(train, target, wts_ij, wts_jk, threshold, learning_coef=0.5, max_iter=100, momentum_coef=0.9, cal=False, RMSE_init=0.5):
    """
    Executes the backpropagation artificial neural network on testing data to try and identify patterns
    and predict the corresponding testing data. Uses Root-Mean-Squared Error as the evaluation metric.
    :param train: Testing data. Dataframe, array, or similar object
    :param target: Target data. Dataframe, array, or similar object
    :param wts_ij: Random matrix of weights of shape (n_hidden_nodes, n_inputs).
    :param threshold: Target threshold RMSE value. ANN will execute until this value is reached.
    :param learning_coef: A constant, set at 0.5 for this HW.
    :param momentum_coef: A constant, set at 0.9 for this HW. Positively correlated with rate of weight
    change.
    :return: Number of iterations, the final RMSE value, a plot of RMSE, and the final set of weights.
    """

    # Add a row of 1's to weight matrices
    if cal == False:
        wts_ij = np.hstack((wts_ij, np.ones((len(wts_ij), 1))))
        wts_jk = np.hstack((wts_jk, np.ones((len(wts_jk), 1))))
        train.loc[:, len(train) + 1] = np.ones(len(train[0]))
    else:
        wts_ij = wts_ij
        wts_jk = wts_jk

    # Create list iterations to count the number of iterations needed to achieve the threshold
    iterations = [0]

    # The whole function is run inside a while loop that runs until the RMSE threshold is met
    RMSE = [RMSE_init]

    while RMSE[-1] >= threshold or iterations[-1] < max_iter:
        # Create an empty list for storing output and target values
        outputs = []
        targets = []

        # store the original weight matrices before they are updated for momentum calculation
        wts_ij_og = wts_ij
        wts_jk_og = wts_jk

        # Create a new a_i for each row in the training data and loop through the process with each row
        for i in range(len(train)):
            # Set your inputs pattern and append your targets to the targets list used to calculate delta values
            a_i = train.loc[i]
            t = target.loc[i]
            targets.append(t)

            # Calculate S_j by computing the dot product of a_i with its respective weights in wts_ij
            S_j = np.array(list(np.dot(wts_ij[i], a_i) for i in range(len(wts_ij))))
            S_j = np.append(S_j, 1)

            # Calculate a_j by passing the respective S_j values through the activation function
            a_j = np.array(list(activation(i) for i in S_j))

            # Calculate S_k by computing the dot product of a_j with its respective weights in wts_jk
            S_k = np.array(list(np.dot(wts_jk[i], a_j) for i in range(len(wts_jk))))

            # Calculate a_k by passing the respective S_k values through the activation function
            a_k = np.array(list(activation(i) for i in S_k))
            outputs.append(a_k)

            # Calculate delta values for back-error propagation
            # Delta_k calculation
            delta_k = np.array(list(((t[i] - a_k[i])*activation_deriv(S_k[i]) for i in range(len(t)))))

            # Delta_j calculation
            #Transpose wts_jk for multiplication
            wts_jk = wts_jk.transpose()
            delta_j = np.array(list((np.dot(delta_k, wts_jk[i]) * activation_deriv(S_j[i])) for i in range(len(wts_jk))))

            # Calculate the delta_wts_jk and update the wts_jk value
            # Transform delta_k and define a_j's shape explicitly
            delta_k = np.reshape(delta_k, (len(delta_k), 1))
            a_j = np.reshape(a_j, (1, len(a_j)))

            # Untranspose wts_jk for next loop and for matrix addition
            wts_jk = wts_jk.transpose()

            # Calculate delta_wts_jk and add to original wts_jk matrix
            delta_wts_jk = learning_coef * np.dot(delta_k, a_j)
            wts_jk = np.add(wts_jk, delta_wts_jk)

            # Calculate the delta_wts_ij and update the wts_ij value
            # Reshape delta_j and define a_i's shape explicitly for dot product calculation
            delta_j = np.reshape(delta_j, (len(delta_j), 1))
            a_i = np.array(a_i).reshape((1, len(a_i)))

            # Drop last element of delta_j because it is associated with the bias and throws off dimensions
            delta_j = delta_j[:(len(delta_j)-1)]

            delta_wts_ij = learning_coef * np.dot(delta_j, a_i)
            wts_ij = np.add(wts_ij, delta_wts_ij)

        # Calculate the average change in the last iteration, and add on momentum change
        delta_wts_ij_avg = np.subtract(wts_ij, wts_ij_og) / len(train)
        delta_wts_jk_avg = np.subtract(wts_jk, wts_jk_og) / len(train)
        wts_ij = np.add(wts_ij, (momentum_coef * delta_wts_ij_avg))
        wts_jk = np.add(wts_jk, (momentum_coef * delta_wts_jk_avg))

        # Calculate RMSE
        # Calculate (t - o)^2
        t_minus_o_squared = list(((targets[i] - outputs[i])**2) for i in range(len(targets)))
        RMSE.append(math.sqrt(np.sum(t_minus_o_squared))/(len(t_minus_o_squared)**2))

        # Add to the total number of iterations
        iterations.append(iterations[-1]+1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(iterations, RMSE)
    ax.scatter(iterations, RMSE, c="red")
    ax.set(title="RMSE vs. Iterations",
           xlabel="RMSE",
           ylabel="Number of Iterations")
    # for i in range(len(iterations)):
    #     plt.annotate(iterations[i], (iterations[i], RMSE[i] + 0.01))
    plt.text(0.75, 0.9,
             "Final RMSE = " + str(round(RMSE[-1], 4)),
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    plt.text(0.75, 0.85,
             "Learning Coefficient  = " + str(round(learning_coef, 1)),
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    plt.text(0.75, 0.8,
             "Momentum Coefficient = " + str(round(momentum_coef, 1)),
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    plt.tight_layout(pad=2.0)
    # plt.savefig(r'C:\Users\ghmye\OneDrive\Documents\UVM\Courses\CE359\HW1\RMSE_bias_momentum.png')
    plt.show()
    return({"N_Iterations": iterations[-1],
            "Final_RMSE": round(RMSE[-1], 6),
            "Outputs": outputs,
            "Weights_IJ": wts_ij,
            "Weights_JK": wts_jk})

# backprop_1 = backpropagation(training_data, target_data, wts, wts_jk, 0.1)
backprop_2 = backpropagation_bm(training_data, target_data, wts, wts_jk, 0.1)

wts_ij_calibrated = backprop_2.get("Weights_IJ")
wts_jk_calibrated = backprop_2.get("Weights_JK")
final_RMSE = backprop_2.get("Final_RMSE")

# Test the backpropagation code on the same data using the set weights
# backpropagation_bm(training_data, target_data, wts_ij_calibrated, wts_jk_calibrated, 0.1, cal=True)

print("time elapsed: {:.2f}s".format(time.time() - start_time))

# Testing #
# Add a random value of -0.05 to 0.05 to all the training and testing data and see how the algorithm performs
# rand_delta_1 = np.random.uniform(-0.05, 0.05, (3, 5))
# rand_delta_2 = np.random.uniform(-0.05, 0.05, (3, 4))
#
# X_test = training_data + rand_delta_1
# y_test = target_data + rand_delta_2

# backpropagation_bm(X_test, y_test, wts_ij_calibrated, wts_jk_calibrated, 0.1, cal=True, max_iter=20, RMSE_init=final_RMSE)
#
# learning_coefs = np.linspace(.1, 1, 10)
#
# # for i in learning_coefs:
# #     backpropagation_bm(training_data, target_data, wts, wts_jk, 0.1, learning_coef=i)
#
# momentum_coefs = np.linspace(.1, 1, 10)

# for i in momentum_coefs:
#     backpropagation_bm(training_data, target_data, wts, wts_jk, 0.1, momentum_coef=i)