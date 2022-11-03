### Counterpropagation ###
# Harrison Myers
# 11/1/2022

# Import packages
import math
import numpy as np


# Set random seed
np.random.seed(44)

def activation_max(S):
    """
    Sigmoidal activation function
    :param S: dot product of weights and input pattern
    :return: return f(S)
    """
    f_s = np.zeros(len(S))
    for i in range(len(S)):
        if S[i] == np.max(S):
            f_s[i] = 1
        else:
            f_s[i] = 0
    return f_s

def activation_min(S):
    """
    Sigmoidal activation function
    :param S: dot product of weights and input pattern
    :return: return f(S)
    """
    f_s = np.zeros(len(S))
    for i in range(len(S)):
        if S[i] == np.min(S):
            f_s[i] = 1
        else:
            f_s[i] = 0

    return f_s

def counterpropagation(train, target=0, n_hidden_nodes=6, cal_wts_ij=0, cal_wts_jk=0, threshold=0.1, alpha=0.7, beta=0.1, max_iter=1000, activation="max", cal=False):
    """
    Executes the backpropagation artificial neural network on testing data to try and identify patterns and predict the
    corresponding testing data. Uses Root-Mean-Squared Error as the evaluation metric.
    :param train: Testing data. Dataframe, array, or similar object
    :param target: Target data, not applicable if using to predict values. Dataframe, array, or similar object
    :param n_hidden_nodes: Number of hidden nodes in middle layer
    :param cal_wts_ij: Calibrated weight_ij matrix
    :param cal_wts_jk: Calibrated weight_jk matrix
    :param threshold: Threshold for RMSE
    :param alpha: A learning coefficient for ij, set at 0.7 for this HW
    :param beta: A learning coefficient for jk, set at 0.1 for this HW
    :param max_iter: Maximum number of allowable iterations
    :param activation: Tells model whether the minimum or maximum S_j is determined as the "winner"
    :return: A plot of RMSE, a plot the derivative of the activation function multiplied by the difference between the
    predicted value and target value, and the final set of weights.
    """
    # Create random weight matrices if not calibrated
    if cal == False:
        wts_ij = np.random.rand(len(train.loc[0]), n_hidden_nodes)
        wts_jk = np.random.rand(n_hidden_nodes, len(target.loc[0]))
        # Create a list storing target values
        targets = [np.array(target.iloc[i, :]) for i in range(len(target))]

    # If the network has been trained, set weights to trained weight matrices
    elif cal == True:
        wts_ij = cal_wts_ij
        wts_jk = cal_wts_jk

    # Create list iterations to count the number of iterations needed to achieve the threshold
    iterations = [0]

    # The whole function is run inside a while loop that runs until the RMSE threshold is met
    RMSE = [0.2]

    # Execute training while RMSE is under a threshold if training the model
    # If the model is already trained, only execute process once
    if cal == False:
        while RMSE[-1] > threshold or iterations[-1] < max_iter:
            # Create an empty list for storing output values
            outputs = []

            for i in range(len(train)):
                # Set your input pattern
                v_i = np.array(train.iloc[i, :])
                target = targets[i]

                # Calculate S_j by computing the dot product of v_i with its respective weights in wts_ij
                S_j = np.dot(v_i, wts_ij)

                # Calculate the distance between the input vectors and their associated weights
                v_i_dist = np.array([np.linalg.norm(v_i - wts_ij[:, i]) for i in range(len(wts_ij[0]))])

                # Calculate a_j by passing the respective S_j values through the activation function
                if activation == "max":
                    a_j = activation_max(v_i_dist)
                elif activation == "min":
                    a_j = activation_min(v_i_dist)
                else:
                    print("invalid activation input, must be 'min' or 'max'!")

                # Calculate a_k by computing the dot product of a_j with wts_jk (not passing through activation function to avoid premature training of weights)
                a_k = np.dot(a_j, wts_jk)
                outputs.append(a_k)

                # Update weights_ij
                for j in range(len(a_j)):
                    for i in range(len(wts_ij)):
                        if a_j[j] == 1:
                            wts_ij[:, j] = wts_ij[:, j] + alpha * (v_i - wts_ij[:, j])
                        else:
                            wts_ij[i, j] = wts_ij[i, j]

                # Update weights_jk
                for k in range(len(a_k)):
                    for j in range(len(wts_jk)):
                        if a_j[j] == 1:
                            wts_jk[j, k] = wts_jk[j, k] + beta*(target[k] - a_k[k])
                        else:
                            wts_jk[j, k] = wts_jk[j, k]

            # Calculate RMSE
            # Calculate (t - o)^2
            t_minus_o_squared = np.sum([(targets[i] - outputs[i])**2 for i in range(len(targets))])
            RMSE.append(math.sqrt(t_minus_o_squared)/(len(outputs)*(len(train))))

            # Add to the total number of iterations
            iterations.append(iterations[-1]+1)
    else:
        # Create an empty list for outputs
        outputs = []

        for i in range(len(train)):
            # Set your input pattern
            v_i = np.array(train.iloc[i, :])
            print(v_i)

            # Calculate S_j by computing the dot product of v_i with its respective weights in wts_ij
            # S_j = np.dot(v_i, wts_ij)

            # Calculate the distance between the input vectors and their associated weights
            v_i_dist = np.array([np.linalg.norm(v_i - wts_ij[:, i]) for i in range(len(wts_ij[0]))])

            # Calculate a_j by passing the respective S_j values through the activation function
            if activation == "max":
                a_j = activation_max(v_i_dist)
            elif activation == "min":
                a_j = activation_min(v_i_dist)
            else:
                print("invalid activation input, must be 'min' or 'max'!")

            # Calculate a_k by computing the dot product of a_j with wts_jk (not passing through activation function to avoid premature training of weights)
            a_k = np.dot(a_j, wts_jk)
            outputs.append(a_k)

    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.plot(iterations, RMSE)
    # # ax.scatter(iterations, RMSE, c="red")
    # ax.set(title="RMSE vs. Iterations",
    #        xlabel="Number of Iterations",
    #        ylabel="Root-Mean-Square-Error")
    # plt.tight_layout(pad=2.0)
    # plt.show()

    return({"Iterations": iterations,
            "RMSE": RMSE,
            "Weights_IJ": wts_ij,
            "Weights_JK": wts_jk,
           "Final_Predictions": outputs})
