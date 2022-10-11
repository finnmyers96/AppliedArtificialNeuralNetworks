### CE359 - Hopfield ANN ###
# Import packages
import time
import math
import numpy as np
import pandas as pd
import cv2

# check how long code takes to run
start_time = time.time()

# Set Random Seed
np.random.seed(75)

# Construct data
# v1 = np.array([0, 0, 0, 0])
# v2 = np.array([1, 1, 0, 1])
# v3 = np.array([1, 0, 1, 0])
# data = np.array([v1, v2, v3])
#
# Create data for testing the algorithm on longer, 4 random patterns of 1s and 0's of length 20
data2 = np.array([])
for i in range(0, 10):
    a = np.ones(200)
    rand_split = np.random.randint(0, 200)
    a[:rand_split] = 0
    np.random.shuffle(a)
    data2 = np.append(data2, a)
data2 = np.reshape(data2, (10, 200))

# Define Learning rule function
def learning_rule(patterns):
    # Create an empty array for storing weights
    wts = np.array([])
    # Iterate through each column in the first pattern (i)
    for i in range(len(patterns[0])):
    # For every i observation, compare to all j observations in the same pattern
        for j in range(len(patterns[0])):
            # Define an empty list for storing the product of the learning rule for the ith and jth observation
            wts_temp = []
            P_v_i = 0
            P_v_j = 0
            # Execute the above loops for every pattern given
            for n in range(len(patterns)):
                # Define v as an individual pattern in your data
                v = patterns[n]
                # Apply the learning rule to the ith and jth observation of pattern n
                if v[i] == 0:
                    P_v_i = -1
                elif v[i] == 1:
                    P_v_i = 1
                if v[j] == 0:
                    P_v_j = -1
                elif v[j] == 1:
                    P_v_j = 1
                # Append the product of P_v_i and P_v_j to the wts_tmp list
                wts_temp.append(P_v_i * P_v_j)
                # print(wts_temp)
            # Append the sum of all products of P_v_i and P_v_j for each pattern to the wts array
            wts = np.append(wts, np.sum(wts_temp))
    # Reshape the wts array to an appropriately sized matrix, and fill the diagonol with 0s
    wts = np.reshape(wts, (int(math.sqrt(len(wts))), int(math.sqrt(len(wts)))))
    np.fill_diagonal(wts, 0)
    return (wts)

# Define the wts matrix for the original data
# wts = learning_rule(data)

# Create test data for the weight matrix by changing the values of 25% of the original arrays
# data_test = data
# for i in range(len(data_test)):
#     v = data_test[i]
#     if v[0] == 1:
#         v[0] = 0
#     else:
#         v[0] = 1

def performance_rule(v, wts, rule):
    # Set the predicted matrix equal to v_pred
    v_pred = np.dot(v, wts)

    # Three part performance rule
    if rule == "three_rule":
        for i in range(len(v_pred)):
            if v_pred[i] == 0:
                v_pred[i] = v_pred[i]
            elif v_pred[i] >= 0:
                v_pred[i] = 1
            else:
                v_pred[i] = 0
    # Two part learning rule
    elif rule == "two_rule":
        for i in range(len(v_pred)):
            if v_pred[i] >= 0:
                v_pred[i] = 1
            else:
                v_pred[i] = 0
    elif rule == "sigmoid":
        for i in range(len(v_pred)):
            v_pred[i] = 1/(1+np.exp(-v_pred[i]))
    else:
        print("invalid rule!")
    return(v_pred)

def stabilization(data, weights, rule):
    v_final_pred = np.array([])
    iterations = []
    for i in range(len(data)):
        v_1 = data[i]
        v_temp = np.array([])
        v_2 = np.array([])
        iterations_counter = 0
        first_pass = True
        while np.array_equal(v_1, v_2) == False:
            if rule == "three_rule":
                if first_pass:
                    v_temp = performance_rule(v_1, weights, "three_rule")
                    first_pass = False
                else:
                    v_temp = performance_rule(v_2, weights, "three_rule")
                v_1 = v_temp
                v_2 = performance_rule(v_temp, weights, "three_rule")
            elif rule == "sigmoid":
                if first_pass:
                    v_temp = performance_rule(v_1, weights, "sigmoid")
                    first_pass = False
                else:
                    v_temp = performance_rule(v_2, weights, "sigmoid")
                v_1 = v_temp
                v_2 = performance_rule(v_temp, weights, "sigmoid")
            elif rule == "two_rule":
                if first_pass:
                    v_temp = performance_rule(v_1, weights, "two_rule")
                    first_pass = False
                else:
                    v_temp = performance_rule(v_2, weights, "two_rule")
                v_1 = v_temp
                v_2 = performance_rule(v_temp, weights, "two_rule")
            else:
                print("Invalid Rule!")
            iterations_counter += 2
        v_final_pred = np.append(v_final_pred, v_2)
        iterations.append(iterations_counter)
    v_final_pred = np.reshape(v_final_pred, (len(data), len(data[0])))

    # Total difference in model predictions
    diff_arr = np.absolute(np.subtract(data, v_final_pred))
    return([v_final_pred, diff_arr, iterations])

# stabilized_preds_sigmoid = stabilization(data_test, wts, "sigmoid")
# stabilized_preds_tworule = stabilization(data_test, wts, "two_rule")
# stabilized_preds_threerule = stabilization(data_test, wts, "three_rule")
#
# # Predicted Vectors from original data
# pred_vectors1_sigmoid = stabilized_preds_sigmoid[0]
# pred_vectors1_tworule = stabilized_preds_tworule[0]
# pred_vectors1_threerule = stabilized_preds_threerule[0]

def result_dataframe(data, predictions, rule):
    """Returns a dataframe from the stabilized prediction vectors"""
    results_dict = {"Pattern Number": np.linspace(1, len(data), len(data)).tolist(),
                   "Learning Rule": [rule]*len(data),
                   "Accuracy (%)": [(1 - (np.sum(predictions[1][i]))/len(data[i])) * 100 for i in range(len(data))],
                   "Iterations": predictions[2]}
    results_df = pd.DataFrame(results_dict).set_index("Pattern Number")
    return(results_df)

# results1_tworule = result_dataframe(data, stabilized_preds_tworule, "Two Rule")
# results1_threerule = result_dataframe(data, stabilized_preds_threerule, "Three Rule")
# results1_sigmoid = result_dataframe(data, stabilized_preds_sigmoid, "Sigmoid")
#
# Define the weights matrix for the created data
wts2 = learning_rule(data2)

# Change 2 random values (10%) in each of the 4 patterns in data2
data_test2 = data2

for i in np.random.randint(75, 125, 20):
    for j in range(len(data_test2)):
        if data_test2[j][i] == 0:
            data_test2[j][i] = 1
        else:
            data_test2[j][i] = 0

stabilized_preds2_tworule = stabilization(data_test2, wts2, "two_rule")
stabilized_preds2_threerule = stabilization(data_test2, wts2, "three_rule")
stabilized_preds2_sigmoid = stabilization(data_test2, wts2, "sigmoid")

# Predicted Vectors from new data
pred_vectors2_sigmoid = stabilized_preds2_sigmoid[0]
pred_vectors2_tworule = stabilized_preds2_tworule[0]
pred_vectors2_threerule = stabilized_preds2_threerule[0]
# print(pred_vectors2_threerule[0])
# print(performance_rule(data_test2[0], wts2, "three_rule"))
#
# results2_tworule = result_dataframe(data_test2, stabilized_preds2_tworule, "Two Rule")
# results2_threerule = result_dataframe(data_test2, stabilized_preds2_threerule, "Three Rule")
# results2_sigmoid = result_dataframe(data_test2, stabilized_preds2_sigmoid, "Sigmoid")

# print("Two Rule Results")
# print(results2_tworule)
# print("Three Rule Results")
# print(results2_threerule)
# print("Sigmoid Results")
# print(results2_sigmoid)


### Image Recreation Test

# Read in Image and Convert to black and white
bonnie = cv2.imread(r'C:\Users\ghmye\Documents\UVM\CE359\HW2\bonnie.jpg', 2)
ret, bw_img = cv2.threshold(bonnie, 127, 255, cv2.THRESH_BINARY)
bw = cv2.threshold(bonnie, 127, 255, cv2.THRESH_BINARY)

# Show the initial binary image
cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert all black cells (value = 255) to have a value of 1 for input into Hopfield
for i in range(len(bw_img)):
    for j in range(len(bw_img[0])):
        if bw_img[i][j] == 255:
            bw_img[i][j] = 1

bonnie_wts = learning_rule(bw_img)

# Create test data by changing ~10% of values in binary image
bonnie_test = bw_img
for i in np.random.randint(0, 443, 44):
    for j in range(len(bonnie_test)):
        if bonnie_test[j][i] == 0:
            bonnie_test[j][i] = 1
        else:
            bonnie_test[j][i] = 0

results_bonnie = stabilization(bonnie_test, bonnie_wts, "three_rule")
bonnie_pred_img = results_bonnie[0]

# Convert 1's back to 255's to show the predicted image
for i in range(len(bonnie_pred_img)):
    for j in range(len(bonnie_pred_img[0])):
        if bonnie_pred_img[i][j] == 1:
            bonnie_pred_img[i][j] = 255

cv2.imshow("Binary", bonnie_pred_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



print("time elapsed: {:.2f}s".format(time.time() - start_time))










