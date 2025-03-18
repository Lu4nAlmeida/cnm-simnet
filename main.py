import matplotlib.pyplot as plt
import numpy as np
import time

def forward(NN, B):
    cache.clear()  # clear cache
    activation = B
    for i, layer in enumerate(NN):
        Z = np.matmul(layer[0], activation) + layer[1]  # multiply weights and add biases

        # activation function
        if i != (len(NN)-1):
            activation = ReLU(Z)
        else:
            activation = Softmax(Z)

        cache.append((Z, activation))  # save Z and activation values into cache
    return activation

def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_derivative(Z):
    return (Z > 0).astype(float)

def Softmax(array):
    exp_values = np.exp(array - np.max(array, axis=0, keepdims=True))  # Stabilize for each column
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)  # Normalize for each column

def cost(input, predicted):
    expected = actual_output(input)
    return np.average((predicted - expected)**2)

def actual_output(input_matrix):
    input_matrix = np.transpose(input_matrix)
    output = np.zeros_like(input_matrix)
    for i, input in enumerate(input_matrix):
        if input[0] < input[1]:
            output[i][0] = 1
        else:
            output[i][1] = 1
    return np.transpose(output)

def backward(NN, B, predicted, learning_rate):
    expected = actual_output(B)
    m = B.shape[1]  # batch size

    # Compute the derivative of the cost function with respect to the output layer
    dZ = (predicted - expected) * 2 / m  # Mean squared error derivative

    for i in reversed(range(len(NN))):
        weights, biases = NN[i]
        Z, activation = cache[i]
        prev_activation = B if i == 0 else cache[i-1][1]

        # Compute gradients for weights and biases
        dW = np.matmul(dZ, prev_activation.T)
        db = np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases
        NN[i][0] -= learning_rate * dW
        NN[i][1] -= learning_rate * db

        # Propagate the error backward
        if i > 0:
            dZ = np.matmul(weights.T, dZ) * ReLU_derivative(cache[i-1][0])


# Neural network initialization
W1 = np.random.uniform(-1, 1, (2, 2))
b1 = np.random.uniform(-1, 1, (2, 1))

W2 = np.random.uniform(-1, 1, (2, 2))
b2 = np.random.uniform(-1, 1, (2, 1))

W3 = np.random.uniform(-1, 1, (2, 2))
b3 = np.random.uniform(-1, 1, (2, 1))

NeuralNetwork = [[W1, b1],
                 [W2, b2],
                 [W3, b3]]

cache = []

# Data Generation
dataset = np.random.rand(100, 2) * 10
data_train = np.transpose(dataset[:80])
data_test = np.transpose(dataset[80:])

# Training
batches = data_train.reshape(8, 2, 10)
learning_rate = 0.001

for epoch in range(1000):
    for batch in batches:
        output = forward(NeuralNetwork, batch)
        cost_value = cost(batch, output)
        backward(NeuralNetwork, batch, output, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Cost: {cost_value}")

# Visualization
# Blue -> point is below y=x (bottom neuron)
# Red -> point is above y=x (upper neuron)
color = {0: "blue", 1: "red"}

x = data_test[0]
y = data_test[1]

figure, axis = plt.subplots(2)

start_time = time.perf_counter()

# Inference
output = forward(NeuralNetwork, data_test)

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Inference computed in {elapsed_time:.9f} seconds.")

# Predicted Outputs
colors = []
for prediction in output[0]:
    colors.append(color[round(prediction)])
plt.xlim(0,10)
plt.ylim(0,10)
axis[0].scatter(x, y, c=colors)
axis[0].set_title("Predictions")

# Actual Outputs
colors = []
for prediction in actual_output(data_test)[0]:
    colors.append(color[prediction])
axis[1].scatter(x, y, c=colors)
axis[1].set_title("Expected")

plt.show()