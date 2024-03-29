import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)


# Modify activation_functions.py
# Import necessary libraries (already imported in the script)
import numpy as np
import matplotlib.pyplot as plt

# Define input range
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Apply activation functions to the input data
relu_output = relu(np.array(random_values))
leaky_relu_output = leaky_relu(np.array(random_values))
tanh_output = tanh(np.array(random_values))

# Plot the output of each activation function
plt.plot(random_values, relu_output, label='ReLU')
plt.plot(random_values, leaky_relu_output, label='Leaky ReLU')
plt.plot(random_values, tanh_output, label='Tanh')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Activation Functions Output')
plt.legend()
plt.show()
