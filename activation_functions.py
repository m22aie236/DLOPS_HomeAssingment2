import numpy as np
import matplotlib.pyplot as plt
import feature1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)



# Modify activation_functions.py
relu_output = relu(np.array(random_values))
leaky_relu_output = leaky_relu(np.array(random_values))
tanh_output = tanh(np.array(random_values))

plt.plot(random_values, relu_output, label='ReLU')
plt.plot(random_values, leaky_relu_output, label='Leaky ReLU')
plt.plot(random_values, tanh_output, label='Tanh')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Activation Functions Output')
plt.legend()
plt.show()
