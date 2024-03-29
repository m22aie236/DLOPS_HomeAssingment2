# feature1.py
import numpy as np
import matplotlib.pyplot as plt

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

sigmoid_output = sigmoid(np.array(random_values))

plt.plot(random_values, sigmoid_output)
plt.xlabel('Input')
plt.ylabel('Sigmoid Output')
plt.title('Sigmoid Activation Function')
plt.show()
