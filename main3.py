# Import necessary libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np

# Load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img = int(input("Enter img no."))

import matplotlib.pyplot as plt
plt.imshow(x_test[img])
plt.show()
print("actual value:", y_test[img])



loaded_model = tf.keras.models.load_model('Model1')

# Make predictions using the loaded model
predictions = loaded_model.predict(x_test[img].reshape(-1, 28*28).astype('float32') / 255.0)
predicted_class = np.argmax(predictions)
print("Predicted table:", predictions)
print("Predicted number:", predicted_class)