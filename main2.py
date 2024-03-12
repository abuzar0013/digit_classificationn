import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if len(physical_devices) > 1:
        tf.config.experimental.set_memory_growth(physical_devices[1], True)


# Check if GPU is availablek
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Check which device is being used for operations
print("Device:", tf.test.gpu_device_name())

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape) # 6000 images 28*28
#print(y_train.shape)


# import matplotlib.pyplot as plt
# plt.imshow(x_test[400])
# plt.show()
# print("actual value:", y_test[400])
# import sys
# sys.exit()

x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

########################################################################################################################

# sequential API very convenient but not very flexible

model =  keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation= 'relu'),
        layers.Dense(256, activation= 'relu'),
        layers.Dense(10),
    ]
)
########################################################################################################################

#usi ko likhne ka dosra tarika

# model  = keras.Sequential()
# model.add(keras.Input(shape=(28*28)))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(10))

########################################################################################################################

#Functional API(a bit more flexible)
# inputs = keras.Input(shape=(28*28))
# x = layers.Dense(512, activation= 'relu', name= 'first_layer')(inputs)
# x = layers.Dense(256, activation= 'relu',  name= 'second_layer')(x)
# outputs = layers.Dense(10, activation= 'softmax')(x)
# model = keras.Model(inputs=inputs, outputs=outputs)



# model = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
# features =  model.predict(x_train)
# for feature in features:
#     print(feature.shape)
# print(model.summary())
########################################################################################################################


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],

)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=1)


model.save('Model1')



########################################################################################################################


# import numpy as np
#
# # Make the prediction
# predictions = model.predict(x_test[400].reshape(-1, 28*28).astype('float32') / 255.0)
#
# # Get the predicted class
# predicted_class = np.argmax(predictions)
#
# print("Predicted number:", predicted_class)
