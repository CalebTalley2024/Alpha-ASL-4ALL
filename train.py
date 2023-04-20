# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# %%
# Show one of the images from the training dataset
img_index = 60
# x_train.shape

# %%
# plt.imshow(x_train[img_index])

# %%
# normalize data dimensions
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# make training, validation, and testing sets

# split the train data into train and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=2)

# %%
# use Sequential model API
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# %%
# configure the learning process
model.compile(loss='sparse_categorical_crossentropy', # use sparse cross entropy since your data is integer encoded, NOT hot-encoded (0s and 1s)
              optimizer='adam',
              metrics=['accuracy'])

# %%
# print(tf.config.list_physical_devices('GPU'))

# %%

# Check if GPU is available and set device accordingly
if tf.test.gpu_device_name():
    print('GPU found')
    device_name = tf.test.gpu_device_name()
else:
    print("No GPU found, using CPU instead")
    device_name = '/cpu:0'

# Create a TensorFlow session and set it to use the specified device
with tf.device(device_name):
    # Your TensorFlow code here



    # train the model
    model.fit(x_train,
              y_train,
              batch_size=64,
              epochs=10,
              validation_data=(x_valid, y_valid), verbose = 1,
              callbacks=None)
    # the number on the left below Epochs is number batch you are on.

# %%
# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])

# %%
model.predict(x_test)

# %%
# x_train.shape

# %%


