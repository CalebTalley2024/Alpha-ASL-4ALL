# %%
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %%
def convert_folder_images(folder_name):

    # specify the folder path
    folder_path = f"alphabet_data/{folder_name}"

    # initialize an empty list to store the image arrays
    image_arrays = []

    # loop through each file in the folder
    for filename in os.listdir(folder_path):
        # check that the file is an image file (e.g. JPEG, PNG)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # construct the full file path
            file_path = os.path.join(folder_path, filename)
            # read the image file as a numpy array
            img_array = cv2.imread(file_path)

            #gray the images
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) # from 3 channels to 1 channel
            # coverting to gray scale gets rid of one dimension, so we add it back in, this time with 1 channel
            # Add an extra dimension to the grayscale image
            img_array = np.expand_dims(img_array, axis=-1)

            # add the image array to the list
            image_arrays.append(img_array)

    # convert the list of image arrays to a numpy array
    image_arrays = np.array(image_arrays)
    return image_arrays

# %%
def get_all_images_and_labels():
    a = convert_folder_images("a")
    # make labels for all photos based on how many images are in the folder
    # the folder names gives you the label for each folder of images

    b = convert_folder_images("b")
    c = convert_folder_images("c")
    d = convert_folder_images("d")
    e = convert_folder_images("e")

    labels_a = np.full(len(a),"a")
    labels_b = np.full(len(b),"b")
    labels_c = np.full(len(c),"c")
    labels_d = np.full(len(d),"d")
    labels_e = np.full(len(e),"e")
    # print(labels_e.shape)
    images = np.vstack((a,b,c,d,e))/255 # divide by 255 for normalization
    # labels = np.append(labels_a, (labels_b, labels_c, labels_d, labels_e))

    # Convert string labels to numerical labels using LabelEncoder
    label_encoder = LabelEncoder()
    # concatenate all the labels together
    labels = np.concatenate(
        (labels_a, labels_b, labels_c, labels_d, labels_e), axis=0)
    labels = label_encoder.fit_transform(labels)


    return images,labels


# %%
images,labels = get_all_images_and_labels()
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=2)
# img_index = 60
# x_train.shape
# plt.imshow(x_train[img_index])
# normalize data dimensions
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# make training, validation, and testing sets

# split the train data into train and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=2)

#

# %%

# use Sequential model API
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(300, 300,1)))
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
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(300, 300,1))
        self.pool1 = MaxPooling2D(pool_size=2)
        self.dropout1 = Dropout(0.3)
        self.conv2 = Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')
        self.pool2 = MaxPooling2D(pool_size=2)
        self.dropout2 = Dropout(0.3)
        self.flatten = Flatten()
        self.dense1 = Dense(256, activation='relu')
        self.dropout3 = Dropout(0.5)
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        return x#%%
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %%
def convert_folder_images(folder_name):

    # specify the folder path
    folder_path = f"alphabet_data/{folder_name}"

    # initialize an empty list to store the image arrays
    image_arrays = []

    # loop through each file in the folder
    for filename in os.listdir(folder_path):
        # check that the file is an image file (e.g. JPEG, PNG)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # construct the full file path
            file_path = os.path.join(folder_path, filename)
            # read the image file as a numpy array
            img_array = cv2.imread(file_path)

            #gray the images
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) # from 3 channels to 1 channel
            # coverting to gray scale gets rid of one dimension, so we add it back in, this time with 1 channel
            # Add an extra dimension to the grayscale image
            img_array = np.expand_dims(img_array, axis=-1)

            # add the image array to the list
            image_arrays.append(img_array)

    # convert the list of image arrays to a numpy array
    image_arrays = np.array(image_arrays)
    return image_arrays

# %%
def get_all_images_and_labels():
    a = convert_folder_images("a")
    # make labels for all photos based on how many images are in the folder
    # the folder names gives you the label for each folder of images

    b = convert_folder_images("b")
    c = convert_folder_images("c")
    d = convert_folder_images("d")
    e = convert_folder_images("e")

    labels_a = np.full(len(a),"a")
    labels_b = np.full(len(b),"b")
    labels_c = np.full(len(c),"c")
    labels_d = np.full(len(d),"d")
    labels_e = np.full(len(e),"e")
    # print(labels_e.shape)
    images = np.vstack((a,b,c,d,e))/255 # divide by 255 for normalization
    # labels = np.append(labels_a, (labels_b, labels_c, labels_d, labels_e))

    # Convert string labels to numerical labels using LabelEncoder
    label_encoder = LabelEncoder()
    # concatenate all the labels together
    labels = np.concatenate(
        (labels_a, labels_b, labels_c, labels_d, labels_e), axis=0)
    labels = label_encoder.fit_transform(labels)


    return images,labels


# %%
images,labels = get_all_images_and_labels()
images.shape

# %%
labels.shape

# %%
one_image = images[5]
one_image[:][:][:].shape

# %%
# plt.imshow(one_image)

# %%

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=2)
# img_index = 60
# x_train.shape
# plt.imshow(x_train[img_index])
# normalize data dimensions
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# make training, validation, and testing sets

# split the train data into train and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=2)

#

# %%
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(300, 300,1))
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        return x
    # @tf.autograph.experimental.do_not_convert




# %%
model = MyModel()
# configure the learning process
model.compile(loss='sparse_categorical_crossentropy',
              # use sparse cross entropy since your data is integer encoded, NOT hot-encoded (0s and 1s)
              optimizer='adam',
              metrics=['accuracy']),
# print(tf.config.list_physical_devices('GPU'))



# %%

# Check if GPU is available and set device accordingly
if tf.test.gpu_device_name():
    print('GPU found')
    device_name = tf.test.gpu_device_name()
else:
    print("No GPU found, using CPU instead")
    device_name = '/cpu:0'

# %%

# configure the learning process
model.compile(loss='sparse_categorical_crossentropy',
              # use sparse cross entropy since your data is integer encoded, NOT hot-encoded (0s and 1s)
              optimizer='adam',
              metrics=['accuracy']),
# print(tf.config.list_physical_devices('GPU'))

# %%
@tf.autograph.experimental.do_not_convert
def fit_model():
    model.fit(x_train,
              y_train,
              batch_size=50,
              epochs=10,
              validation_data=(x_valid, y_valid), verbose=1,
              callbacks=None)

# %%

# Create a TensorFlow session and set it to use the specified device

with tf.device(device_name):
    # Your TensorFlow code here

    # train the model
    fit_model()
    # @tf.autograph.experimental.do_not_convert
    # model.fit(x_train,
    #           y_train,
    #           batch_size=50,
    #           epochs=10,
    #           validation_data=(x_valid, y_valid), verbose=1,
    #           callbacks=None)
 # the number on the left below Epochs is number batch you are on.
# @tf.autograph.experimental.do_not_convert


# %%
# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])
# model.predict(x_test)


# %%



