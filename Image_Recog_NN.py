# Name: Nathan Grasso
# Date: 12/5/2023
# Purpose: Create a CNN that can identify images of Cats and Dogs. Make the model Object Oriented for future use.

import tensorflow as tf
import keras 
from keras import models, layers, metrics
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

class ImageClassifier:
    def __init__(self, data_dir='data', image_exists=['jpeg', 'jpg', 'bmp', 'png']):
        self.data_dir = data_dir
        self.image_exists = image_exists
        self.model = self.build_model()

    def limit_gpu_memory(self):
        #limit GPU memory to avoid OOM errors
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def filter_unsupported_images(self):
        #loop through folder for classes
        for image_class in os.listdir(self.data_dir):
            for image in os.listdir(os.path.join(self.data_dir, image_class)):
                image_path = os.path.join(self.data_dir, image_class, image)
                try:
                    img = cv2.imread(image_path)#opens image as numpy array
                    tip = imghdr.what(image_path)#checks extension
                    if tip not in self.image_exists:
                        print(f'Image not in the approved extension list {image_path}')
                        os.remove(image_path)#remove image if not in list
                except Exception as e:
                    print(f'Issue with image {image_path}')

    def load_data(self):
        #keras creates classes and will resize the image for me
        data = tf.keras.utils.image_dataset_from_directory(self.data_dir)
        return data

    def preprocess_data(self, data):
        #we want our data to be between 0-1 not 0-255
        #x represents images and y represents labels 
        scale_data = data.map(lambda x, y: (x / 255, y))
        return scale_data

    def split_data(self, scale_data):
        #We will split our data into train, validation, and test (the majority will be in train)
        train_size = int(len(scale_data) * .76)
        val_size = int(len(scale_data) * .2)
        test_size = int(len(scale_data) * .1)
        
        #telling the model how to use our data
        train = scale_data.take(train_size)
        val = scale_data.skip(train_size).take(val_size)
        test = scale_data.skip(train_size + val_size).take(test_size)

        return train, val, test

    def build_model(self):
        model = models.Sequential()
        # adding the layers of the model
        model.add(keras.layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Conv2D(32, (3, 3), 1, activation='relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Conv2D(16, (3, 3), 1, activation='relu'))
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

        return model

    def train_model(self, train_data, val_data, epochs=20):
        #log our training epochs
        logdir = 'logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        history = self.model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[tensorboard_callback])
        return history

    def evaluate_model(self, test_data):
        pre = tf.keras.metrics.Precision()
        re = tf.keras.metrics.Recall()
        acc = tf.keras.metrics.BinaryAccuracy()

        for batch in test_data.as_numpy_iterator():
            x, y = batch
            yhat = self.model.predict(x)
            pre.update_state(y, yhat)
            re.update_state(y, yhat)
            acc.update_state(y, yhat)

        precision = pre.result().numpy()
        recall = re.result().numpy()
        accuracy = acc.result().numpy()

        return precision, recall, accuracy

    def test_single_image(self, image_path):
        img = cv2.imread(image_path)
        resize = tf.image.resize(img, (256, 256))
        plt.imshow(resize.numpy().astype(int))
        plt.show()

        #turn a single image into the correct shape
        np.expand_dims(resize, 0)
        yhat = self.model.predict(np.expand_dims(resize / 255, 0))

        if yhat > 0.5:
            print(f"This image is of a Dog. {yhat}")
        else:
            print(f"This image is of a Cat.{yhat}")

    def save_model(self, model_path='models/dogcatclassifier.h5'):
        self.model.save(model_path)

# run model on individual image:
classifier = ImageClassifier()
classifier.limit_gpu_memory()
classifier.filter_unsupported_images()
data = classifier.load_data()
scaled_data = classifier.preprocess_data(data)
train_data, val_data, test_data = classifier.split_data(scaled_data)
history = classifier.train_model(train_data, val_data)
precision, recall, accuracy = classifier.evaluate_model(test_data)
print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")
classifier.test_single_image("Cattest2.jpeg")
#classifier.save_model()