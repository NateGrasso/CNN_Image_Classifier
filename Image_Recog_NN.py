"""
Name: Nathan Grasso
Purpose: Create a CNN to classify images using tensorflow and keras
Date: November 13, 2024
"""
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt


## Limit GPU memory consumption to avoid Out Of Memory errors

#Grab the available GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
#limit GPU so that tensorflow only uses what it needs
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Create a variable that points to our image or data folder
data_dir = 'data'
# image loop to find acceptable image extensions 
image_exists = ['jpeg', 'jpg', 'bmp', 'png']

##check whats in the data folder
    # print(os.listdir(data_dir))
##list out all images
    # print(os.listdir(os.path.join(data_dir, "Dogs")))

for image_class in os.listdir(data_dir):#looping through the folder for classes
    for image in os.listdir(os.path.join(data_dir, image_class)): #joining looping through each image
        image_path = os.path.join(data_dir, image_class, image)# 
        try:
            img = cv2.imread(image_path)#opens image as an numpy array
            tip = imghdr.what(image_path)#checks the extension
            if tip not in image_exists:
                print('Image not in the approved extension list {}'.format(image_path))
                os.remove(image_path)#if extension is not in list we remove it
        except Exception as e:
            print('Issue with image {}'.format(image_path))

##LOAD DATA
data = tf.keras.utils.image_dataset_from_directory('data')#keras will create the classes and resize the images for me with this line
data_iterator = data.as_numpy_iterator()# this allows access to the data we created in the above line of code so we can loop through it
batch = data_iterator.next()# this is grabbing one batch of data

# visualize the classes and images so we know what class belongs to 0 and 1
#fig, ax = plt.subplots(ncols=4, figsize=(20,20))#graph size
#for index, img in enumerate(batch[0][:4]):#grab the first four images in the batch
    #ax[index].imshow(img.astype(int))
    #ax[index].title.set_text(batch[1][index])
#plt.show() #explicit call for matplotlib to show the graph (0 = Cats, 1 = Dogs)

##PREPROCESSING DATA
#Scale Data (we want the value of the images to be between 0-1 not 0-255)
scale_data = data.map(lambda x,y: (x/255, y)) #x represents images, y represents labels
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

#checking if image works after being scaled
#fig, ax = plt.subplots(ncols=4, figsize=(20,20))
#for index, img in enumerate(batch[0][:4]):
    #ax[index].imshow(img)# since we scaled to 0-1 we dont need it to be an int
    #ax[index].title.set_text(batch[1][index])
#plt.show()

#Split data into Training and Testing (allocating data)

#print(len(data))

## PRE-PROCESSING DATA

#allocating data to our different training and testing functions
train_size = int(len(scale_data)* .76)
val_size = int(len(scale_data)*.2)
#test size is what we use after training to test model
test_size = int(len(scale_data)*.1)

#telling the model how to use the data we divided and allocated
train = scale_data.take(train_size)
val = scale_data.skip(train_size).take(val_size)
test = scale_data.skip(train_size-val_size).take(test_size)

## BUILDING DEEP LEARNING MODEL
model = Sequential()
# add in the layers
model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile('adam', loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
#model.summary()

#TRAIN THE MODEL

#log training 
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
#print(history.history)

#plot the performance of training
#fig = plt.figure()
#plt.plot(history.history['loss'], color = 'red', label='loss')
#plt.plot(history.history['val_loss'], color='blue', label = 'val_loss')
#fig.suptitle('Loss', fontsize=20)
#plt.legend(loc="upper left")
#plt.show()

#Evaluate results
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

#loop through the batches in the data set aside for test
for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
print(f"Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}")

#test the image classifier with images it has never seen before
img = cv2.imread('Dogtest.jpg')
resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

#turn single image into the correct shape, the model expects a batch so this will help
np.expand_dims(resize,0)
yhat = model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5:
    print("This image is of a Dog.")
else:
    print("This image is of a Cat.")

## SAVE THE MODEL
model.save(os.path.join('models', 'dogcatclassifier.h5'))