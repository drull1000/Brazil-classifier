#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
data_dir = 'data'
image_exts = ['jpeg','jpg','png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print(f"Image not in list {image_path}")
                os.remove(image_path)
        except Exception as e:
            print(f"Issue with image {image_path}")

data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
scaled = batch[0] / 255

scaled.max()

data = data.map(lambda x,y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()

batch = scaled_iterator.next()
batch[0].max()

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
model = load_model(os.path.join('models', 'brazilclassifier.h5'))
loss, acc = model.evaluate(train, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
print(model.predict(train).shape)
model_checkpoint=tf.keras.callbacks.ModelCheckpoint('CIFAR10{epoch:02d}.h5',period=2,save_weights_only=False)
hist = model.fit(train, epochs=7, validation_data=val, callbacks=[model_checkpoint], shuffle=True)
model.save((os.path.join('models', './brazilclassifier')))

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x,y=batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y,yhat)
print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
img = cv2.imread('wallpaper.jpg')
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat)
if yhat > 0.5:
    print("Probably not macaco")
else:
    print("Probably macaco")

for batch in test.as_numpy_iterator():
    x,y=batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y,yhat)
print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
import cv2
img = cv2.imread('braziltest.jpg')
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat)

if yhat > 0.5:
    print("Probably not macaco")
else:
    print("Probably macaco")

