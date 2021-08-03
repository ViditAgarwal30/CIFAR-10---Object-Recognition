# Import modules
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras
from keras.layers import Dropout
from keras.layers import BatchNormalization

#loading the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# buliding the cnn
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))



#compile the cnn
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#training the cnn
model.fit(train_images, train_labels, epochs=200, 
                    validation_data=(test_images, test_labels))



#prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('ab.jpeg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image/255.0)
print(result)
#train_images.class_indices
classes=['airplane','automobile','bird','cat','dear','dog','frog','horse','ship','truck']
print(classes[np.argmax(result)])


