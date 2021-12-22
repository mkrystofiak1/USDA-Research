import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.preprocessing import image

import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
from keras import callbacks
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

print("Using Tensorflow version " + tf.__version__)

## ----- Visualization Functions -----

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1 )
    plt.plot(epochs,train_metrics)
    plt.plot(epochs,val_metrics)
    plt.title("Training and Validation " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, "val_"+metric])
    plt.show()

def plot_lr(history):
    learning_rate = history.history['lr']
    epochs = range(1, len(learning_rate) + 1)
    plt.plot(epochs, learning_rate)
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.show()

def get_test_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0)
    print('accuracy: {acc:0.3f}'.format(acc=test_acc))

## ----- Data Preprocessing -----

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    '/home/mitchell/USDA_Project/dataset/',
    target_size=(120,120),
    batch_size=32,
    class_mode='binary',
    save_to_dir='/home/mitchell/USDA_Project/augment'
)

test_datagen = ImageDataGenerator(
        rescale=1./255
)

test_set = test_datagen.flow_from_directory(
    '/home/mitchell/USDA_Project/dataset/test_set',
    target_size=(120,120),
    batch_size=32,
    class_mode='binary'
)


## ----- Build CNN -----

CNN = tf.keras.models.Sequential()

CNN.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[120,120,3]))
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

CNN.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

CNN.add(tf.keras.layers.Flatten())

CNN.add(tf.keras.layers.Dense(units=128, activation='relu'))

CNN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

## ----- Model Compile -----

CNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

## ----- Training the CNN -----

CNN.fit(x=training_set, validation_data=test_set, epochs=10)

## ----- Single Predition -----

test_image = image.load_img('/home/mitchell/USDA_Project/dataset/single_prediction/fish1.png',
        target_size=(120,120))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)

result = CNN.predict(test_image/255.0)

training_set.class_indices

if result[0][0] > 0.5:
    prediction = 'shellfish'
else:
    prediction = 'fish'

print(prediction)
