import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from kerastuner.tuners import RandomSearch
from keras import callbacks
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler 
import glob
from glob import glob
import os
import re
import math

import PIL
from PIL import Image

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

## ----- Fish Images Training and Testing Directories -----

fish1_path = "/home/mitchell/USDA_Project/species_1/sample"
fish2_path = "/home/mitchell/USDA_Project/species_2/sample"
fish3_path = "/home/mitchell/USDA_Project/species_3/sample"
fish4_path = "/home/mitchell/USDA_Project/species_4/sample"
fish5_path = "/home/mitchell/USDA_Project/species_5/sample"

test1_path = "/home/mitchell/USDA_Project/species_1/test"
test2_path = "/home/mitchell/USDA_Project/species_2/test"
test3_path = "/home/mitchell/USDA_Project/species_3/test"
test4_path = "/home/mitchell/USDA_Project/species_4/test"
test5_path = "/home/mitchell/USDA_Project/species_5/test"

def search_dir(parent):
    file_name = parent + '/**/*.png'
    files = glob(file_name, recursive=True)
    return files

fish1_images = search_dir(fish1_path)
fish2_images = search_dir(fish2_path)
fish3_images = search_dir(fish3_path)
fish4_images = search_dir(fish4_path)
fish5_images = search_dir(fish5_path)

test1_images = search_dir(test1_path)
test2_images = search_dir(test2_path)
test3_images = search_dir(test3_path)
test4_images = search_dir(test4_path)
test5_images = search_dir(test5_path)



def combine_data(full_image_list, full_label_list, new_img_list, new_label):
    for file in new_img_list:
        img = Image.open(file).resize((128,128))
        img_arr = np.asarray(img)
        full_image_list.append(img_arr)
        full_label_list.append(new_label)
    return full_image_list, full_label_list

def unison_shuffled_copies(I, L):
    assert len(I) == len(L)
    p = np.random.permutation(len(I))
    return I[p], L[p]

images = []
labels = []
images_1, labels_1 = combine_data(images, labels, fish1_images, 0)
images_2, labels_2 = combine_data(images_1, labels_1, fish2_images, 1)
images_3, labels_3 = combine_data(images_2, labels_2, fish3_images, 2)
images_4, labels_4 = combine_data(images_3, labels_3, fish4_images, 3)
images_5, labels_5 = combine_data(images_4, labels_4, fish5_images, 4)

## ----- Names of Fish Images -----

CATEGORIES = ['Dascyllus reticulatus ', 'Myripristis kuntee ', 'Hemigymnus fasciatus ', 'Neoniphon sammara ', 'Lutjanus fulvus ']
category_to_index = dict((name,index) for index,name in enumerate(CATEGORIES))
category_to_index

## ----- Convert Images to Numpy Arrays and Normalize

fish_dataset = np.array(images_5)
label_dataset = np.array(labels_5)
fish_dataset, label_dataset = unison_shuffled_copies(fish_dataset, label_dataset)
split = int(len(fish_dataset)*.1)

train_img = np.array(fish_dataset)[split:]
train_lbl = np.array(label_dataset)[split:]
test_img  = np.array(fish_dataset)[0:split]
test_lbl  = np.array(label_dataset)[0:split]

train_img = train_img.astype('float32')
test_img  = test_img.astype('float32')

train_img /= 255
test_img /= 255

## ----- Keras Callback Declarations -----

## Early Stopping
early_stopping = EarlyStopping(
        monitor='val_loss', # value to monitor as performance measure: validation loss
        patience=3, # number of epochs with no improvement
        restore_best_weights= True, # 
        mode='min'
        )

## CSV Logging

csv_log = CSVLogger(
        "results.csv",
        separator=',',
        append=False
        )

## Model Checkpoint

checkpoint_path = 'model_checkpoints/'

model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq='epoch',
        save_weights_only=True,
        verbose=1
        )

#Load Weights ==> new_model.load_weights('model_checkpoints/')

## Reduce LR On Plateau

reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.001,
        verbose=2
        )

## Learning Rate Scheduler

def lr_decay(epoch, lr):
    if epoch != 0 and epoch % 5 == 0:
        return lr * 0.2
    return lr

learning_rate_scheduler = LearningRateScheduler(
        lr_decay,
        verbose=1
        )

## Lambda Callback ==> Used to build custom callbacks, Not currently defined

## Other callbacks ==> Callback, TensorBoard, Remote Monitor

## ----- Build Model Function for the Keras Tuner -----

def build_model(hp):

    #hp.Choice("layer name", [choice 1, choice 2, choice 3,...])
    #for i in range(hp.Int("Conv layers", min_value=0, max_value=5)):


    model = keras.Sequential()
    
    for i in range(hp.Int("Conv layers", min_value=0, max_value=4)):
        model.add(keras.layers.Conv2D(hp.Choice(f"Conv layer_{i}_filters", [32,64,128]), kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Dropout(hp.Choice(f"Dropout {i}", [0.1,0.2,0.3,0.4,0.5])))

    

    #model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
    #activation='relu', input_shape=(128,128,3)))
    
    #model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    #model.add(keras.layers.Dropout(hp.Choice("dropout", [0.1, 0.2, 0.3])))

    #model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same',
    #activation='relu'))

    #model.add(keras.layers.MaxPooling2D(pool_size=(2,2))),
    #model.add(keras.layers.Dropout(0.5))

    #model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same',
    #activation='relu'))

    #model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    #model.add(keras.layers.Dropout(0.5))

    #model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same',
    #activation='relu'))

    #model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    #model.add(keras.layers.Dropout(0.5))

    #model.compile(optimizer='adam',
    #        loss=keras.losses.SparseCategoricalCrossentropy(),
    #        metrics=['accuracy'])

    return model

## ----- Best Model Declaration from Tuner Trials ----- 

model1 = keras.Sequential([

    keras.layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(128,128,3)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.1),
    
    keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(0.2),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.25),

    keras.layers.Dense(5, activation='softmax')])

model2 = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(128,128,3)),
    keras.layers.MaxPool2D(pool_size=2, strides=2),
    keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    keras.layers.MaxPool2D(pool_size=2, strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=1, activation='softmax')])


## ----- Model Compiler ----- 

model1.compile(optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

## ----- Model Train -----

print("Training model:")
print()
history = model1.fit(train_img, train_lbl,
        epochs=20)  #, callbacks=[early_stopping])

## ----- Model Viewing -----

plot_metric(history, 'loss')
plot_metric(history, 'accuracy')

## ----- Model Test -----

print("Testing trained model:")
print()
model1.evaluate(test_img, test_lbl)

##plt.style.use('dark_background')
#model.evaluate(test_img, test_lbl, batch_size = 1, verbose = 1)

def make_labels( new_img_list, new_label):
    images = []
    labels = []
    for file in new_img_list:
        img = Image.open(file).resize((128,128))
        img_arr = np.asarray(img)
        images.append(img_arr)
        labels.append(new_label)
    return images, labels

images = []
labels = []
#print("Make labels function")
images_test1, labels_test1 = make_labels(test1_images, 0)
images_test2, labels_test2 = make_labels(test2_images, 1)
images_test3, labels_test3 = make_labels(test3_images, 2)
images_test4, labels_test4 = make_labels(test4_images, 3)
images_test5, labels_test5 = make_labels(test5_images, 4)

## ----- Tuner Declaration -----

tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=1,
        directory='best_models/6'
        )

## ----- Tuner Compilation and Training  -----

#tuner.search(train_img, train_lbl, validation_data=(test_img, test_lbl),
#        epochs=10, batch_size=32)

## ----- Tuner Test -----

#print("Testing the best model!")
#best_model = tuner.get_best_models()[0]
#best_model.evaluate(test_img, test_lbl)

## ----- Tuner Best Model Description -----

#best_model.summary()


