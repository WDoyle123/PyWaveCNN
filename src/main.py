import os 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as img

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from plotter import get_class_samples

def main():
 
    # Visualise the types of data in the training data
    get_class_samples()
    
    # Get directories for image generator
    train_dir = '../data/train/'
    valid_dir = '../data/validation/'
    test_dir = '../data/test/'
    class_names = os.listdir(train_dir)

    img_size = (400, 400)  # This should be a tuple
    batch_size = 32

    # Create the datasets for each directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=220301,
        image_size=img_size,
        shuffle=True,
        batch_size=batch_size)

    valid_ds = tf.keras.utils.image_dataset_from_directory(
            valid_dir,
            seed=220301,
            image_size=img_size,
            batch_size=batch_size)

    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(img_size[0], img_size[1]),
        layers.Rescaling(1./255)
        ])

    model = tf.keras.models.Sequential([
        resize_and_rescale,

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_size + (3,))),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(22, activation='softmax')
        ])

    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', 
                                   restore_best_weights=True, verbose=1, 
                                   baseline=0.95)

    model.build((None, 400, 400, 3))

    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='rmsprop', metrics=['accuracy'])

    history = model.fit(train_ds,
                        epochs=100,
                        validation_data=valid_ds,
                        callbacks=[early_stopping])




if __name__ == '__main__':
    main()
