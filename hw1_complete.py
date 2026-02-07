#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from sklearn.model_selection import train_test_split
import os


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


def build_model1():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32,32,3)),
        tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_model2():
    model = keras.Sequential([
        layers.Conv2D(32, 3, strides=2, padding='same', activation='relu', input_shape=(32,32,3)),
        layers.BatchNormalization(),

        layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_model3():
  model = tf.keras.Sequential([
    tf.keras.layers.SeparableConv2D(32, 3, strides=2, padding='same', activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.SeparableConv2D(64, 3, strides=2, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
  ])
  model.compile(
      optimizer='adam',
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
    )
  return model

def build_model50k():
  model = keras.Sequential([

    # 32x32x3 → 16x16x32
    layers.SeparableConv2D(32, 3, padding='same', activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # 16x16x32 → 8x8x64
    layers.SeparableConv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # 8x8x64 → 8x8x128
    layers.SeparableConv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),

    # Reduce spatial size cheaply
    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(10)
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=0.001),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy']
    )
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import


if __name__ == '__main__':

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    val_images = train_images[-5000:]
    val_labels = train_labels[-5000:]
    train_images = train_images[:-5000]
    train_labels = train_labels[:-5000]

    model1 = build_model1()
    model1.fit(train_images, train_labels, epochs=30,
               validation_data=(val_images, val_labels))

    model2 = build_model2()
    model2.fit(train_images, train_labels, epochs=30,
               validation_data=(val_images, val_labels))

    model3 = build_model3()
    model3.fit(train_images, train_labels, epochs=30,
               validation_data=(val_images, val_labels))

    best_model = build_model50k()
    best_model.fit(train_images, train_labels, epochs=30,
                   validation_data=(val_images, val_labels))

    best_model.save("best_model.h5")



  
  
