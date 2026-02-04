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


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


def build_model1():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32,32,3)),
        tf.keras.layers.Dense(128), # Dense Layer 1
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(128),  # Dense Layer 2
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(128),  # Dense Layer 3
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(10)
    ])
    return model

def build_model2():

  model = tf.keras.Sequential([
        # Layer 1: 32 filters, stride 2
        tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),

        # Layer 2: 64 filters, stride 2
        tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),

        # Four more pairs of 128-filter layers (Stride defaults to 1)
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),

        # Output Section
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
  
  return model

def build_model3():
  model = None # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = None # Add code to define model 1.
  return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':
  ########################################
  ## Add code here to Load the CIFAR10 data set
  ########################################
  (train_images, train_labels), (test_images, test_labels)= tf.keras.datasets.cifar10.load_data()
  ## Build and train model 1
  train_images, val_images, train_labels, val_labels = train_test_split(
    train_images,
    train_labels,
    test_size=0.2,      # 20% for validation
    random_state=42
)
  train_images = train_images / 255.0
  val_images   = val_images / 255.0
  test_images  = test_images / 255.0

# ----------------------------MODEL 1------------------------------------------------------------
  """model1 = build_model1()
  # compile and train model 1
  model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  model1.summary()

  history = model1.fit(
              train_images,
              train_labels,
              epochs=30,
              validation_data=(val_images, val_labels)
            )
train_acc = history.history['accuracy'][-1]       # last epoch training accuracy
val_acc   = history.history['val_accuracy'][-1]   # last epoch validation accuracy

test_loss, test_acc = model1.evaluate(test_images, test_labels)

test_loss, test_accuracy = model1.evaluate(
    test_images,
    test_labels
)

print("Test accuracy:", test_accuracy)

#------------------------------------------MODEL 1 END--------------------------------------------
"""
#-----------------------------------------MODEL 2 ----------------------------------------------
# Build, compile, and train model 2 (DS Convolutions)
model2 = build_model2()
# Compiling the model
model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']) 
model2.summary()

history = model2.fit(
              train_images,
              train_labels,
              epochs=30,
              validation_data=(val_images, val_labels)
            )
train_acc = history.history['accuracy'][-1]       # last epoch training accuracy
val_acc   = history.history['val_accuracy'][-1]   # last epoch validation accuracy

test_loss, test_acc = model2.evaluate(test_images, test_labels)

test_loss, test_accuracy = model2.evaluate(
    test_images,
    test_labels
)

print("Test accuracy:", test_accuracy)



  
  ### Repeat for model 3 and your best sub-50k params model
  
  
