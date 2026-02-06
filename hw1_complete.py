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
    
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # LAYER 1: Standard Conv2D 
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    # LAYER 2: SeparableConv2D 
    x = tf.keras.layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    shortcut = x 

    # Two Separable Layers (Stride 1)
    x = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SeparableConv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, shortcut]) 
    x = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SeparableConv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="model3_residual")
    return model

def build_model50k():
    model = tf.keras.Sequential([
        # LAYER 1: Standard Conv
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        
        # LAYER 2: Separable Downsample (Stride 2)
        tf.keras.layers.SeparableConv2D(32, (3, 3), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2), # Early dropout to handle noise

        # LAYER 3: Deepening the features
        tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        
        # LAYER 4: Final Downsample (Stride 2)
        tf.keras.layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3), # Stronger dropout as features get abstract

        # LAYER 5: Final Feature Processing
        tf.keras.layers.SeparableConv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        # Global Average Pooling replaces Flatten to keep params low
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Final Classification
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10) # No activation here because from_logits=True
    ])
    return model

# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import