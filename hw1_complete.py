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


# Build and Compile
  model_best = build_model50k()
  model_best.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
      
  # Check the param count - MUST BE UNDER 50,000
  model_best.summary() 

  # Train (Maybe use 50 epochs since small models learn slower)
  model_best.fit(train_images, train_labels, 
                  epochs=50, 
                  validation_data=(val_images, val_labels))

  # SAVE THE MODEL AS REQUESTED
  model_best.save("best_model.h5")
  print("Success! best_model.h5 has been saved.")

    ### Repeat for model 3 and your best sub-50k params model

  
  
