# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

An autoencoder is a neural network trained to reconstruct its input. It encodes the input into a lower-dimensional representation and then decodes it back to the original form. Using MaxPooling, convolutional, and upsampling layers, autoencoders denoise images. In this experiment with the MNIST dataset of handwritten digits, we're building a convolutional neural network to classify each image into its numerical value from 0 to 9.

## Convolution Autoencoder Network Model

![image](https://github.com/Pavithraramasaamy/convolutional-denoising-autoencoder/assets/118596964/21c1b6a1-685f-4e37-ba33-df36f3849322)

## DESIGN STEPS

### STEP 1:
 Import the necessary libraries and dataset.

### STEP 2:
Load the dataset and scale the values for easier computation.

### STEP 3:
Add noise to the images randomly for both the train and test sets.

### STEP 4:
Build the Neural Model using Convolutional Layer Pooling Layer Up Sampling Layer. Make sure the input shape and output shape of the model are identical. 

### STEP 5:
Pass test data for validating manually. Step 6: Plot the predictions for visualization.

## PROGRAM
```
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
x_train_scaled = x_train.astype('float32') / 255.
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape)
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

print("Name: PAVITHRA R")
print("Register number: 212222230106")
input_img = keras.Input(shape=(28, 28, 1))

conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
maxpool1 = layers.MaxPooling2D((2, 2), padding='same')(conv1)

conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(maxpool1)
encoded=  layers.MaxPooling2D((2, 2), padding='same')(conv2)

# Encoder output dimension is ## Mention the dimention ##
print("Shape of the encoder output:", encoded.shape)

# Write your decoder here
conv3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
upsample1 = layers.UpSampling2D((2, 2))(conv3)
conv4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(upsample1)
upsample2 = layers.UpSampling2D((2, 2))(conv4)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upsample2)

autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))

metrics = pd.DataFrame(autoencoder.history.history)
metrics[['loss','val_loss']].plot()
print("Name: PAVITHRA R")
print("Register number: 212222230106")
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
print("Name: PAVITHRA R")
print("Register number: 212222230106")
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![Screenshot 2024-05-06 090618](https://github.com/Pavithraramasaamy/convolutional-denoising-autoencoder/assets/118596964/166335c9-dc15-40c1-b7e3-c37e0b8ad723)


### Original vs Noisy Vs Reconstructed Image

![Screenshot 2024-05-06 090623](https://github.com/Pavithraramasaamy/convolutional-denoising-autoencoder/assets/118596964/12696cf6-fe91-40f2-ba17-3191e889d98f)



## RESULT
Thus we have successfully developed a convolutional autoencoder for image denoising application.
