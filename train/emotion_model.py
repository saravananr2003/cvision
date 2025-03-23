import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load dataset
def load_data(data_dir):
    images = []
    labels = []
    for label in ['happy', 'sad']:
        path = os.path.join(data_dir, label)
        print (path)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = tf.keras.preprocessing.image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
            img_array = tf.keras.preprocessing.image.img_to_array(img_array) / 255.0
            images.append(img_array)
            labels.append(0 if label == 'sad' else 1)
    return np.array(images), np.array(labels)

# Prepare data
data_dir = 'path_to_dataset'
images, labels = load_data(data_dir)
images = np.expand_dims(images, axis=-1)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(x_train)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_test, y_test), epochs=20)

# Save model
model.save('emotion_model.h5')