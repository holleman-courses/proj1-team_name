# train.py

import tensorflow as tf
from tensorflow.keras import layers, models
import os

def build_model():
    # A simple CNN for binary classification (dog vs. other)
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),  # Adjust to image size
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary output: 1 for dog, 0 for other
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Use image_dataset_from_directory to load the training data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'data/train',
        image_size=(64, 64),
        batch_size=2,  # Since dataset is very small
        label_mode='binary'
    )
    
    model = build_model()
    model.summary()
    
    # Train
    history = model.fit(train_ds, epochs=5)
    
    # Save the trained model
    model.save('trained_model.h5')
