import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# -------------------------------
# 1. Set Parameters and Directories
# -------------------------------
IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 32
EPOCHS = 100
DATA_DIR = "TrainingData/Dog_vs_Not-Dog"  # dataset should be structured in subdirectories per class
TEST_DIR = "TrainingData"

# -------------------------------
# 2. Data Loading & Preprocessing
# -------------------------------
# Create ImageDataGenerators for training (with augmentation) and validation/testing (only rescaling)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,  # use 20% for validation
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches using train_datagen (grayscale mode)
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


# -------------------------------
# 3. Build the CNN Model
# -------------------------------
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # First Convolutional Block
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# Determine input shape (height, width, channels)
input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)  # 1 channel for grayscale
num_classes = len(train_generator.class_indices)

model = build_cnn_model(input_shape, num_classes)
model.summary()

# -------------------------------
# 4. Compile the Model
# -------------------------------
# Create a learning rate scheduler using cosine decay
steps_per_epoch = train_generator.samples // BATCH_SIZE
lr_schedule = optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-2,
    decay_steps=steps_per_epoch * EPOCHS
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 5. Train the Model
# -------------------------------
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# -------------------------------
# 6. Evaluate the Model
# -------------------------------
# Optionally, if you have a separate test set, you can evaluate:
#test_generator = test_datagen.flow_from_directory(
#    TEST_DIR,
#    target_size=(IMG_HEIGHT, IMG_WIDTH),
#    color_mode='grayscale',
#    batch_size=BATCH_SIZE,
#    class_mode='categorical'
#)
#test_loss, test_acc = model.evaluate(test_generator)
#print("Test accuracy:", test_acc)

# -------------------------------
# 7. Convert the Model to TensorFlow Lite
# -------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optionally, you can enable optimizations to reduce model size (important for microcontrollers)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save model
model.save("saved_model6.h5")

# Save the TFLite model to a file
with open('model.tflite6', 'wb') as f:
    f.write(tflite_model)

print("Model has been converted to TFLite and saved as model.tflite6")


# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.epoch, history.history['accuracy'], label='Train Accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.epoch, history.history['loss'], label='Train Loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
