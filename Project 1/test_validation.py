import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
MODEL_PATH = "saved_model.h5"  # Update if necessary
DATA_DIR = "TrainingData/Dog_vs_Not-Dog"
IMG_HEIGHT = 96
IMG_WIDTH = 96

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Set up the ImageDataGenerator for the validation set
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=1,  # one image at a time
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get the class labels (make sure these match the training order)
CLASS_NAMES = list(validation_generator.class_indices.keys())

# Iterate over validation images and display predictions
num_images = len(validation_generator)
for i in range(num_images):
    img, true_label = validation_generator[i]  # load one image and its label
    # Get model prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    # Display the image along with its predicted label and confidence
    # img is already normalized; convert back to original scale for display if desired
    img_disp = img[0]  # remove batch dimension
    # If you want a cleaner image, multiply by 255 and cast to int (optional)
    img_disp = (img_disp * 255).astype("uint8").squeeze()

    plt.imshow(img_disp, cmap="gray")
    plt.title(f"Prediction: {CLASS_NAMES[predicted_class]} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()
