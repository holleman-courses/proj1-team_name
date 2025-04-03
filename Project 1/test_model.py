import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define paths
MODEL_PATH = "saved_model3.h5"  # Update this if your model is saved with a different name
TEST_DIR = "TrainingData/TestImages/"  # Folder with test images

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Define image properties (same as used in training)
IMG_HEIGHT = 96
IMG_WIDTH = 96

# Get class labels (same order as training)
CLASS_NAMES = ["Dog", "Not-Dog"]  # Adjust if needed

# Get list of test images
test_images = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# Process and predict each image
for img_name in test_images:
    img_path = os.path.join(TEST_DIR, img_name)

    # Load and preprocess image
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get model prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get index of highest confidence score
    confidence = np.max(predictions) * 100  # Get confidence percentage

    # Display results
    plt.imshow(img, cmap="gray")
    plt.title(f"Prediction: {CLASS_NAMES[predicted_class]} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()
