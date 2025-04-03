import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
MODEL_PATH = "saved_model6.h5"  # Ensure your model is saved with this name
DATA_DIR = "TrainingData/Dog_vs_Not-Dog"
IMG_HEIGHT = 96
IMG_WIDTH = 96

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Set up the ImageDataGenerator for the validation set
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

# Using batch_size=1 to process individual images
validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=1,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get the class labels in the same order as during training
CLASS_NAMES = list(validation_generator.class_indices.keys())

# -------------------------------
# 1. Compute the Confusion Matrix
# -------------------------------
all_true_labels = []
all_pred_labels = []

num_val_images = len(validation_generator)
for i in range(num_val_images):
    img, true_label = validation_generator[i]
    true_index = np.argmax(true_label[0])
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions[0])

    all_true_labels.append(true_index)
    all_pred_labels.append(predicted_index)

# Compute confusion matrix using scikit-learn
cm = confusion_matrix(all_true_labels, all_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# 2. Collect Misclassified Images
# -------------------------------
misclassified = []

for i in range(num_val_images):
    img, true_label = validation_generator[i]
    true_index = np.argmax(true_label[0])
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100  # Confidence percentage

    if predicted_index != true_index:
        misclassified.append({
            'image': img[0],
            'true_label': CLASS_NAMES[true_index],
            'predicted_label': CLASS_NAMES[predicted_index],
            'confidence': confidence
        })

# -------------------------------
# 3. Display Misclassified Images in Batches of 4
# -------------------------------
batch_size = 4
total = len(misclassified)

if total == 0:
    print("No misclassifications found!")
else:
    print(f"Total misclassifications: {total}")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        current_batch = misclassified[start:end]
        cols = len(current_batch)  # number of images in this batch
        plt.figure(figsize=(cols * 3, 3))

        for idx, data in enumerate(current_batch):
            plt.subplot(1, cols, idx + 1)
            # Convert normalized grayscale image to displayable format (0-255)
            img_disp = (data['image'] * 255).astype("uint8").squeeze()
            plt.imshow(img_disp, cmap="gray")
            plt.title(f"T: {data['true_label']}\nP: {data['predicted_label']}\n{data['confidence']:.1f}%")
            plt.axis("off")

        plt.tight_layout()
        plt.show()  # Blocks until the figure window is closed
        input("Press Enter to view the next batch...")
        plt.close("all")
