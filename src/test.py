import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set paths and parameters
model_path = 'deepfake_recognition_model.h5'  # Path to your saved model
test_dir = 'Dataset/Test'  # Path to your test dataset
img_height = 128  # Image height
img_width = 128  # Image width
batch_size = 32  # Batch size

# Load the saved model
print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully.")

# Set up the test data generator
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Create the test data generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # For binary classification (Fake or Real)
    shuffle=False  # Do not shuffle so predictions match the file order
)

# Evaluate the model
print("Evaluating the model on the test set...")
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions on the test data
print("Making predictions...")
predictions = model.predict(test_generator)

# Map predictions to class labels
predicted_classes = (predictions > 0.5).astype("int").flatten()

# Get filenames of the test images
filenames = test_generator.filenames
labels = test_generator.class_indices

# Display results for each file
for i, filename in enumerate(filenames):
    predicted_label = "Fake" if predicted_classes[i] == 1 else "Real"
    print(f"{filename}: Predicted - {predicted_label}")
