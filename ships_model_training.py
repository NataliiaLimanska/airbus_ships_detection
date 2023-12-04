import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from keras import backend as K


def rle_decode(encoded_pixels, shape):
    """
        Decode a Run-Length Encoding (RLE) representation of a binary mask.

        Parameters:
        - encoded_pixels (str): RLE-encoded string representing the mask.
        - shape (tuple): Shape of the target mask in the format (height, width).

        Returns:
        - mask (numpy.ndarray): Decoded binary mask with specified shape.
        """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    encoded_pixels = list(map(int, encoded_pixels.split()))
    for i in range(0, len(encoded_pixels), 2):
        start = encoded_pixels[i] - 1
        length = encoded_pixels[i + 1]
        mask[start:start + length] = 1
    return mask.reshape((shape[1], shape[0])).T


def rle_encode(mask):
    """
        Encode a binary mask using Run-Length Encoding (RLE).

        Parameters:
        - mask (numpy.ndarray): Binary mask to be encoded.

        Returns:
        - rle_string (str): RLE-encoded string representation of the mask.
        """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) % 2 != 0:
        runs = np.append(runs, len(pixels))
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# Function to calculate Dice coefficient
def dice_coefficient(y_true, y_pred):
    """
        Calculate the Dice coefficient, a metric for image segmentation accuracy.

        Parameters:
        - y_true (tf.Tensor): Ground truth binary mask.
        - y_pred (tf.Tensor): Predicted binary mask.

        Returns:
        - dice_coeff (tf.Tensor): Computed Dice coefficient value.
        """
    smooth = 1e-10
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def load_and_preprocess_image(image_path, target_shape=(256, 256)):
    """
        Load and preprocess an image from the specified path.

        Parameters:
        - image_path (str): Path to the image file.
        - target_shape (tuple): Target shape for resizing the image.

        Returns:
        - image (numpy.ndarray): Preprocessed image in RGB format.
        """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_shape)
    return image


# Path to directory with images
DATA_PATH = "..............."

# Path to directory with .csv file with rle of the ships
DESCRIPTIVE_PATH = "................."

# Read the CSV file to get the mapping between image IDs and RLE-encoded masks
description_file = pd.read_csv(DESCRIPTIVE_PATH)
image_id_to_rle = dict(zip(description_file['ImageId'], description_file['EncodedPixels']))

# List all files in the directory with images
image_files = [file for file in os.listdir(DATA_PATH) if file.endswith(".jpg")]

# Initialize lists to store images, masks, and ship counts
images = []
masks = []

# Iterate through image files
for image_file in image_files:
    image_path = os.path.join(DATA_PATH, image_file)

    # Read and preprocess the image
    original_image = load_and_preprocess_image(image_path)
    if original_image is None:
        continue

    # Get all rows from the description file that correspond to the current image ID
    image_rows = description_file[description_file['ImageId'] == image_file]

    # Initialize a mask for the current image
    decoded_mask = np.zeros((256, 256), dtype=np.uint8)

    # Iterate through the rows to get RLE masks
    for _, row in image_rows.iterrows():
        rle_mask = row['EncodedPixels']

        # Check if EncodedPixels is not NaN
        if not pd.isna(rle_mask):
            shape = original_image.shape[:2]
            mask = rle_decode(rle_mask, shape)
            # Apply the mask to the decoded mask
            decoded_mask += mask
        else:
            # If EncodedPixels is NaN, set to zero (as a background)
            decoded_mask = np.zeros((256, 256), dtype=np.uint8)

    # Append the image to the list
    images.append(original_image)

    # Transpose the mask to match the required shape
    decoded_mask = decoded_mask.T

    # Append decoded masks to the list
    masks.append(decoded_mask)


# Convert lists to NumPy arrays
images = np.array(images).astype(np.float32)
masks = np.array(masks).astype(np.float32)

# Add a channel dimension to ground truth masks
masks = np.expand_dims(masks, axis=-1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Convert NumPy arrays to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)


# The U-Net model
def unet_model(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Middle
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Create the model
model = unet_model()

# Compile the model with Dice coefficient as the evaluation metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])

# Train the model and store the history
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=8)

# Save the trained model in certain directory
model.save("..............")
print("Trained model saved successfully!")

# Generate predictions for the validation set
predictions = model.predict(X_val)

# Convert predictions to binary masks
binary_predictions = (predictions > 0.5).astype(np.float32)

# Convert binary predictions to RLE format
predicted_rles = [rle_encode(binary_prediction) for binary_prediction in binary_predictions]

# Assuming you have the ground truth RLEs for the validation set
ground_truth_rles = [rle_encode(gt_mask) for gt_mask in y_val.numpy()]

# Evaluate the model on the validation set using Dice coefficient
dice_scores = [dice_coefficient(y_true, y_pred) for y_true, y_pred in zip(y_val, binary_predictions)]
mean_dice = np.mean(dice_scores)

print(f"Mean Dice Coefficient on Validation Set: {mean_dice}")
