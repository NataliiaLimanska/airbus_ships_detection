import os
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf


# Function to decode RLE-encoded masks
def rle_decode(encoded_pixels, shape):
    """
        Decode a Run-Length Encoding (RLE) representation of a binary mask.

        Parameters:
        - encoded_pixels (str): RLE-encoded string representing the mask.
        - shape (tuple): Shape of the target mask in the format (height, width).

        Returns:
        - mask (numpy.ndarray): Decoded binary mask with specified shape,
          if rle is NaN, return all zeros
        """
    if pd.isna(encoded_pixels):
        return np.zeros(shape, dtype=np.uint8).T
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    encoded_pixels = list(map(int, str(encoded_pixels).split()))
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

    # Ensure runs have an even length
    if len(runs) % 2 != 0:
        runs = runs[:-1]

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


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
    return float(
        (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))


# Path to the directory with test images
TEST_PATH = "..........................."

# Path to directory with file containing RLE
CSV_PATH = "............................."

# Path to save file with predictions
OUTPUT_PATH = "..........................."

# Load the CSV file containing image IDs and RLE-encoded masks
test_labels = pd.read_csv(CSV_PATH)

# Load the trained model with custom metric
model = load_model("/content/drive/MyDrive/ship_detection_unet_model_7.keras",
                   custom_objects={'dice_coefficient': dice_coefficient})

# Initialize lists to store predictions and Dice coefficients
predictions = {'ImageId': [], 'EncodedPixels': [], 'DiceCoefficient': []}

# Iterate through entries in the CSV file with RLE
for index, row in test_labels.iterrows():
    image_id = row['ImageId']

    # Check if the image file exists in the testing folder
    image_path = os.path.join(TEST_PATH, image_id)

    if os.path.exists(image_path):
        # Read and preprocess the test image
        test_image = load_img(image_path, target_size=(256, 256))
        test_image = img_to_array(test_image) / 255.0  # Normalize pixel values
        test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

        # Generate predictions for the test image
        predictions_mask = model.predict(test_image)

        # Convert predictions to binary mask using a threshold
        binary_predictions = (predictions_mask > 0.5).astype(np.uint8)

        # Convert binary predictions to TensorFlow tensor
        binary_predictions = tf.convert_to_tensor(binary_predictions, dtype=tf.float32)

        # Squeeze the extra dimension
        binary_predictions_squeezed = tf.squeeze(binary_predictions)

        # Encode the binary mask to RLE format
        rle_mask = rle_encode(binary_predictions_squeezed.numpy())

        # Decode ground truth mask and resize predictions
        ground_truth_mask = tf.cast(rle_decode(row['EncodedPixels'], shape=(256, 256)), dtype=tf.float32)
        binary_predictions_resized = tf.image.resize(binary_predictions, (256, 256))

        # Calculate Dice coefficient
        dice_coef = dice_coefficient(ground_truth_mask, binary_predictions_resized)

        # Add predictions and Dice coefficient to the lists
        predictions['ImageId'].append(image_id)
        predictions['EncodedPixels'].append(rle_mask)
        predictions['DiceCoefficient'].append(float(dice_coef))

# Create a DataFrame from the predictions
predictions_df = pd.DataFrame(predictions)

# Save the predictions to a CSV file
predictions_df.to_csv(OUTPUT_PATH, index=False)

# Calculate the mean Dice coefficient for all predictions
mean_dice_coefficient = np.mean(predictions_df['DiceCoefficient'])

print(f"Predictions saved to {OUTPUT_PATH}")

# Print out the mean_dice_coefficient
print(f"Mean Dice Coefficient for all test images: {mean_dice_coefficient}")


