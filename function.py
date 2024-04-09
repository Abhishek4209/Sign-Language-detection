# Import necessary libraries
import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize Mediapipe drawing utilities and Hands model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define a function to perform hand detection using the Mediapipe model
def mediapipe_detection(image, model):
    # Convert the image from BGR to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Make the image unwriteable to ensure its integrity during processing
    image.flags.writeable = False
    # Process the image to make predictions
    results = model.process(image)
    # Make the image writeable again
    image.flags.writeable = True
    # Convert the image back to BGR color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Define a function to draw styled landmarks on an image
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks and connections using Mediapipe drawing functions
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

# Define a function to extract keypoints from the results
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert hand landmarks to a flattened array of (x, y, z) coordinates
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21 * 3)
            return np.concatenate([rh])

# Define the path for exporting data (numpy arrays)
DATA_PATH = os.path.join('MP_Data')

# Define an array of actions (A, B, C)
actions = np.array(['A', 'B', 'C'])

# Define the number of sequences and sequence length
no_sequences = 30
sequence_length = 30
