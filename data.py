from function import *
from time import sleep

# Loop through each action and sequence and create directories to store data
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Initialize Mediapipe Hands model for hand tracking
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Loop through each action
    for action in actions:
        
        # Loop through each sequence (video)
        for sequence in range(no_sequences):
            
            # Loop through each frame in the sequence
            for frame_num in range(sequence_length):

                # Read the frame from the stored images
                frame = cv2.imread('Image/{}/{}.png'.format(action, sequence))
                
                # Make detections using the Mediapipe Hands model
                image, results = mediapipe_detection(frame, hands)
                
                # Draw landmarks on the image
                
                # 2. Draw landmarks on the image using the function `draw_styled_landmarks`
                draw_styled_landmarks(image, results)
                
                                # Display a message indicating the start of frame collection
                if frame_num == 0:
                    # Set the starting collection message at position (120, 200) on the image
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    # Display the message with action and sequence information at position (15, 12)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show the image on screen
                    cv2.imshow('OpenCV Feed', image)
                    # Wait for 200 milliseconds to allow the message to be displayed
                    cv2.waitKey(200)
                else:
                    # Display the message with action and sequence information at position (15, 12)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show the image on screen
                    cv2.imshow('OpenCV Feed', image)


                # Extract keypoints from the detection results
                
                # 3. Extract keypoints from the detection results using the function `extract_keypoints`
                keypoints = extract_keypoints(results)
                
                # Define the path to save keypoints in a .npy file
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                
                # Save the keypoints as a .npy file
                
                # 4. Save the keypoints as a .npy file using the `np.save` function
                np.save(npy_path, keypoints)

                # Break the loop gracefully if the 'q' key is pressed
                
                # 5. Check if the 'q' key is pressed to break the loop
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

# Close all OpenCV windows
cv2.destroyAllWindows()
