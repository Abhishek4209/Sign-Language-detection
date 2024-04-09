# Import necessary functions and libraries from external file 'function.py'
from function import *
# Import required functions and classes from libraries
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Create a mapping of action labels to numerical values
label_map = {label: num for num, label in enumerate(actions)}

# Initialize empty lists to store sequences and labels
sequences, labels = [], []

# Loop through each action label (A, B, C)
for action in actions:
    # Loop through a fixed number of sequences for each action
    for sequence in range(no_sequences):
        window = []
        # Loop through a fixed sequence length
        for frame_num in range(sequence_length):
            # Load the stored keypoints for each frame in the sequence
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        # Append the sequence of keypoints to the sequences list
        sequences.append(window)
        # Append the numerical label for the action to the labels list
        labels.append(label_map[action])

# Convert sequences and labels to NumPy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Specify the directory for TensorBoard logs
log_dir = os.path.join('Logs')
# Create a TensorBoard callback for visualizing training progress
tb_callback = TensorBoard(log_dir=log_dir)

# Initialize a Sequential model
model = Sequential()
# Add LSTM layers with specified configurations
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
# Add Dense layers with specified activation functions
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# Add the final output layer with softmax activation for multiclass classification
model.add(Dense(actions.shape[0], activation='softmax'))

# Specify an example result array (not used)
# This array is provided as an example and is not used in the code. It represents
# a hypothetical set of prediction results for a given sequence, with three values.
res = [.7, 0.2, 0.1]

# Compile the model with specified optimizer, loss function, and evaluation metric
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model on the training data for a specified number of epochs, using the TensorBoard callback
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Display a summary of the model architecture
model.summary()

# Convert the model to JSON format and save it to a file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights to a separate file
model.save('model.h5')
