# Let's import the libraries required for the image classification
import os
import glob
RAND_SEED = 12345
import numpy as np
np.random.seed(RAND_SEED)
import tensorflow as tf
tf.random.set_seed(RAND_SEED)
import random
random.seed(RAND_SEED)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load data from files
IMG_DIM = 96
IMG_PER_CLASS = 500
TEST_SPLIT_FRAC = 0.1
TRAIN_SPLIT_FRAC = 0.9
BATCH_SIZE = 32
NUM_CLASSES = 3

data_dir = './data'
class_names = ['fault', 'salt', 'other']
file_list = []

for class_name in class_names:
    class_files = glob.glob(data_dir + os.path.sep + \
                            class_name + os.path.sep + '*.png')
    for f in class_files:
        file_list.append(f)

def get_class_label(file_path):
    # Split file path to get directory and file names.
    path_split = file_path.split(os.path.sep)
    # The second-last string contains name of the class directory;
    # Use it to create class label.
    return class_names.index(path_split[-2].strip())

def decode_image(img):
    # Convert the passed image to a uint8 tensor;
    # Using 1 channel, as the images are grayscale.
    img = tf.image.decode_png(img, channels=1)
    # Convert image to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize the image to the predefined image size.
    return tf.image.resize(img, [IMG_DIM, IMG_DIM]).numpy()

def process_file_path(file_path):
    label = get_class_label(file_path)
    # Load data from the image file.
    img = tf.io.read_file(file_path)
    # Decode and convert image data to float32.
    img = decode_image(img)
    return img, label

images = []
labels = []
for f in file_list:
    img, label = process_file_path(f)
    images.append(img)
    labels.append(label)
    
images = np.array(images)
labels = np.array(labels)

# Create training, validation, and test data sets
def split_datasets(images, labels):
    # Split data into train, validation, and test data sets.
    X_rest, X_test, y_rest, y_test = \
    train_test_split(images, labels, test_size=TEST_SPLIT_FRAC, 
                     random_state=RAND_SEED, stratify=labels)
    X_train, X_valid, y_train, y_valid = \
    train_test_split(X_rest, y_rest, train_size=TRAIN_SPLIT_FRAC, 
                     random_state=RAND_SEED, stratify=y_rest)
    return X_train, X_valid, X_test, y_train, y_valid, y_test
    
X_train, X_valid, X_test, y_train, y_valid, y_test = \
split_datasets(images, labels)

# Plot class population distribution in the training data
# Calculate population of three classes
population = [len(np.where(y_train == 0)[0]), 
              len(np.where(y_train == 1)[0]),
              len(np.where(y_train == 2)[0])]
# Plot
plt.bar(class_names, population)
plt.title('Population distribution: Training Data')

# determine the number of input features
n_train = population[0] + population[1] + population[2]
n_test = int(NUM_CLASSES * IMG_PER_CLASS * TEST_SPLIT_FRAC)
n_valid = NUM_CLASSES * IMG_PER_CLASS - n_train - n_test
X_train = np.reshape(X_train, (n_train, IMG_DIM, IMG_DIM, 1))
X_valid = np.reshape(X_valid, (n_valid, IMG_DIM, IMG_DIM, 1))
X_test = np.reshape(X_test, (n_test, IMG_DIM, IMG_DIM, 1))

# Build and train a Deep Neural Network model
# Define a Deep Neural Network model
model = Sequential()
model.add(Conv2D(96, (5, 5), activation='relu', 
                 input_shape=(IMG_DIM, IMG_DIM, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

# compile the model
model.compile(optimizer='Adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Early-stopping callback using validation
earlystop_callback = EarlyStopping(
    monitor='val_accuracy', min_delta=0.0001,
    patience=50)
# Save the best model
ckpt_path = './models/cnn.h5'
ckpt_callback = ModelCheckpoint(filepath=ckpt_path, mode='max', 
                                monitor='val_accuracy', verbose=1, 
                                save_best_only=True)
# Train the model
history = model.fit(X_train, y_train, epochs=500, batch_size=BATCH_SIZE, 
                    validation_data=(X_valid, y_valid), 
                    callbacks=[earlystop_callback, ckpt_callback], 
                    verbose=1)

# Use saved model to predict test data
# Evaluate the best model using test data
model = load_model(ckpt_path)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
