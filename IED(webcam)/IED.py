import cv2                           # to access the webcam ,draw text on video frames and show live video window.
import numpy as np                   # To handle the data in the form of arrays and numbers
import tensorflow as tf              # Used to build, train, and fine-tune the CNN model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator #Loading images from folders and applying data augmentation (rotation, zoom, flip) 
import matplotlib.pyplot as plt      # Used to plot accuracy & loss graphs and Help analyze model performance visually.


#Configuration
# This resizes all the images to 160 x 160
IMG_SIZE = 160          # smaller = faster + less heat 
# Number of images processed together in one step
BATCH_SIZE = 32
# Train classifier head
INITIAL_EPOCHS = 5
# Fine - tune base model
FINE_TUNE_EPOCHS = 5
# Number of classes for emotions 
NUM_CLASSES = 7

# To train and test
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# DATA GENERATORS

# It converts converts pixel values from 0–255 → 0–1
# It creates augmented images to avoid overfitting

train_gen = ImageDataGenerator(
    rescale=1./255,
    # rotation handles head tilt
    rotation_range=20,
    # handles distance variation
    zoom_range=0.2,
    # handles left/right face orientation
    horizontal_flip=True
)

#  Test generator
test_gen = ImageDataGenerator(rescale=1./255)

# this is to load the images 
train_data = train_gen.flow_from_directory(
   # reads folder names as labels
    TRAIN_DIR,
    # Resizes images
    target_size=(IMG_SIZE, IMG_SIZE),
    # Converts labels to one hot vectors
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Same logic for testing the data
test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)
# Stores label order and are needed later for webcam prediction mapping
emotion_labels = list(train_data.class_indices.keys())

# MODEL (TRANSFER LEARNING)
# we are using MobileNetV2 as it is lightweight ,very fast , has high accuracy and is ideal for real time web applications
base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
# This is to freeze pretrained layers to prevent destroying the learned features and trains only new classification layers  
base_model.trainable = False  # freeze initially

# output feature map from CNN
x = base_model.output

# Converts feature maps → single vector and reduces overfitting
x = layers.GlobalAveragePooling2D()(x)

# To stabilize learning and for faster convergence
x = layers.BatchNormalization()(x)

# this learns emotion specific features using Relu
# we use relu to introduce non-linearity, enabling networks to learn complex patterns (beyond simple linear combinations), solve complex problems like image recognition, 
# and train faster by mitigating the vanishing gradient problem (gradients don't shrink as much) and computationally efficiently (simple max(0, x) operation).
x = layers.Dense(256, activation="relu")(x)

# this randonly drops neurons to avoid overfitting
x = layers.Dropout(0.5)(x)
# "softmax" provides the probability of each emotion
output = layers.Dense(NUM_CLASSES, activation="softmax")(x)

# This is to call the final complete model
model = keras.Model(inputs=base_model.input, outputs=output)

# COMPILE & TRAIN (STAGE 1)


model.compile(
    # we use adam optimizer for adaptive learning
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    # "Categorial crossentropy" is a multi class classification
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🔹 Training classifier...")

# This trains only top layers and learns to classify emotions
history1 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=INITIAL_EPOCHS
)

# FINE-TUNING (STAGE 2)

print("\n🔹 Fine-tuning last layers...")
# this allows cnn to learn task specific features
base_model.trainable = True

# Fine tunes only last 30 layers to prevent overfitting
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Lower learning rate as fine tuning must be slow and carefully done
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# This improves accuracy
history2 = model.fit(
    train_data,
    validation_data=test_data,
    epochs=FINE_TUNE_EPOCHS
)

# SAVE MODEL
model.save("emotion_detector_model.keras")
print("\n Model saved!")

# PLOT ACCURACY & LOSS
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.title("Loss")

plt.show()

# WEBCAM EMOTION DETECTION
print("\n Starting webcam... Press Q to quit")

# Opens default webcam
# 0 --> for built in camera
cap = cv2.VideoCapture(0)

# this is for infinite loop that is for cont video frames
while True:
    # Reads a single frame and frame = image matrix and ret = success flag
    ret, frame = cap.read()
    if not ret:
        break

# Resizes frame to match model input
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    # this normalizes pixels which is same as training
    img = img / 255.0
    # adds batch dimension
    img = np.expand_dims(img, axis=0)

   # This predicts emotion probability
    preds = model.predict(img, verbose=0)
    # Finds highest prob emotion and it converts index to label name
    emotion = emotion_labels[np.argmax(preds)]

   # This displays predicted emotion on screen
    cv2.putText(
        frame,
        emotion,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # This shows live webcam window
    cv2.imshow("Emotion Detector", frame)
   
    # to exit we press Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releases webcam and Closes OpenCV windows and prevents camera locks issues
cap.release()
cv2.destroyAllWindows()
