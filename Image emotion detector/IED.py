# pylance: reportMissingImports=false

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 7

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

# =========================
# DATA GENERATORS
# =========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# =========================
# TRANSFER LEARNING MODEL
# =========================
base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE MODEL
# =========================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN MODEL
# =========================
model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# =========================
# SAVE MODEL
# =========================
model.save("emotion_detector_model.h5")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_emotion(image_path):
    emotions = list(train_data.class_indices.keys())

    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    print("Predicted Emotion:", emotions[np.argmax(prediction)])

# Example:
# predict_emotion("test.jpg")
