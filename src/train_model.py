import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

dataset_path = "../data/malaria_dataset"

# --------------------------
# DATA GENERATORS
# --------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    dataset_path + "/train",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

validation_data = test_datagen.flow_from_directory(
    dataset_path + "/validation",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    dataset_path + "/test",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

# --------------------------
# LOAD PRETRAINED MODEL
# --------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers:
    layer.trainable = False

# --------------------------
# CUSTOM CLASSIFIER
# --------------------------

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --------------------------
# COMPILE
# --------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------
# TRAIN MODEL
# --------------------------

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

# --------------------------
# TEST MODEL
# --------------------------

test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy:", test_accuracy)

# --------------------------
# SAVE MODEL
# --------------------------

model.save("malaria_model.h5")

print("Model Saved!")