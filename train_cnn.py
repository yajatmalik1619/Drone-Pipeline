import tensorflow as tf
from keras import layers, models
import pathlib

DATASET_DIR = pathlib.Path("C:/Users/yajat/Code/drone_pipeline/dataset_grayscale")
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
SEED = 0

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Training classes:", class_names)
print("Number of classes:", num_classes)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=(IMG_SIZE, IMG_SIZE, 3),
#     include_top=False,
#     weights="imagenet"
# )

# base_model.trainable = False

# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.BatchNormalization(),

#     layers.Dense(128, activation="relu"),
#     layers.Dropout(0.5),

#     layers.Dense(num_classes, activation="softmax")
# ])

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(256, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.GlobalAveragePooling2D(),

    layers.Dense(32, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

loss, acc = model.evaluate(val_ds)
print("Validation accuracy:", acc)

model.save("C:/Users/yajat/Code/drone_pipeline/gesture_cnn.h5")

with open("C:/Users/yajat/Code/drone_pipeline/classes.txt", "w") as f:
    for c in class_names:
        f.write(c + "\n")
