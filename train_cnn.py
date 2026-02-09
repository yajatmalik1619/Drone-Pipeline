import tensorflow as tf
from keras import layers, models
import pathlib

DATASET_DIR = pathlib.Path("C:\\Users\\Kanishka\\Code\\Drone-Pipeline\\dataset")
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

ALLOWED_CLASSES = ["fist", "palm", "peace", "thumbs_up"]
class_to_index = {name: i for i, name in enumerate(ALLOWED_CLASSES)}

image_paths = []
labels = []

for cls in ALLOWED_CLASSES:
    for img_path in (DATASET_DIR / cls).glob("*"):
        image_paths.append(str(img_path))
        labels.append(class_to_index[cls])

image_paths = tf.constant(image_paths)
labels = tf.constant(labels)

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.shuffle(len(image_paths), seed=SEED)

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

val_size = int(0.2 * len(image_paths))
val_ds = dataset.take(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_ds = dataset.skip(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

class_names = ALLOWED_CLASSES
num_classes = len(class_names)

print("Training classes:", class_names)
print("Total samples:", len(image_paths))


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation="softmax")
])

model.build((None, IMG_SIZE, IMG_SIZE, 3))
model.summary()


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

model.save("C:\\Users\\Kanishka\\Code\\Drone-Pipeline\\gesture_cnn.h5")

with open("C:\\Users\\Kanishka\\Code\\Drone-Pipeline\\classes.txt", "w") as f:
    for c in class_names:
        f.write(c + "\n")
