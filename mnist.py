# Two-layered neural network for MNIST dataset using TensorFlow and TensorFlow Datasets

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

(train_data, val_data), info = tfds.load(
    "mnist", split=["train[:80%]", "train[80%:]"], with_info=True, as_supervised=True
)


def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


train_data = (
    train_data.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
)
val_data = val_data.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(train_data, validation_data=val_data, epochs=10)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

min_val_loss = min(history.history["val_loss"])
min_val_loss_epoch = history.history["val_loss"].index(min_val_loss) + 1

max_val_accuracy = max(history.history["val_accuracy"])
max_val_accuracy_epoch = history.history["val_accuracy"].index(max_val_accuracy) + 1

print(f"Minimum Validation Loss: {min_val_loss} at Epoch: {min_val_loss_epoch}")
print(
    f"Maximum Validation Accuracy: {max_val_accuracy} at Epoch: {max_val_accuracy_epoch}"
)

plt.show()
