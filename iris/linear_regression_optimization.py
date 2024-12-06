# Closed for linear regression for Iris dataset using TensorFlow and TensorFlow Datasets

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

dataset, info = tfds.load("iris", split="train", with_info=True, as_supervised=True)
dataset_size = info.splits["train"].num_examples

dataset = dataset.shuffle(dataset_size, seed=1234)
train_size = int(0.8 * dataset_size)


def preprocess(features, target):
    features = tf.cast(features, tf.float32)
    target = tf.cast(features[2], tf.float32)
    features = tf.concat([features[:2], features[3:]], axis=0)
    return features, target


dataset = dataset.shuffle(dataset_size, seed=1234).map(preprocess)
train_dataset = dataset.take(train_size).batch(train_size)
val_dataset = dataset.skip(train_size).batch(dataset_size - train_size)

model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[3], use_bias=True)])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss="mean_squared_error"
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=250)

val_loss = model.evaluate(val_dataset)
print(f"Validation Mean Squared Error: {val_loss}")

val_features, val_targets = next(iter(val_dataset))
val_predictions = model.predict(val_features)

plt.scatter(val_targets, val_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted Values")
plt.show()

# Plot the training and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
