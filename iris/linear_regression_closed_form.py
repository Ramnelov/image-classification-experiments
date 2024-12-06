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

train_features, train_targets = next(iter(train_dataset))
val_features, val_targets = next(iter(val_dataset))
train_features = np.c_[np.ones((train_features.shape[0], 1)), train_features]
val_features = np.c_[np.ones((val_features.shape[0], 1)), val_features]

X, y = train_features, train_targets.numpy().reshape(-1, 1)
theta = np.linalg.inv(X.T @ X) @ X.T @ y

val_predictions = val_features @ theta
val_mse = np.mean((val_predictions - val_targets.numpy().reshape(-1, 1)) ** 2)
print(f"Validation Mean Squared Error: {val_mse}")

plt.scatter(val_targets, val_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted Values")
plt.show()
