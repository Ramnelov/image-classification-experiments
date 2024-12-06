# Closed for linear regression for Iris dataset using TensorFlow and TensorFlow Datasets

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset, info = tfds.load("iris", split="train", with_info=True, as_supervised=True)
dataset_size = info.splits["train"].num_examples


def preprocess(features, _):
    target = features[2]
    features = tf.concat([features[:2], features[3:]], axis=0)
    return features, target


dataset = dataset.shuffle(dataset_size, seed=1234).map(preprocess)
train_size = int(0.8 * dataset_size)

train_dataset = dataset.take(train_size).batch(train_size)
val_dataset = dataset.skip(train_size).batch(dataset_size - train_size)


for train_features, train_targets in train_dataset:
    X_train, y_train = train_features, train_targets.numpy().reshape(-1, 1)

for val_features, val_targets in val_dataset:
    X_val, y_val = val_features, val_targets.numpy().reshape(-1, 1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

val_mse = mean_squared_error(y_val, y_pred)
print(f"Validation Mean Squared Error: {val_mse}")

plt.scatter(y_val, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True vs Predicted Values")
plt.show()
