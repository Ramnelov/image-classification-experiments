# Convolutional Neural Network for MNIST dataset using TensorFlow and TensorFlow Datasets

import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess_data import train_data, val_data

if __name__ == "__main__":

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    history = model.fit(
        train_data, validation_data=val_data, epochs=20, callbacks=[early_stopping]
    )

    model.save("mnist/models/cnn_model.h5")

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
