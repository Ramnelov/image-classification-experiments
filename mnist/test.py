import random

import matplotlib.pyplot as plt
import tensorflow as tf

# Assuming test_data is already defined and preprocessed in tensorflow_2_layer_neural_network.py
from preprocess_data import test_data

model_2_layer_nn = tf.keras.models.load_model("mnist/models/2_layer_nn_model.h5")
model_cnn = tf.keras.models.load_model("mnist/models/cnn_model.h5")

random_samples = random.sample(list(test_data.unbatch().as_numpy_iterator()), 9)

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(random_samples):
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    pred_2_layer_nn = model_2_layer_nn.predict(image)
    pred_cnn = model_cnn.predict(image)
    pred_label_2_layer_nn = tf.argmax(pred_2_layer_nn, axis=1).numpy()[0]
    pred_label_cnn = tf.argmax(pred_cnn, axis=1).numpy()[0]

    plt.subplot(3, 3, i + 1)
    plt.imshow(image[0, :, :, 0], cmap="gray")
    plt.title(
        f"True: {label}, 2-Layer NN: {pred_label_2_layer_nn}, CNN: {pred_label_cnn}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()
