import tensorflow as tf
import tensorflow_datasets as tfds

(train_data, val_data), info = tfds.load(
    "mnist",
    split=["train[:80%]", "train[:80%]"],
    with_info=True,
    as_supervised=True,
)

test_data, info = tfds.load(
    "mnist_corrupted/glass_blur",
    split="test",
    with_info=True,
    as_supervised=True,
)


def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


train_data = (
    train_data.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
)
val_data = val_data.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
test_data = test_data.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
