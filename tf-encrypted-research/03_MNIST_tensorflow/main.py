import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import time as t

def normalize_img(image, label):
    """Normalize images: `uint8` -> `float32`"""
    return (tf.cast(image,tf.float32) / 255.0, label)

def main():
    index = 1
    start = t.time()

    print(f"{index}. Loading the training data...")
    # Load the training data
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    end = t.time()

    print(f"{index}. Done in {end - start} seconds!")
    index += 1

    print(f"{index}. Preparing main training set...")
    start = t.time()

    # Prepare training dataset - Main training set
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    
    end = t.time()
    print(f"{index}. Done in {end - start} seconds!")
    index += 1

    print(f"{index}. Preparing test training set...")
    start = t.time()

    # Prepare training dataset - Test training set
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    end = t.time()
    print(f"{index}. Done in {end - start} seconds!")
    index += 1

    print(f"{index}. Building Keras model...")
    start = t.time()

    # Plug the TFDS input pipeline into a simple Keras model, compile the model and train it
    model = tf.keras.models.Sequential([

        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    end = t.time()
    print(f"{index}. Done in {end - start} seconds!")
    index += 1

    print(f"{index} Compiling Keras model")
    start = t.time()

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    end = t.time()
    print(f"{index}. Done in {end - start} seconds!")
    index += 1

    print(f"{index} Fitting Keras model")
    start = t.time()

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

    end = t.time()
    print(f"{index}. Done in {end - start} seconds!")
    index += 1

if __name__ == "__main__":
    main()


