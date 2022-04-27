import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt


class Config:
    KDE = True
    IMAGE_SIZE_Y = 256
    IMAGE_SIZE_X = 640
    IMAGE_BATCH_SIZE = 24
    SEED = [2023, ]
    EPOCHS = 1000
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    LOG_PATH = f'/home/titoare/Documents/IFAE/asteroid/logs/log_{IMAGE_SIZE_Y}_{IMAGE_SIZE_X}_{IMAGE_BATCH_SIZE}_{int(ts)}/'
    NAME = f"IMG_LOG"


def create_model(config_in):
    np.random.seed(Config.SEED[0])
    tf.random.set_seed(Config.SEED[0])

    # filters: how many features maps will be generated to be optimized.
    # the final Conv2D we set 3 filters because we want our output in RGB.
    # Because we are using Relu ->max(0, v), the values through the layers will be compressed between 0 and v.
    # But in the last layer we have a Sigmoid, therefore will be compressed between 0 and 1.

    new_model = Sequential([
        Conv2D(filters=68, kernel_size=(3, 3), activation="relu", padding='same',
               input_shape=(config_in.IMAGE_SIZE_Y, config_in.IMAGE_SIZE_X, 3)),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(filters=34, kernel_size=(3, 3), activation="relu", padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(filters=24, kernel_size=(3, 3), activation="relu", padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(filters=17, kernel_size=(3, 3), activation="relu", padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(filters=10, kernel_size=(3, 3), activation="relu", padding='same'),
        MaxPooling2D((2, 2), padding='same'),

        Conv2D(filters=10, kernel_size=(3, 3), activation="relu", padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(filters=17, kernel_size=(3, 3), activation="relu", padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(filters=24, kernel_size=(3, 3), activation="relu", padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(filters=34, kernel_size=(3, 3), activation="relu", padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(filters=68, kernel_size=(3, 3), activation="relu", padding='same'),
        UpSampling2D((2, 2)),

        Conv2D(filters=3, kernel_size=(3, 3), activation="sigmoid", padding='same'),

    ])

    # Adam optimization is a stochastic gradient descent method that is
    # based on adaptive estimation of first-order and second-order moments.
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    new_model.compile(optimizer=opt,
                      loss='mean_squared_error',
                      metrics=['mae']
                      )
    print(new_model.summary())

    return new_model


def create_callbacks(path, log_name):
    """"""
    tensorboard = TensorBoard(log_dir=path + log_name)

    filename = "RNN_Final-{epoch:02d}-{loss:.3f}"
    checkpoint = ModelCheckpoint("{}{}.model".format(path, filename,
                                                     monitor='loss',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max'))  # saves only the best ones.
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    return [tensorboard, checkpoint, early_stop]


if __name__ == "__main__":
    config = Config()

    #############################################################################
    # Define generators for training, validation and also anomaly data.
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        '/home/titoare/Documents/IFAE/asteroid/capsule_data_cropped/train/',
        target_size=(config.IMAGE_SIZE_Y, config.IMAGE_SIZE_X),
        batch_size=config.IMAGE_BATCH_SIZE,
        class_mode='input',
        save_format='png',
    )

    validation_generator = datagen.flow_from_directory(
        '/home/titoare/Documents/IFAE/asteroid/capsule_data_cropped/test/',
        target_size=(config.IMAGE_SIZE_Y, config.IMAGE_SIZE_X),
        batch_size=config.IMAGE_BATCH_SIZE,
        class_mode='input',
        save_format='png',
    )

    # Model creation
    model = create_model(config)
    callbacks = create_callbacks(Config.LOG_PATH, Config.NAME)
    history = model.fit(
        train_generator,
        steps_per_epoch=240 // config.IMAGE_BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        validation_steps=24 // config.IMAGE_BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,

    )
    model.save(os.path.join(Config.LOG_PATH, 'complete_model'))

    # Plot loss
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs[7:], loss[7:], 'r', label='Training loss')
    plt.plot(epochs[7:], val_loss[7:], 'y', label='Validation loss')
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # KDE Model:
    if Config.KDE:
        kde_model = Sequential([
            Conv2D(filters=68, kernel_size=(3, 3), activation="relu", padding='same',
                   input_shape=(config.IMAGE_SIZE_Y, config.IMAGE_SIZE_X, 3),
                   weights=model.layers[0].get_weights()),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(filters=34, kernel_size=(3, 3), activation="relu", padding='same',
                   weights=model.layers[2].get_weights()),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(filters=24, kernel_size=(3, 3), activation="relu", padding='same',
                   weights=model.layers[4].get_weights()),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(filters=17, kernel_size=(3, 3), activation="relu", padding='same',
                   weights=model.layers[6].get_weights()),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(filters=10, kernel_size=(3, 3), activation="relu", padding='same',
                   weights=model.layers[8].get_weights()),
            MaxPooling2D((2, 2), padding='same')

        ])

        print(kde_model.summary())
        kde_model.save(os.path.join(Config.LOG_PATH, 'encoder_complete_model'))
