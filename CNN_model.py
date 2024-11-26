import os

import pandas as pd
import numpy as np

from tqdm import tqdm
import tensorflow as tf
from keras import optimizers, losses, activations, models
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau
)
from keras.layers import (
    Dense,
    Input,
    Dropout,
    Convolution1D,
    MaxPool1D,
    GlobalMaxPool1D,
    GlobalAveragePooling1D,
    concatenate
)
from sklearn.metrics import f1_score, accuracy_score

from constants import (
    FIVE_MIN_TIME_SLICES_FOLDER,
    COMBINED_DATA_FOLDER,
    N_CLASSES,
    TIME_SERIES_LENGTH,
    MODEL_FOLDER,
    N_EPOCHS
)


def reset_tf_session():
    """Clears the TensorFlow session and graph to avoid variable reuse issues."""
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


def prepare_data():
    # Extracted from the Research Paper
    train_file_names = [
        "16273", "16483", "16539", "16773", "16786", "18177", "18184", "19090", "19140",
        "chf03", "chf04", "chf09", "chf12", "chf13", "chf14", "chf15"
    ]
    validation_file_names = [
        "16272", "16420", "17052", "19088", "chf01", "chf05", "chf07", "chf10"
    ]
    test_file_names = [
        "16795", "17453", "19093", "19830", "chf02", "chf06", "chf08", "chf11"
    ]

    if not os.path.exists(COMBINED_DATA_FOLDER + f"train.csv"):
        temp = []
        for file in tqdm(train_file_names):
            temp.append(pd.read_csv(FIVE_MIN_TIME_SLICES_FOLDER + f"{file}.csv").iloc[:, 1:])
        train = pd.concat(temp, ignore_index=True)
        train.to_csv(COMBINED_DATA_FOLDER + f"train.csv")
    else:
        train = pd.read_csv(COMBINED_DATA_FOLDER + f"train.csv").iloc[:, 1:]

    if not os.path.exists(COMBINED_DATA_FOLDER + f"validation.csv"):
        temp = []
        for file in tqdm(validation_file_names):
            temp.append(pd.read_csv(FIVE_MIN_TIME_SLICES_FOLDER + f"{file}.csv").iloc[:, 1:])
        validation = pd.concat(temp, ignore_index=True)
        validation.to_csv(COMBINED_DATA_FOLDER + f"validation.csv")
    else:
        validation = pd.read_csv(COMBINED_DATA_FOLDER + f"validation.csv").iloc[:, 1:]

    if not os.path.exists(COMBINED_DATA_FOLDER + f"test.csv"):
        temp = []
        for file in tqdm(test_file_names):
            temp.append(pd.read_csv(FIVE_MIN_TIME_SLICES_FOLDER + f"{file}.csv").iloc[:, 1:])
        test = pd.concat(temp, ignore_index=True)
        test.to_csv(COMBINED_DATA_FOLDER + f"test.csv")
    else:
        test = pd.read_csv(COMBINED_DATA_FOLDER + f"test.csv").iloc[:, 1:]

    return train, validation, test


def separate_features_and_labels(df):
    return df.iloc[:, :-2].values, df.iloc[:, -2].values, df.iloc[:, -1].values


def get_model():
    nclass = N_CLASSES
    inp = Input(shape=(TIME_SERIES_LENGTH, 1))

    # Block 1
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)

    # Block 2
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)

    # Block 3
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)

    # Block 4
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    # Dense Block 1
    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    print(model.summary())

    return model


def train_model(X, y, X_validation, y_validation):
    model = get_model()

    file_path = MODEL_FOLDER + "CNN_model.keras"

    # Callbacks
    checkpoint = ModelCheckpoint(
        file_path,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    early = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5,
        verbose=1
    )
    redonplat = ReduceLROnPlateau(
        monitor="val_acc",
        mode="max",
        patience=3,
        verbose=2
    )
    callbacks_list = [checkpoint, early, redonplat]

    model.fit(
        X,
        y,
        epochs=N_EPOCHS,
        verbose=5,
        callbacks=callbacks_list,
        validation_data=(X_validation, y_validation)
    )
    model.load_weights(file_path)

    return model


if __name__ == '__main__':
    reset_tf_session()

    train, validation, test = prepare_data()

    X_train, y_train, ann_train = separate_features_and_labels(train)
    X_validation, y_validation, ann_validation = separate_features_and_labels(validation)
    X_test, y_test, ann_test = separate_features_and_labels(test)

    model = train_model(X_train, y_train, X_validation, y_validation)
