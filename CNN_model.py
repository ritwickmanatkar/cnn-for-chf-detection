import os

import pandas as pd
import numpy as np

from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    ReLU,
    Flatten,
    Dense
)
from keras import optimizers
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau
)
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from constants import (
    MODEL_FOLDER,
    N_EPOCHS,
    BATCH_SIZE
)


def reset_tf_session():
    """Clears the TensorFlow session and graph to avoid variable reuse issues."""
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


def prepare_data(data_folder: str, save_to_folder: str):
    """ Splits data into train, validation and test. This is based on the research paper. (Fig. 2)

    :param data_folder: Data Read Folder
    :param save_to_folder: Data Save Folder

    :return: train, validation, and test dataframes
    """
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

    if not os.path.exists(save_to_folder + f"train.csv"):
        temp = []
        for file in tqdm(train_file_names):
            temp.append(pd.read_csv(data_folder + f"{file}.csv").iloc[:, 1:])
        train = pd.concat(temp, ignore_index=True)
        train.to_csv(save_to_folder + f"train.csv")
    else:
        train = pd.read_csv(save_to_folder + f"train.csv").iloc[:, 1:]

    if not os.path.exists(save_to_folder + f"validation.csv"):
        temp = []
        for file in tqdm(validation_file_names):
            temp.append(pd.read_csv(data_folder + f"{file}.csv").iloc[:, 1:])
        validation = pd.concat(temp, ignore_index=True)
        validation.to_csv(save_to_folder + f"validation.csv")
    else:
        validation = pd.read_csv(save_to_folder + f"validation.csv").iloc[:, 1:]

    if not os.path.exists(save_to_folder + f"test.csv"):
        temp = []
        for file in tqdm(test_file_names):
            temp.append(pd.read_csv(data_folder + f"{file}.csv").iloc[:, 1:])
        test = pd.concat(temp, ignore_index=True)
        test.to_csv(save_to_folder + f"test.csv")
    else:
        test = pd.read_csv(save_to_folder + f"test.csv").iloc[:, 1:]

    return train, validation, test


def separate_features_and_labels(df):
    """ Separates given df into features and labels

    :param df: Dataframe

    :return: features and labels numpy arrays
    """
    return df.iloc[:, :-1].values, df.iloc[:, -1].values


def get_model(n_classes, n_features):
    """ This model is defined on the basis of the research paper.(Section 2.5)
    https://sci-hub.usualwant.com/10.1016/j.bspc.2019.101597

    :param n_classes: no of classes we are working with
    :param n_features: no of samples that are used for training.

    :return: Model
    """
    inp = Input(shape=(n_features, 1))

    x = Conv1D(filters=20, kernel_size=10, strides=1, padding="valid", name="Conv_Block1")(inp)
    x = BatchNormalization(name="BatchNorm_Block1")(x)
    x = ReLU(name="ReLU_Block1")(x)

    x = Conv1D(filters=20, kernel_size=15, strides=1, padding="valid", name="Conv_Block2")(x)
    x = BatchNormalization(name="BatchNorm_Block2")(x)
    x = ReLU(name="ReLU_Block2")(x)

    x = Conv1D(filters=20, kernel_size=20, strides=1, padding="valid", name="Conv_Block3")(x)
    x = BatchNormalization(name="BatchNorm_Block3")(x)
    x = ReLU(name="ReLU_Block3")(x)

    x = Flatten(name="Flatten")(x)

    x = Dense(30, activation="relu", name="MLP_Hidden_Layer")(x)
    x = Dense(n_classes, activation="softmax", name="Output_Layer")(
        x)

    model = Model(inputs=inp, outputs=x, name="ECG_Classifier")

    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # Compile the model
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Summary
    model.summary()

    return model


def train_model(X, y, X_validation, y_validation, n_classes):
    """ This function is used to train the model.

    :param X: Features
    :param y: Labels
    :param X_validation: Validation Features
    :param y_validation: Validation Labels
    :param n_classes: No of classes we are dealing with

    :return: trained model
    """
    model = get_model(n_classes, X.shape[1])

    file_path = MODEL_FOLDER + "CNN_model.keras"

    # Callbacks
    checkpoint = ModelCheckpoint(
        file_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    early = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        verbose=1
    )
    redonplat = ReduceLROnPlateau(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        verbose=2
    )
    callbacks_list = [checkpoint, early, redonplat]

    model.fit(
        X,
        y,
        epochs=N_EPOCHS,
        verbose=1,
        callbacks=callbacks_list,
        validation_data=(X_validation, y_validation),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    try:
        # best_model = get_model(n_classes, X.shape[1])
        model.load_weights(file_path)

        return model
    except Exception as err:
        print(f"Error loading best model: {err}.\nReturning the model from the last epoch.")
        return model


def evaluate_model(model, X_test, y_test):
    """ Returns an evaluation of the model.

    :param model: Model in question
    :param X_test: Test features
    :param y_test: Test Labels

    :return: None
    """
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)

    f1 = f1_score(y_test, pred_test, average="macro")

    print("Test f1 score : %s " % f1)

    acc = accuracy_score(y_test, pred_test)

    print("Test accuracy score : %s " % acc)

    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Sinus Rhythm", "Congestive Heart Failure"])
    disp.plot()
    plt.show()
