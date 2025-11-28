import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from scripts import mylogging as log

ORGFILE = "data/hcv+data (original)/hcvdat0.csv"
TARGET = "Category"
FEATURES = ["AST", "CHE", "ALT", "ALP", "GGT"]
MODELSDIR = Path("models")
MODELNAME = f"hcv-{'-'.join(FEATURES)}"
EPOCHS = 50
BATCH_SIZE = 16
OPTIMIZER = "adam"


# FIXME: maybe fetch data directly instead of storing it? But how Orange will use it?
def fetch_data():
    """
    Check https://archive.ics.uci.edu/dataset/571/hcv+data and click on 'IMPORT IN PYTHON'
    """
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    hcv_data = fetch_ucirepo(id=571)

    # Rename CGT to GGT
    hcv_data["data"]["original"].rename(columns={"CGT": "GGT"}, inplace=True)

    return hcv_data["data"]["original"]


def plot(history):
    log.info("Generating plots...")
    history_dict = history.history
    log.debug(f"history: {history_dict.keys()}")

    loss = history_dict["loss"]
    accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    epochs = range(1, len(accuracy) + 1)

    # Create a figure with 2 subplots
    fig, (plot1, plot2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 colums

    # Plot 1: training and validation loss
    plot1.plot(epochs, loss, "bo", label="Training loss")
    plot1.plot(epochs, val_loss, "b", label="Validation loss")
    plot1.set_title("Training and validation loss")
    plot1.set_xlabel("Epochs")
    plot1.set_ylabel("Loss")
    plot1.legend()

    # Plot 2: training and validation accuracy
    plot2.plot(epochs, accuracy, "ro", label="Training accuracy")
    plot2.plot(epochs, val_accuracy, "r-", label="Validation accuracy")
    plot2.set_title("Training and validation accuracy")
    plot2.set_xlabel("Epochs")
    plot2.set_ylabel("Accuracy")
    plot2.legend()

    plt.savefig(f"{MODELSDIR}/{MODELNAME}-train-val.png")
    plt.close()
    log.info(f"Saved plot: `{MODELSDIR}/{MODELNAME}-train-val.png`")


def train(epochs: int, batch_size: int, convert: bool = False):
    # 1) Load data
    log.info(f"Loading file: {ORGFILE}")
    df = pd.read_csv(ORGFILE, na_values=["?"])
    df = df.rename(columns={df.columns[0]: "ID"})
    # df = fetch_data()
    log.debug("Original data:\n", df)
    log.debug(f"Features: {FEATURES}")
    log.debug(f"Summary of missing values:\n{df.isna().sum()}")

    # 2) Sanity check: drop rows with missing Category value
    df = df.dropna(subset=[TARGET])

    # 3) Extract features + labels (raw)
    x = df[FEATURES].astype(float).values
    y_raw = df[TARGET].values

    # 4) Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    log.debug("Classes: ", le.classes_)

    # 5) Train/validation split
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6) Impute only chosen FEATURES - fit on training, apply to val
    imputer = SimpleImputer(strategy="mean")
    x_train = imputer.fit_transform(x_train)  # calculate the mean
    x_val = imputer.transform(x_val)  # apply the mean from above (no re-calculating)

    # Cast to float32 for TF
    x_train = x_train.astype("float32")
    x_val = x_val.astype("float32")

    # 7) One-hot labels for Keras
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

    # 8) Normalization layer — learn mean/std from TRAINING data only
    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(x_train)

    # 9) Construct small MLP
    inputs = tf.keras.Input(shape=(len(FEATURES),), dtype=tf.float32)
    x = norm(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=OPTIMIZER, loss="categorical_crossentropy", metrics=["accuracy"])

    log.info("Model summary:\n")
    model.summary()

    # 10) Train
    log.info("Training model...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # 11) Evaluate the model
    log.info("Evaluating model...")
    # loss - a number which represents our error, lower values are better
    loss, accuracy = model.evaluate(x_val, y_val)
    log.info(f"Loss: {loss}")
    log.info(f"Accuracy: {accuracy}")

    # 11.1) Create a plot of accuracy and loss over time
    plot(history)

    # 12) Inference on original data for sanity check
    log.info('Testing model with samples from original data')
    # 12.1) Pick 10 rows without NaN in the feature columns
    subset = df.dropna(subset=FEATURES).sample(n=10)  # TIP: bump higher, like 100

    # 12.2) Extract features for the model
    x_sample = subset[FEATURES].to_numpy(dtype=np.float32)
    x_sample = imputer.transform(x_sample)

    # 12.3) Run the model
    predictions = model.predict(x_sample)
    max_idxs = np.argmax(predictions, axis=1)

    # 12.4) Map indices → original diagnoses using the LabelEncoder
    predicted_labels = le.inverse_transform(max_idxs)

    # 12.5) Get the true labels and IDs from the same rows
    actual_labels = subset["Category"].to_numpy()
    ids = subset["ID"].to_numpy()

    # 12.6) Print everything together
    # Disables scientific notation
    np.set_printoptions(suppress=True)
    correct = 0
    for row_id, actual, predicted, probabilities in zip(
        ids, actual_labels, predicted_labels, predictions
    ):
        is_correct = actual == predicted
        if is_correct:
            correct += 1
        log.debug(
            f"ID: {row_id}, actual: {actual}, predicted: {predicted}, probabilities: {probabilities}"
        )

    log.debug(f"Total correctly predicted: {correct}")

    # 12.7) Synthetic data
    synthetic_data = np.array(
        [
            [23.4, 7.8, 18.2, 65.1, 72.0],
            [31.7, 9.5, 25.6, 70.3, 80.4],
            [19.8, 6.9, 15.3, 55.2, 68.7],
            [42.1, 8.2, 33.9, 82.7, 95.5],
            [27.5, 10.1, 22.4, 60.8, 74.3],
            [35.9, 7.4, 29.7, 77.6, 88.1],
            [21.2, 8.7, 17.9, 63.4, 70.9],
            [46.3, 9.9, 38.5, 85.2, 102.6],
            [25.6, 7.1, 20.8, 58.9, 73.2],
            [33.8, 8.5, 27.1, 72.4, 84.7],
        ]
    )
    log.info('Testing model with new "synthetic" data')
    predictions = model.predict(synthetic_data)
    for p in predictions:
        max_idx = np.argmax(p)
        log.debug(f"Prediction: {le.inverse_transform([max_idx])[0]}, probabilities: {p}")

    # 13) Save Keras model to TFLite
    if convert:
        # Dump results to a file
        modelname = f"{MODELSDIR}/{MODELNAME}.tflite"
        results = {
            "tf-version": tf.__version__,
            "model": modelname,
            "epochs": epochs,
            "batch-size": batch_size,
            "optimizer": OPTIMIZER,
            "loss": loss,
            "accuracy": accuracy,
            "history": {
                "training-loss": history.history["loss"],
                "training-accuracy": history.history["accuracy"],
                "validation-loss": history.history["val_loss"],
                "validation-accuracy": history.history["val_accuracy"],
            },
        }

        with open(f"{MODELSDIR}/{MODELNAME}-report.json", "w") as fd:
            json.dump(results, fd, indent=4)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(modelname, "wb") as f:
            f.write(tflite_model)
        log.info(f"Saved TFLite model to `{modelname}`")


def main():
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of epochs to train the model"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--convert", action="store_true", help="Convert model to TFLite")
    args = parser.parse_args()

    # Setup
    if not MODELSDIR.exists():
        MODELSDIR.mkdir(exist_ok=True)

    # Start training
    log.info(f"Tensorflow version: {tf.__version__}")
    train(args.epochs, args.batch_size, args.convert)


if __name__ == "__main__":
    main()
