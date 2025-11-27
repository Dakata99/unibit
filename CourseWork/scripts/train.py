import json
import argparse
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
MODELSDIR = Path('models')
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
    hcv_data['data']['original'].rename(columns={"CGT": "GGT"}, inplace=True)

    return hcv_data['data']['original']


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
    # df = fetch_data()
    log.debug("Original data:\n", df)
    log.debug("Summary of missing values (features: ):\n", FEATURES, df.isna().sum())

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
    x_train, x_val, y_train_int, y_val_int = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # 6) Impute only chosen FEATURES - fit on training, apply to val
    imputer = SimpleImputer(strategy="mean")
    x_train = imputer.fit_transform(x_train)
    x_val = imputer.transform(x_val)

    # Cast to float32 for TF
    x_train = x_train.astype("float32")
    x_val = x_val.astype("float32")

    # 7) One-hot labels for Keras
    y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=num_classes)
    y_val = tf.keras.utils.to_categorical(y_val_int, num_classes=num_classes)

    # 8) Normalization layer — learn mean/std from TRAINING data only
    norm = tf.keras.layers.Normalization(axis=-1)
    norm.adapt(x_train)

    # 10) Construct small MLP
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
    # loss - a number which represents our error, lower values are better
    log.info("Evaluating model...")
    loss, accuracy = model.evaluate(x_val, y_val)
    log.info(f"Loss: {loss}")
    log.info(f"Accuracy: {accuracy}")

    # 11.1) Create a plot of accuracy and loss over time
    plot(history)

    # 11.2) TODO: some inference on new data???
    # examples = tf.constant(
    #     # "AST", "CHE", "ALT", "ALP", "GGT"
    #     [35, 7200, 42, 110, 55] # U/L
    # )
    # prediction = model.predict(examples)
    # log.debug(f'Prediction: {prediction}')

    # 12) Save Keras model to TFLite
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
