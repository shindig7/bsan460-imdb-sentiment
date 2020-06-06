import re
import pandas as pd
from numpy import vstack
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt


wnl = WordNetLemmatizer()


def clean(text):
    text = re.sub("<br\s/>", "", text)
    text = text.lower()
    text = re.sub("[^a-z ]", " ", text)
    text = re.sub("\s+", " ", text)
    out = []
    for word in word_tokenize(text):
        out.append(wnl.lemmatize(word))
    return " ".join(out)


def load_clean_data():
    df = pd.read_csv("data/imdb_dataset.csv")
    df["review"] = df.review.apply(clean)
    df["sentiment"] = [1 if x == "positive" else 0 for x in df.sentiment]
    return df


def create_model(input_size, layer_size, dropout_rate=0.5):
    model = Sequential(
        [
            Dense(
                layer_size,
                input_shape=(input_size,),
                activation="relu",
                kernel_initializer="uniform",
            ),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    tf_vec = TfidfVectorizer(
        strip_accents="ascii", lowercase=True, stop_words="english"
    )
    df = load_clean_data()
    X = tf_vec.fit_transform(df.review)
    y = df.sentiment.values

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    model = create_model(X.shape[1], 5)

    epochs = 100

    callbacks = [
        EarlyStopping(patience=5, monitor="val_loss", mode="min", verbose=1),
        ModelCheckpoint(
            filepath="models/model.{epoch:02d}-{val_loss:.3f}.h5",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_x,
        vstack(train_y),
        validation_data=(test_x, test_y),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
    )

    pred = model.predict_classes(test_x)

    print("===============Metrics===============")
    print(f"## F1 Score: {f1_score(test_y, pred)}")
    print(f"## Accuracy: {accuracy_score(test_y, pred)}")
    print(f"## Confusion Matrix: \n{confusion_matrix(test_y, pred)}")
    print(f"## Balanced Accuracy: {balanced_accuracy_score(test_y, pred)}")
    print(f"## ROC AUC Score: {roc_auc_score(test_y, pred)}")
    print("=====================================")

    plt.figure(figsize=((15, 10)))
    e = len(history.history["loss"])
    plt.title("Model Loss")
    plt.plot(range(1, e + 1), history.history["loss"], "b.")
    plt.plot(range(1, e + 1), history.history["val_loss"], "r")
    plt.axvline(x=3)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(labels=["Train Loss", "Val Loss"])
    plt.show()


if __name__ == "__main__":
    main()
