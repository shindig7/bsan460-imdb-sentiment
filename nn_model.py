import re
import pandas as pd
from numpy import vstack
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import Input

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
            Dense(layer_size, activation="relu", kernel_initializer="uniform"),
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

    model = create_model(X.shape[1], 25)

    history = model.fit(
        train_x,
        vstack(train_y),
        validation_data=(test_x, test_y),
        epochs=50,
        batch_size=128,
    )

    print(history.history)


if __name__ == "__main__":
    main()
