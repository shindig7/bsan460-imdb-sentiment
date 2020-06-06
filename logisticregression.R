setwd("~/Drexel/Senior_19_20/Spring/BSAN460")
library(dplyr)
library(tm)
library(textstem)
library(xgboost)
library(tidyverse)
library(caret)

set.seed(8675309)

df <- read.csv("Data/IMDB Dataset.csv")
binarize <- function(x) {if (x == "positive") {1} else {0}}
df$sentiment <- apply(array(df$sentiment), MARGIN=1, FUN=binarize)

# Remove break tags, lower strings
df <- df %>% mutate(review = gsub("<br />", " ", review))
df <- df %>% mutate(review = tolower(review))

# Remove stopwords
stopwords_regex = paste(stopwords('en'), collapse = '\\b|\\b')
df <- df %>% mutate(review = gsub(stopwords_regex, " ", review))

# Remove punctutation, digits, multiple spaces
df <- df %>% mutate(review = gsub('[[:punct:]]', ' ', review))
df <- df %>% mutate(review = gsub('[[:digit:]]','', review))
df <- df %>% mutate(review = gsub("\\s{2,}", " ", review))

# Lemmatize strings
df <- df %>% mutate(review = lemmatize_strings(review))

# Create corpus and DocumentTermMatrix
corp <- Corpus(VectorSource(df$review))
tf <- DocumentTermMatrix(corp, control = list(weighting = weightTfIdf))
tf

# Remove sparse terms
small_tf <- removeSparseTerms(tf, sparse=.97)
small_tf

data <- as.data.frame(as.matrix(small_tf))
data <- cbind(sentiment = df$sentiment, data)

# Remove large objects
rm(corp, df, small_tf, tf)

# Split training and test data sets
train_size <- floor(0.8 * nrow(data))
train_idx <- sample(nrow(data), size=train_size)

traindf <- data[train_idx, ]
testdf <- data[-train_idx, ]

# XGBoost Logistic Regression, binary classification metrics
model <- xgboost::xgboost(data=data.matrix(traindf[,-1]),
                        label=data.matrix(traindf$sentiment),
                        eval_metric="error",
                        objective="binary:logistic",
                        nrounds=1000)

# Plot of training error over time
model$evaluation_log %>% 
  as_tibble() %>% 
  ggplot(aes(x= iter, y = train_error)) + 
  geom_line() + 
  labs(title = "Training Error")

# Predict class, rounding up to 1 and down to 0
pred <- ifelse(predict(model, data.matrix(testdf[, -1])) > 0.5, 1, 0)
# Create data.frame with predicted and actual values
resdf <- data.frame(cbind(model_pred=pred, actual=testdf$sentiment))

# Confusion matrix
confusionMatrix(data=as.factor(resdf$model_pred), reference=as.factor(resdf$actual), positive='1', mode="everything")
