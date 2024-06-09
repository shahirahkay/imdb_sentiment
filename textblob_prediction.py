import pandas as pd
from textblob import TextBlob, Word, Blobber

df = pd.read_csv('imdb_sup_cleaned.csv')

def format_output(score):

    polarity = "positive"

    if (score > 0):
        polarity = 1

    elif (score <= 0):
        polarity = 0

    return polarity

def textblob_sentiment_scores(text):
    textblob_sentiment = TextBlob(text)
    score = textblob_sentiment.sentiment.polarity
    return format_output(score)

df['Review_clean'] = df['Review_clean'].astype(str)
df['TextBlob_sentiment'] = df['Review_clean'].apply(textblob_sentiment_scores)


# Accuracy
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

accuracy = accuracy_score(df['Sentiment'], df['TextBlob_sentiment'])

print("Accuracy: {}\n".format(accuracy))
print("F1 score: {}\n".format(f1_score(df['Sentiment'], df['TextBlob_sentiment'], average='weighted')))
print("precision score: {}\n".format(precision_score(df['Sentiment'], df['TextBlob_sentiment'], average='weighted')))
print("recall score: {}\n".format(recall_score(df['Sentiment'], df['TextBlob_sentiment'], average='weighted')))
print(classification_report(df['Sentiment'], df['TextBlob_sentiment']))
# print(confusion_matrix(df['Sentiment'], df['TextBlob_sentiment']))
