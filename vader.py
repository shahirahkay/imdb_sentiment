import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Load your dataset
df = pd.read_csv('imdb_sup_cleaned.csv')

analyser = SentimentIntensityAnalyzer()

# all polarity including neutral

def predict_sentiment(text):
    output_dict = analyser.polarity_scores(text)
    return output_dict['compound']

df['Review_clean'] = df['Review_clean'].astype(str)
df['compound_score'] = df['Review_clean'].apply(predict_sentiment)

# Categorise compound score
df['vader_prediction'] = df['compound_score'].apply(lambda x: 1 if x > 0.5 else 0)

# Accuracy
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

accuracy = accuracy_score(df['Sentiment'], df['vader_prediction'])

print("Accuracy: {}\n".format(accuracy))
print("F1 score: {}\n".format(f1_score(df['Sentiment'], df['vader_prediction'], average='weighted')))
print("precision score: {}\n".format(precision_score(df['Sentiment'], df['vader_prediction'], average='weighted')))
print("recall score: {}\n".format(recall_score(df['Sentiment'], df['vader_prediction'], average='weighted')))
print(classification_report(df['Sentiment'], df['vader_prediction']))
# print(confusion_matrix(df['Sentiment'], df['vader_prediction']))
