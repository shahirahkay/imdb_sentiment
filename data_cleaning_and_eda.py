import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# Load your dataset
df = pd.read_csv('imdb_sup.csv')

ax = sns.countplot(x = df.Sentiment)

abs_values = df['Sentiment'].value_counts(ascending=True).values

ax.bar_label(container=ax.containers[0], labels=abs_values)

# Define functions for text preprocessing
lemma = WordNetLemmatizer()
TAG_remove = re.compile(r'<[^>]+>')  # Remove html tags

def remove_tags(text):
    return TAG_remove.sub('', text)

def cleaning(sen):
    s = sen.lower()
    s = remove_tags(s)  
    s = re.sub('[^a-zA-Z]', ' ', s) 
    s = re.sub(r"\s+[a-zA-Z]\s+", ' ', s)  
    s = re.sub(r'\s+', ' ', s)  
    
    # Stopwords removal
    p = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    s = p.sub('', s)
    s = ' '.join([lemma.lemmatize(word) for word in s.split()]) 

    return s

df['Review_clean'] = [cleaning(s) for s in df['Review']]

# Stopwords


from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS) 

to_add = ['film', 'movie']

for i in to_add:
    stopwords.add(i)

df['Review_clean'] = df['Review_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))

X_str = ' '.join(df['Review_clean'])

wordcloud = WordCloud(collocations = False, width = 1600, height = 800,
                      background_color = 'white', stopwords = stopwords,
                     max_words = 150, 
                      random_state = 42).generate(X_str)
print(wordcloud)
plt.figure(figsize = (9,8))
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

df.to_csv('imdb_sup_cleaned.csv')
