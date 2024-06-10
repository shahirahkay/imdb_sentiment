# IMDB Reviews Sentiment Analysis: Project Overview
* Created a tool that predicts the sentiment of reviews to categorise the reviews automatically.  
* Preprocessed data to remove noise. 
* Optimised lexicon-based approaches (VADER and TextBlob) and a neural network (LSTM) to find the best prediction model.

## Resources used
  * **Dataset**: https://ai.stanford.edu/~amaas/data/sentiment/
  * **Python Version**: 3.10
  * **Packages**: pandas, matplotlib, seaborn, nltk, re, wordcloud, sklearn, textblob, keras, tensorflow

## Dataset
This dataset consists of 50K movie reviews from IMDB. Attributes form this dataset are:
* **Review** - reviews left by viewers
* **Rating** - scores given by viewers
* **Sentiment** - aggregrate from rating. 1 indicates positive whereas 0 indicates negative sentiment. 

## Data Cleaning
Data pre-processing techniques are important to provide clean textual data and ensure better accuracy of the sentiment analysis.

* Tokenisation - splitting the textual data into a list of tokens 
* Removal of punctuations, special characters and numeric values
* Conversion to lower case 
* Removal of stopwords such as *'I'* and *'the'* which are commonly used in English and do not have sentimental insights. 
* Lemmatisation - returning a word to its dictionary form. 

## Explanatory Data Analysis (EDA)
* Finding the distribution of positive and negative sentiment towards movie reviews
  
   ![alt text](https://github.com/shahirahkay/imdb_sentiment/blob/master/Sentiment%20from%20score.png "score")
  
* Built wordcloud to determine the most frequent words used in the reviews.
  
 ![alt text](https://github.com/shahirahkay/imdb_sentiment/blob/master/wordcloud.png "wordcloud")


## Model Building
Three models were chosen in this project and evaluated using accuracy. I chose accuracy as the polarity of the dataset is balanced. 
* Lexicon-based approach: First, I transformed textual data into a string. Then clasified sentiment polarity of the textual data based on their threshold scores.
  * **VADER** - calculates compound score by comparing positive and negative score.
  * **TextBlob** - gives polarity score to textual data.
  
* Neural network (**LSTM**): I split the data into train and tests sets with a test size of 30%. Then, I tokenise them based on word frequency and padded the sequences for uniform length. After that, I built and trained an LSTM-based neural network model to perform sentiment analysis, using embedding and dropout layers to prevent overfitting. 

## Model Performance
The LSTM model outperformed the other approaches on the accuracy. 
* **VADER** - Accuracy: 70.7%
* **TextBlob** - Accuracy: 69.9%
* **Neural Network (LSTM)** - Accuracy: 86.9%


