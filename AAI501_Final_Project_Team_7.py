# AAI501 Final Project, Team 7
# Issa Ennab, Matt Purkeypile
# University of San Diego
# Spring 2024

import pandas as pd
import seaborn as sb
import numpy as nump
import statsmodels.formula.api as sm
import matplotlib.pyplot as plot
import nltk

# pull the data locally (data from http://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com)
AllDataTrain = pd.read_csv(".\\drugsComTrain_raw.tsv", sep = "\t")
print(str(AllDataTrain.head()))
print(str(AllDataTrain.shape))

### Exploratory Data Analysis (can refactor into functions later) ###

#construct the historgram
plot.hist(AllDataTrain["rating"], density = False, bins = 10)
plot.ylabel('Rating (number of stars)')
plot.xlabel('Number of occurrences')
plot.title('Histogram all drug ratings')
#plot.show() #need in order to show when running in the console

plot.hist(AllDataTrain["usefulCount"], density = False, bins = 50)
plot.ylabel('Useful count (number of likes)')
plot.xlabel('Number of occurrences')
plot.title('Histogram all drug useful counts')
#plot.show() #need in order to show when running in the console

#n, mean, standard deviation, and five number summary (min, lower quartile, median, upper quartile, max)
print("Ratings description")
print(AllDataTrain["rating"].describe())
print("\nUseful count description")
print(AllDataTrain["usefulCount"].describe())

#AllDataTest = pd.read_csv(".\\drugsComTest_raw.tsv", sep = "\t")

### Start doing NLP (can also refactor and encapsulate this later) ###

# These instructions are from the From the AAI TA:
# If you have a specific score that you want to use to train a model for sentiment 
# analysis, rather than relying solely on pre-built tools like NLTK's 
# SentimentIntensityAnalyzer, you will have to engage in a more custom machine 
# learning (ML) process. Here’s a high-level outline of the steps you might follow:
# 1.  ​Prepare Your Dataset:
#     •   You'll need a dataset consisting of review texts and corresponding 
#     sentiment scores. Make sure your data is clean and preprocessed. This preprocessing 
#     might include tokenization, removing stop words, stemming, lemmatization, etc.

# More specific instructions for preparing the data set:
        # ​2 Cleaning the Data:
        #    •   Text data is often messy and requires cleaning. This step might include 
        #    removing unnecessary elements such as HTML tags, special characters, and numbers 
        #    that might not be relevant to sentiment analysis.
        #    •   Normalize the text: This could include converting all text to lowercase to 
        #    ensure consistency.
AllDataTrain["review"] = AllDataTrain["review"].str.lower()
print(str(AllDataTrain["review"].head(30)))

        # 3.  ​Text Tokenization:
        #    •   Tokenization involves splitting the text into individual words (tokens). 
        #    This is crucial for analyzing the text and for most feature extraction techniques.
        # 4.  ​Removing Stop Words:
        #    •   Stop words are common words (e.g., "the", "is", "at") that are usually 
        #    irrelevant for sentiment analysis. Removing these can reduce the dimensionality 
        #    of the data and focus on more meaningful words.
nltk.download('stopwords')
StopWordList = nltk.corpus.stopwords.words("english")
StopWordList.remove("no")    # keep negation
StopWordList.remove("not")
        # 5.  ​Stemming and Lemmatization:
        #    •   These processes aim to reduce words to their base or root form. For example, 
        #    “running”, “runs”, “ran” all stem to “run”. It helps in generalizing different forms 
        #    of a word to a common base form.
        # 6.  ​Feature Selection:
        #    •   Not all words in the text may be useful for determining sentiment. Feature selection 
        #    involves choosing the most relevant features (words or phrases) to use in your model. Techniques 
        #    such as TF-IDF can help identify the importance of words in the context of the dataset.
        # 7.  ​Vectorization:
        #    •   Machine learning models do not understand text directly; thus, it's essential to convert 
        #    the text into a numerical format. This can be achieved through methods like Bag of Words, 
        #    TF-IDF, or word embeddings.
        #    •   ​Bag of Words​ counts the occurrence of words within the text and represents the 
        #    text as a vector based on these counts.
        #    •   ​TF-IDF​ adjusts these counts based on how unique a word is to the document within 
        #    the larger corpus, helping identify more meaningful words.
        #    •   ​Word embeddings​ provide a dense representation where similar words have similar encoding. 
        #    Pre-trained embeddings like Word2Vec or GloVe can capture semantic relationships between words.
        # 8.  ​Splitting the Dataset:
        #    •   Once the data is prepared, it needs to be split into training and testing (and 
        #    possibly validation) sets. This ensures that the model can be trained on one set of data and 
        #    tested on unseen data to evaluate its performance accurately.




# 2.  ​Feature Extraction:
#     •   Convert text data into a numerical format that ML models can understand. 
#     Common approaches include Bag of Words, TF-IDF (Term Frequency-Inverse Document 
#     Frequency), or using embeddings like Word2Vec or GloVe.
# 3.  ​Split the Dataset:
#     •   Divide your dataset into training and testing sets. A common split ratio is 
#     80% for training and 20% for testing. This ensures you have unseen data to test 
#     the performance of your model.
# 4.  ​Choose a Model:
#     •   Select an appropriate ML model. For sentiment analysis, models like Logistic 
#     Regression, Naive Bayes, Support Vector Machines, or even deep learning models 
#     like recurrent neural networks (RNN) can be suitable. The choice of the model 
#     might depend on the complexity of your data and the problem at hand.
# 5.  ​Train the Model:
#     •   Train your chosen model on the training dataset. This involves feeding the 
#     input features (extracted in step 2) and the corresponding sentiment scores to the 
#     model so it can learn to associate the input features with the desired output.
# 6.  ​Evaluate the Model:
#     •   Use the model to make predictions on your test set and compare the predicted 
#     sentiment scores against the actual scores present in your data. Common evaluation 
#     metrics include accuracy, precision, recall, F1-score, and mean squared error, depending 
#     on whether your sentiment scores are categorical or continuous.
# 7.  ​Fine-tune and Optimize:
#     •   Based on the performance, you might need to go back and adjust your model, feature 
#     extraction method, or even preprocess your data differently. This could involve parameter 
#     tuning, using different models, or employing techniques like cross-validation for a more robust evaluation.
# 8.  ​Deployment:
#     •   Once satisfied with model performance, you can use this model to predict the sentiment of new, unseen reviews.
# Remember, this is a simplified overview, and the process involves various nuances and potential 
# challenges, especially regarding data preprocessing, choosing the right model, and feature extraction methods.
#  Exploring existing literature and case studies on sentiment analysis can also give you insights 
#  into best practices and innovative approaches.
# Additionally, libraries like scikit-learn for traditional ML approaches or tensorflow/keras for 
# neural networks can be invaluable in this process.
