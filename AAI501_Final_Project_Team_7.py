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
import re
import unicodedata
from bs4 import BeautifulSoup
from sklearn.linear_model import Ridge

# pull the data locally (data from http://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com)
AllDataTrain = pd.read_csv(".\\drugsComTrain_raw.tsv", sep = "\t")
AllDataTest = pd.read_csv(".\\drugsComTest_raw.tsv", sep = "\t")
#print(str(AllDataTrain.head()))
#print(str(AllDataTrain.shape))

# Strip out HTML tags
# based on pages 118 - 119 of
# Sarkar, D. (2019). Text Analytics with Python: A Practitioner's Guide to Natural Language Processing. Apress.
def StripHtmlTags(Text):
    Soup = BeautifulSoup(Text, "html.parser")
    [s.extract() for s in Soup(["iframe", "script"])]
    StrippedText = Soup.getText()
    StrippedText = re.sub(r"[\r|\n|\r\n]+", "\n", StrippedText)
    return StrippedText

# based on pages 135 of
# Sarkar, D. (2019). Text Analytics with Python: A Practitioner's Guide to Natural Language Processing. Apress.
def RemoveAccentedChars(Text):
    Text = unicodedata.normalize("NFKD", Text).encode("ascii", "ignore").decode("utf-8", "ignore")
    return Text

# based on pages 138 of
# Sarkar, D. (2019). Text Analytics with Python: A Practitioner's Guide to Natural Language Processing. Apress.
def RemoveSpecialChars(Text, RemoveDigits = False):
    Pattern = r"[^a-zA-z0-9\s]" if not RemoveDigits else r"[^a-zA-z\s]"
    Text = re.sub(Pattern, "", Text)
    return Text

# Taken from the following site on March 29, 2024 0241 Zulu
# https://github.com/dipanjanS/text-analytics-with-python/blob/master/New-Second-Edition/Ch03%20-%20Processing%20and%20Understanding%20Text/contractions.py
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

# based on pages 137 of
# Sarkar, D. (2019). Text Analytics with Python: A Practitioner's Guide to Natural Language Processing. Apress.
def ExpandContractions(Text, ContractionMapping = CONTRACTION_MAP):
    ContractionsPattern = re.compile("({})".format("|".join(ContractionMapping.keys())), flags = re.IGNORECASE | re.DOTALL)
    def ExpandMatch(Contraction):
        Match = Contraction.group(0)
        FirstChar = Match[0]
        ExpandedContraction = ContractionMapping.get(Match) \
                              if ContractionMapping.get(Match) \
                              else ContractionMapping.get(Match.lower())
        ExpandedContraction = FirstChar+ExpandedContraction[1:]
        return ExpandedContraction
    ExpandedText = ContractionsPattern.sub(ExpandMatch, Text)
    ExpandedText = re.sub("'", "", ExpandedText)
    return ExpandedText
    


# Encapsulate the exploratory data analysis (EDA)
def ExploratoryDataAnalysis():
    #construct the historgram
    plot.hist(AllDataTrain["rating"], density = False, bins = 10)
    plot.ylabel('Rating (number of stars)')
    plot.xlabel('Number of occurrences')
    plot.title('Histogram all drug ratings')
    plot.show() #need in order to show when running in the console

    plot.hist(AllDataTrain["usefulCount"], density = False, bins = 50)
    plot.ylabel('Useful count (number of likes)')
    plot.xlabel('Number of occurrences')
    plot.title('Histogram all drug useful counts')
    plot.show() #need in order to show when running in the console

    #n, mean, standard deviation, and five number summary (min, lower quartile, median, upper quartile, max)
    print("Ratings description")
    print(AllDataTrain["rating"].describe())
    print("\nUseful count description")
    print(AllDataTrain["usefulCount"].describe())

def AnalysisTheHardWay():
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
    AllDataTrain["review"] = AllDataTrain["review"].apply(StripHtmlTags)
    AllDataTrain["review"] = AllDataTrain["review"].apply(ExpandContractions)
    AllDataTrain["review"] = AllDataTrain["review"].apply(RemoveAccentedChars)
    AllDataTrain["review"] = AllDataTrain["review"].apply(RemoveSpecialChars)
    print(str(AllDataTrain["review"].head(30)))

            # 3.  ​Text Tokenization:
            #    •   Tokenization involves splitting the text into individual words (tokens). 
            #    This is crucial for analyzing the text and for most feature extraction techniques.
    # only doing word tokenization. Do we need to do sentence?
    nltk.download('punkt')
    DefaultWordTokenizer = nltk.word_tokenize
    AllDataTrain["words"] = AllDataTrain["review"].apply(DefaultWordTokenizer)
    print(str(AllDataTrain["words"].head(30)))

            # 4.  ​Removing Stop Words:
            #    •   Stop words are common words (e.g., "the", "is", "at") that are usually 
            #    irrelevant for sentiment analysis. Removing these can reduce the dimensionality 
            #    of the data and focus on more meaningful words.
    nltk.download('stopwords')
    StopWordList = nltk.corpus.stopwords.words("english")
    StopWordList.remove("no")    # keep negation
    StopWordList.remove("not")

    #TODO: FINISH!

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



# go off of the neural net in module 5
# import tensorflow_hub as hub
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# model = hub.load(module_url)

# # clean data
# print("Cleaning data...\n")
# AllDataTrain["review"] = AllDataTrain["review"].str.lower()
# AllDataTrain["review"] = AllDataTrain["review"].apply(StripHtmlTags)
# AllDataTrain["review"] = AllDataTrain["review"].apply(ExpandContractions)
# AllDataTrain["review"] = AllDataTrain["review"].apply(RemoveAccentedChars)
# AllDataTrain["review"] = AllDataTrain["review"].apply(RemoveSpecialChars)

# AllDataTest["review"] = AllDataTest["review"].str.lower()
# AllDataTest["review"] = AllDataTest["review"].apply(StripHtmlTags)
# AllDataTest["review"] = AllDataTest["review"].apply(ExpandContractions)
# AllDataTest["review"] = AllDataTest["review"].apply(RemoveAccentedChars)
# AllDataTest["review"] = AllDataTest["review"].apply(RemoveSpecialChars)
# train the model
# print("Creating testing model...\n")
# TestModel = model(AllDataTest["review"])
# print("Creating training model...\n")
# TrainModel = model(AllDataTrain["review"])

# fit and graph
# print("Fitting and graphing...")
# TrainRidge = Ridge()
# TrainRidge.fit(TrainModel, AllDataTrain["rating"])
# PredictedTestValues = TrainRidge.predict(TrainModel)
# print("Coefficient of determination (rating): " + str(TrainRidge.score(TestModel, AllDataTest["rating"])))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge

# clean data
# print("Cleaning data...\n")
AllDataTrain["review"] = AllDataTrain["review"].str.lower()
AllDataTrain["review"] = AllDataTrain["review"].str.replace("\0", "")
AllDataTrain["review"] = AllDataTrain["review"].apply(StripHtmlTags)
AllDataTrain["review"] = AllDataTrain["review"].apply(ExpandContractions)
AllDataTrain["review"] = AllDataTrain["review"].apply(RemoveAccentedChars)
AllDataTrain["review"] = AllDataTrain["review"].apply(RemoveSpecialChars)

AllDataTest["review"] = AllDataTest["review"].str.lower()
AllDataTrain["review"] = AllDataTrain["review"].str.replace("\0", "")
AllDataTest["review"] = AllDataTest["review"].apply(StripHtmlTags)
AllDataTest["review"] = AllDataTest["review"].apply(ExpandContractions)
AllDataTest["review"] = AllDataTest["review"].apply(RemoveAccentedChars)
AllDataTest["review"] = AllDataTest["review"].apply(RemoveSpecialChars)

CVTrainTest = CountVectorizer()

# need to do across both data sets so transform() is on the same size
AllDataTotal = AllDataTrain + AllDataTest
CVTrainTest.fit(AllDataTotal["review"].values.astype("str"))
CVTrainXform = CVTrainTest.transform(AllDataTrain["review"])
CVTestXform = CVTrainTest.transform(AllDataTest["review"])

TrainRidge = Ridge()
TrainRidge.fit(CVTrainXform, AllDataTrain["rating"])
PredictedTestValues = TrainRidge.predict(CVTestXform)
print("CV coefficient of determination (rating): " + str(TrainRidge.score(CVTestXform, AllDataTest["rating"])))

plot.clf()    # clear what was there from before
plot.hist(PredictedTestValues, bins = 10, label = "Predicted rating", alpha = 0.5)
plot.hist(AllDataTest["rating"], bins = 10, label = "Ground truth rating", alpha = 0.5)
plot.legend()
plot.xlabel("Rating")
plot.ylabel("Number of Occurrences")
plot.title("Rating: predicted vs ground truth")
plot.show()
