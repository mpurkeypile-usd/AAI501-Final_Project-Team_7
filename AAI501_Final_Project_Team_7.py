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

#AllDataTest = pd.read_csv(".\\drugsComTest_raw.tsv", sep = "\t")

### Start doing NLP (can also refactor and encapsulate this later) ###

# Some of this code based on the following:
# Sarkar, D. (2019). Text Analytics with Python: A Practitioner's Guide to Natural Language Processing. Apress.
from normalization import normalize_corpus

nltk.download('stopwords')
StopWordList = nltk.corpus.stopwords.words("english")
StopWordList.remove("no")    # keep negation
StopWordList.remove("not")

NormCorpus = normalize_corpus(corpus = AllDataTrain["review"])
AllDataTrain["cleanReview"] = NormCorpus
print(str(AllDataTrain["review", "cleanReview"].head(20)))