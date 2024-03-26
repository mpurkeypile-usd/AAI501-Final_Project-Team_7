# AAI501 Final Project, Team 7
# Issa Ennab, Matt Purkeypile
# University of San Diego
# Spring 2024

import pandas as pd
import seaborn as sb
import numpy as nump
import statsmodels.formula.api as sm
import matplotlib.pyplot as plot

# pull the data locally
AllData = pd.read_csv(".\\drugsComTrain_raw.tsv", sep = "\t")
print(str(AllData.head()))
print(str(AllData.shape))