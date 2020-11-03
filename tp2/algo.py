import pandas as pd
import sklearn
import joblib
from sklearn.pipeline import pipeline
from sklearn.svm import SVC
from sklearn.featur_extraction.text import TfidVectorizer
from stop_words import get_stop_words
from sklearn.multiclass import OneVsRestClassifier

#Import dataset
labelDf= pd.read_csv("labels.csv",sep=',',encoding='iso-8859-1')
labelDf= labelDf.drop("Unnamed:0",1)