import numpy as np 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import re
from string import digits

def tagAndTrainingData(file):
	#returning tokenize sentence
	tags = []
	documents = []
	counter = 1
	with open(file) as f:
		for line in f:
			#skip the first line
			if counter == 1:
				counter += 1
				continue

			tags.append(line[:3]) #separating document_id
			documents.append(line[3:]) #separating body of the document
	return tags,documents


Y,X = tagAndTrainingData('trainingdataset.txt')
df_X = pd.Series(X,index=None)
data = []

for sent in df_X:
	data.append(sent.translate(None,digits))

save = pd.DataFrame(data,header=None,index=None).to_csv("cleaned_document.txt",sep="\t")

