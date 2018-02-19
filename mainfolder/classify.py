#import necessary libraries
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from collections import namedtuple
from sklearn.pipeline import Pipeline 
from sklearn .ensemble import RandomForestClassifier
#for plotting
import matplotlib.pyplot  as plt
#use ggplot as style
plt.style.use('ggplot')

#parameters
n_classes = 116
n_estimators = 30
RANDOM_SEED = 15

def changeintoDataFrame(cont,columns):
	df = pd.DataFrame(cont,columns=columns)
	shape = df.shape
	return df,shape

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

#main function starts from here
Y,X = tagAndTrainingData('trainingdataset.txt')
#splitting the docs
#the dataset is in shape (11476,2)
#75:25 testing dataset
count_vect = CountVectorizer()
x_train_count = count_vect.fit_transform(X)
print(x_train_count.shape)

tf_idf = TfidfTransformer()
X_train_tfidf = tf_idf.fit_transform(x_train_count)

#SGD Classifier
doc_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),
					('clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,n_iter=40,random_state=50))])

doc_clf = doc_clf.fit(X,Y)
predicted = doc_clf.predict(X)
score = np.mean(predicted == Y)

print("Accuracy Using StochasticGradientClassifier:{}".format(score * 100)) #77% accuracy

#predict the document
category = doc_clf.predict([r'Cotton plays a major role in pharmacy. It helps to clot the blood '])
print(category)


#multinomial NBClassifier
mnb_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),
					('clf',MultinomialNB(alpha=1.0,fit_prior=True,class_prior=None))])
mnb_predict = mnb_clf.fit(X,Y).predict(X)
mnb_score = np.mean(mnb_predict == Y)

print("Accuracy using Multinomial NaiveBayes Classifier is:{}".format(mnb_score*100)) # 58% accuracy

#support vector machine
svc_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),
					('clf',SVC(C=1.0,kernel='linear'))])
svc_predict = svc_clf.fit(X,Y).predict(X)
svc_score = np.mean(svc_predict == Y)

print("Accuracy using Support Vector Machine is:{}".format(svc_score*100)) # 82 % accuracy

# Random Forest Classifier
rndm_clf = Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),
					('clf',RandomForestClassifier(n_estimators=1000,criterion='gini',oob_score=True))])

rndm_predict = rndm_clf.fit(X,Y).predict(X)
rndm_score = np.mean(rndm_predict==Y)

print("Accuracy using Random Forest Classifier is:{}".format(rndm_score * 100))



