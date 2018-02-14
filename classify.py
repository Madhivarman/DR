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
					('clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,n_iter=10,random_state=50))])

doc_clf = doc_clf.fit(X,Y)
predicted = doc_clf.predict(X)
score = np.mean(predicted == Y)

print("Accuracy Using StochasticGradientClassifier:{}".format(score * 100)) #76% accuracy

#predict the document
category = doc_clf.predict(["US COULD COMPLAIN TO GATT ON CANADA CORN DUTY WASHINGTON, March 16  US Trade Representative Clayton Yeutter suggested the US could file a formal complaint with the General Agreement on Tariffs and Trade (GATT) challenging Canadas decision to impose duties on US corn imports Asked about the Canadian government decision to apply a duty of 849 cents per bushel on US corn shipments, Yeutter said the US could file a formal complaint GATT under the dispute settlement procedures of the subsidies code Other US options would be to appeal the decision in Canadian courts, or to retaliate against Canadian goods, a lowerlevel US trade official said However, retaliation is an unlikely step, at least initially, that official said No decision on US action is expected at least until after documents on the ruling are received here later this week"])
print("Document Belongs to:{}".format(category))
