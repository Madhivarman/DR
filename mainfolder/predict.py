#load the trained model
from sklearn.externals import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import os,shutil
#load the trained model
trained_model = joblib.load('rf_trained_model.pkl')

def preprocess_the_document(data_to_preprocess):
	#remove the numbers and expressions from the statemen
	#separate the document by newline
	to_remove = "0123456789"
	table = str.maketrans("","",to_remove)
	data = data_to_preprocess.translate(table)
	data = data.replace("\n"," ")
	stopset = set(stopwords.words('english'))
	word_tokens = word_tokenize(data)
	filtered_data = [w for w in word_tokens if w not in stopset] 
	data = ' '.join(filtered_data)
	s = re.sub(r'[^\w\s]','',data)
	#replace double whitespace by single
	s = s.replace('  ',' ')
	return s

#load the trained model
trained_model = joblib.load('rf_trained_model.pkl')

#read the document
document_to_predict = open('document_predict.txt')
file_Data = document_to_predict.read()
document_data = preprocess_the_document(file_Data)
predicted_category = trained_model.predict([r'{}'.format(document_data)])
print("Document is predicted under {} category".format(predicted_category))

#add the document to the document database  in the particular category
#add the topics_list.txt file to know the category
dictionary = {}
with open('../preprocessing/topics.txt') as fp:
	for line in fp:
		(key,topic) = line.split()
		dictionary[int(key)] = topic

#category
category = None
for key,topics in dictionary.items():
	if key == predicted_category.astype(int):
		category = topics

print("The document belongs to {} category".format('\033[92m'+category+'\033[0m'))