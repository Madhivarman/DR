import pandas as pd
import numpy as np
import os
import sys
import re

#global declaration
path_file = "/home/madhi/Documents/python programs/neuralnetworks/fp/Reuters21578-Apte-115Cat/training"
topic_folders = os.listdir(path_file)

test_path_file = "/home/madhi/Documents/python programs/neuralnetworks/fp/Reuters21578-Apte-115Cat/test"
test_topic_folders = os.listdir(test_path_file)

#file to change into dataframe

def createDataFrame(file,filename):
	read_file = pd.read_csv(file,delimiter="\n",header=None)
	dataframe = read_file.to_csv(filename +".csv")
	return dataframe

#save as csv file
def saveAsCSVFile(training_set,filename):
	save = training_set.to_csv(filename+".csv")
	return save


trainingdataset = [] #store the docs in arrays
testingdataset = []

def data_preprocessing(data):
	replace = ['\n','.','?','<','>']
	for x in replace:
		data = data.replace(x," ")
	return data

def changeintodataframe(topic_details,topic_folders):
	#topic folders contains folders
	#topic_details contains <folder_id and folder names>
	doc_id = 0
	for folders in topic_folders:
		no_of_docs = os.listdir(path_file+"/"+folders)
		for org_docs in no_of_docs:
			with open(path_file+"/"+folders+"/"+org_docs,"r") as f:
				data = f.read()
				data = data_preprocessing(data) #remove all regex expressions and newline characters
				trainingdataset.append([doc_id,data])
		doc_id += 1

	return trainingdataset

def testchangeintodataframe(topic_details,topic_folders):
	doc_id = 0
	for folders in topic_folders:
		no_of_docs = os.listdir(test_path_file+"/"+folders)
		for org_docs in no_of_docs:
			with open(test_path_file+"/"+folders+"/"+org_docs,"r") as f:
				data = f.read()
				data = data_preprocessing(data) #cleaning the data
				testingdataset.append([doc_id,data])

		doc_id += 1

	return testingdataset

def datasetPreparation(array_dataset):
	df = pd.DataFrame(array_dataset,columns=['docs_class','body'])
	shape = df.shape
	return df,shape

#main function
if __name__ == '__main__':
	#dataset
	index_file  = 'index.txt'

	#index_file_df = createDataFrame(index_file)

	#creating a trainingdataset for document classification
	topic_names = [] #list to save topic folders
	topic_id = 0

	for topics in topic_folders:
		topic_names.append((topic_id,topics))
		topic_id += 1

	topic_details = dict(topic_names)

	#print(topic_names)

	#change docs into the dataframe
	training_set = changeintodataframe(topic_names,topic_folders)
	testing_set = testchangeintodataframe(topic_names,test_topic_folders)


	#assign documents with respective id's
	prepare_training_set,training_shape = datasetPreparation(training_set)
	prepare_testing_set,testing_shape = datasetPreparation(testing_set)

	#print(training_shape)

	#save as csv file
	trainingdataset = saveAsCSVFile(prepare_training_set,"trainingdataset")
