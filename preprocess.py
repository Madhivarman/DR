import pandas as pd
import numpy as np
import os
import sys

#global declaration
path_file = "/<path_to_training_dataset>"
topic_folders = os.listdir(path_file)

test_path_file = "<path_to_testing_dataset>"
test_topic_folders = os.listdir(test_path_file)

#file to change into dataframe

"""def createDataFrame(file):
	read_file = pd.read_csv(file,delimiter="\n",header=None)
	dataframe = read_file.to_csv(file+".csv")
	return dataframe"""

def assignIdToDocuments(filename,topic_details_list,folder_path):

	doc_id = 0 #initial declaration
	with open(filename,"w") as fp:
		#opening the project folders
		for folders in folder_path:
			no_of_docs = os.listdir(path_file+"/"+folders)
			#iterate through the number of documents
			for org_docs in no_of_docs:
				with open(path_file + "/" + folders +"/" + org_docs,"r") as f:
					data = f.read()
					fp.write(str(doc_id)+","+data)
					fp.write("\n")

			doc_id += 1



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

	topic_names = sorted(topic_names)
	topic_details = dict(topic_names)

	print(topic_names)

	#assign documents with respective id's
	assign_id_to_docs_train = assignIdToDocuments("trainingdataset.txt",topic_names,topic_folders)
	assign_id_to_docs_test = assignIdToDocuments("testdataset.txt",topic_names,test_topic_folders)
