import numpy as np
import pandas as pd
import nltk
import re
import os
import sys
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sn
#convert document_id as list
def document_category(features_id):
    docs_category = []
    for ids in features_id:
        docs_category.append(ids)
    return docs_category

#convert the features as list for applyting to tfidf Vectorization
def separate_features_the_series(feature_series):
    main_feature_series = []
    for features in feature_series:
        data_features = features.replace(","," ")
        main_feature_series.append(data_features)
    return main_feature_series


def decide_number_of_clusters(tfidf_matrix):
    cluster_range = range(115,125)
    cluster_error = []
    for num_clusters in cluster_range:
        clusters = KMeans(num_clusters)
        clusters.fit(tfidf_matrix)
        cluster_error.append(clusters.inertia_)
    return cluster_range,cluster_error

def  analayse_the_similarity_between_clusters(tfidf_matrix):
    cluster_range = range(120,125)

    for n_clusters in cluster_range:
        #Initiliaze the clusterer with n_clusters
        cluster = KMeans(n_clusters=n_clusters,random_state=10)
        cluster_labels = cluster.fit_predict(tfidf_matrix)
        silhouette_avg = silhouette_score(tfidf_matrix,cluster_labels)
        print("For {} clusters:\n The Silhouette Average Score is {}".format(n_clusters,silhouette_avg))
        sample_silhouette_values  = silhouette_samples(tfidf_matrix,cluster_labels)
    

# initially we need document folders and document id's
path_to_dataset = '/home/madhi/Documents/python programs/neuralnetworks/fp/Reuters21578-Apte-115Cat/training'
folder_list = os.listdir(path_to_dataset)

#make a dictionary of document category and documents contain in the folder
cat_and_docs = [] #list
document_folder = [] #list
for docs_folder in folder_list:
    folder_path = path_to_dataset + "/" + docs_folder
    list_of_documents = os.listdir(folder_path)
    for real_docs in list_of_documents:
        document_folder.append(real_docs)
    #append folder_name and docs into the same list
    cat_and_docs.append((docs_folder,document_folder))
    #empty the document_folder list
    document_folder = []
    
#print(cat_and_docs[1]) #the list is in format category and number of docs in that list

#make into dictionary
cate = [] 
documents = []

for category,docs in cat_and_docs:
    cate.append(category)
    documents.append(docs)
    
#convert into dictionary
total_dataset = dict(zip(cate,documents)) #length of the dictionary is 116

feature_df = pd.read_csv('../mainfolder/features.txt',sep='\t',header=None,names=['doc_id','features'])
doc_category = feature_df['doc_id']
features = feature_df['features']
#convert into pandas Series
features_id = pd.Series(doc_category)
features_series = pd.Series(features) #both features_id and features_series has shape((11475,)(11475,))

docs_id = document_category(features_id)
features_splitted = separate_features_the_series(features_series) #both docs_id and features_splitted have length(11475,11475)

features_data = []
#preprocessing
for sent_of_features in features_splitted:
    feat_sent = sent_of_features.replace("\n"," ")
    feat_sent = sent_of_features.replace(" ",",")
    features_data.append(feat_sent) #return list

#store the features_data as txt file for future references

total_all_features = open('features_splitted.txt').read()
#replace total_all_features newline by space
list_totat_all_features = total_all_features.replace("\n"," ")
#convert into list
list_totat_all_features = list_totat_all_features.split(",") #len is 6961


#Tfidf Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=1.0,max_features=15000,use_idf=True,ngram_range=(1,4))
tfidf_matrix = tfidf_vectorizer.fit_transform(list_totat_all_features) #shape of the matrix is (1406,1406)

#cosine similarity
dist = 1- cosine_similarity(tfidf_matrix) #the matrix is in shape(1406,1406)

#decide number of clusters for  efficiency
num,cluster_error = decide_number_of_clusters(tfidf_matrix)
cluster_df = pd.DataFrame({"Number":num,"ErrorRate":cluster_error})

#plt.plot(cluster_df.Number,cluster_df.ErrorRate,marker='o') --> save this graph

#analyse the similarity between the clusters
similarity_analysing = analayse_the_similarity_between_clusters(tfidf_matrix)
