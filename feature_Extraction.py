
# coding: utf-8

# In[106]:


import numpy as np 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import re


# In[107]:


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


# In[109]:


Y,X = tagAndTrainingData('preprocessing/trainingdataset.txt')
df_X = pd.Series(X,index=None)


# In[110]:


#vecotrizer
stopwords = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stopwords,use_idf=True)


# In[112]:


vect_X = vectorizer.fit_transform(df_X)


# In[113]:


vect_X[0]


# In[114]:


terms = vectorizer.get_feature_names()
print(len(terms))


# In[115]:


#list to store the TruncatedSVD value
decomposition = []
i=1
prev = 0
svd = TruncatedSVD(n_components=765,n_iter=10)
while(i*765 <= vect_X.shape[0]):
    index = i*765
    data = df_X.loc[prev:index]
    decomposition.append(data) # document in the list shape(766,) *15 times
    prev=index
    i+=1


# In[116]:


data_decomp = []
for data in decomposition:
    data_decomp.append(vectorizer.fit_transform(data))


# In[117]:


data_decomp #did broke down into smaller pieces


# In[119]:


svd_fit=[]
i=1
for data in data_decomp:
    svd_fit.append(svd.fit(data))
    print("Finished {} iteration".format(i))
    i+=1


# In[122]:


#sample
svd_fit[5]


# In[125]:


with open("features.txt","w") as fp:
    batch=1
    for trunc_svd  in svd_fit:
        fp.write("BATCH:{}\n".format(batch))
        for i,comp in enumerate(trunc_svd.components_):
            termsInGroup = zip(terms,comp)
            sortedItems = sorted(termsInGroup,key=lambda x:x[1],reverse=False)[:10] #maximum of 10 keywords per document
            fp.write("{},".format(i))
            for term in sortedItems:
                fp.write("{},\t".format(term[0]))
            fp.write("\n")
        fp.write("\n")
        batch +=1

