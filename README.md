# DR
document retrieval

Document Retrieval : Retrieving document from the database by related to User Queries. 

Algorthims :

- **Indexing** - Hashing Technique 
- **Topic Modeling** - Latent Dirichlet Allocation (LDA)
- **Clustering** - K means clustering
- **Selecting Docs** - Euclidean Distance to select nearby document from the clusters
- **Snippet Generation** - Long Short Term Memory(LSTM) Recurrence Neural Network
- **Feature Extraction** - Scikit-Learning or Rake Extraction

Work Description :

- **index.py** - Create indexing for the document to retrieve and locate
- **preprocess.py** - Creating training and testing dataset
- **classify.py** - To classify the Uploading document
- **feature_Extraction.ipynb** -To extract important features from the document
- **clustering.ipynb** - To cluster the extracted_features into 5 clusters and visualize the documents located in the clustering
