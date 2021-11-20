# Clustering-Mechanical-Problems-MPNet-KNN
Problem Statement:

The file Dataset_Clustering_Problems.csv includes three columns in which one column name is "problems". The problem states to cluster/categorize all the records present in the file based on  "problems" column text. Identify all the unique labels first and then cluster/categorize the records into these unique labels.  

My Approach:  

All the uids in the uid column are unique. But all the uids start with either WO, DE, EP, US, JP or CN. So I have considered these six unique labels or uids and clustered all the records which fall under these uids. For example, I first took all the extracted_problems records whose uid start with CN and then  transformed them into embeddings using the MPNet embedding technique. Users can also use BERT if they wish. After that, I used cosine similarity  metric and drew the dendogram with the extracted_problems sentences. I also, showed the K-Means clustering using a scatter plot and printed the cluster number for each of the sentences. I repeated the same procedures for the other uids that start with WO, DE, EP, US and JP.  

To run the program:  

There are multiple .py extension files and a .ipynb file. For the visualization purpose I put all the codes along with their outputs in the ipynb file. dataset.py, embed.py and cluster.py contains three classes. dataset.py file has a class called Load_Preprocess which is used for loading and preprocessing  the dataset. In the embed.py file there is a class called Sentence_Embedding, which creates the sentence embeddings of the extracted problems belonging  to a particular label/uid (WO, DE, EP, US, JP or CN). The cluster.py has a class, Clustering_Visualization, which draws dendogram and kmeans clusters  in a scatterplot for a particular type of uid.  You will have to run the cn_uid.py, wo_uid.py, de_uid.py, ep_uid.py, us_uid.py and jp_uid.py to see the dendogram and scatterplot of CN, WO, DE, EP, US and  JP uids respectively. The data_load.py file prints the first 20 rows of the provided dataset before and after applying the preprocessing techniques.   The figures folder contain the graphs for the six unique ids or labels. 
