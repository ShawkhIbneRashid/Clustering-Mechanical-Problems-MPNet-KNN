from scipy.cluster import  hierarchy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#this class draws dendogram and kmeans clusters in a scatterplot for a particular type of uid(uid strating with either WO, DE, EP, US, JP or CN)
class Clustering_Visualization():

  def __init__(self, uid_sentence_embed_pair):

    self.uid = uid_sentence_embed_pair[0]
    self.embed_matrix = uid_sentence_embed_pair[1]
    #setting the number of classes for each type of UID seperately for more use control
    if uid_sentence_embed_pair[0] == 'WO':
      self.clusters = 2
    elif uid_sentence_embed_pair[0] == 'DE':
      self.clusters = 4
    elif uid_sentence_embed_pair[0] == 'EP':
      self.clusters = 6
    elif uid_sentence_embed_pair[0] == 'US':
      self.clusters = 5
    elif uid_sentence_embed_pair[0] == 'JP':
      self.clusters = 2
    elif uid_sentence_embed_pair[0] == 'CN':
      self.clusters = 6


  def dendrogram_draw(self):

    # Performs hierarchical/agglomerative clustering based on Ward variance minimization algorithm
    hier_link = hierarchy.linkage(self.embed_matrix, method="average", metric = "cosine")
    print("Hierarchical clustering of records whose uid start with " + self.uid)
    plt.figure(figsize=(20, 20))
    #labelList = range(1, 256)
    dn = hierarchy.dendrogram(hier_link, orientation='right',
                distance_sort='descending')

    plt.title('Dendogram')
    plt.ylabel('Cosine Similarity')
    plt.show()

  def k_means_labels(self, uid_problem_pair):
    #shows the cluster numbers of each record for a particular type( WO, DE, EP, US, JP or CN)
    n_clusters = self.clusters
    if self.uid == 'CN':
      max_iter = 4000
    elif self.uid == 'WO':
      max_iter = 500
    elif self.uid == 'DE':
      max_iter = 2000
    elif self.uid == 'EP':
      max_iter = 4000
    elif self.uid == 'US':
      max_iter = 3000
    elif self.uid == 'JP':
      max_iter = 500
    clf = KMeans(n_clusters=n_clusters, max_iter=max_iter,init = 'k-means++', n_init=1)
    labels = clf.fit_predict(self.embed_matrix)
    print("Printing the cluster labels for all the records whose uid start with " + self.uid + " total record number under this category is " + str(len(list(uid_problem_pair.keys()))) + "\n")
    print(labels)
    for index, sentence in enumerate(list(uid_problem_pair.values())):
      print(str(labels[index]) + ":  UID:" + str(list(uid_problem_pair.keys())[index]) + "     Problem:" + str(sentence) )
    return labels, clf

  def k_means_clustering(self, labels, clf):
    #clusters are plotted using Principal Component Analysis
    print("Scatter plot for records whose uid start with " + self.uid + "\n")
    pca = PCA(n_components=self.clusters).fit(self.embed_matrix)
    coords = pca.transform(self.embed_matrix)
    label_colors = ["#2AB0E9", "#2BAF74", "#D7665E", "#CCCCCC", "#D2CA0D", "#522A64", "#A3DB05", "#FC6514", "#DAF7A6" , "#581845"]
    colors = [label_colors[i] for i in labels]
    print("Records with their respective cluster labels are plotted using a Scatter plot." +"\n"+ "For dimentionality reduction of matrix containing sentence embeddings, we have used Principal Component Analysis." + "\n" +" X denotes cluster centers in the scatter plot." )
    plt.scatter(coords[:, 0], coords[:, 1], c = colors)
    centroids = clf.cluster_centers_
    centroid_coords = pca.transform(centroids)
    plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], marker='X', s=200, linewidths=2, c='#444d61')
    plt.show()
