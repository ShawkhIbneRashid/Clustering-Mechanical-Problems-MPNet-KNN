from embed import Sentence_Embedding
from cluster import Clustering_Visualization
from dataset import Load_Preprocess

directory = '../Dataset_Clusterin_Problems.csv'
load_preprocess = Load_Preprocess(directory)
df = load_preprocess.load_data()
df = load_preprocess.preprocess(df)
df_unique_labels = load_preprocess.unique_uid(df)
#There are total 6 unique uids based on the first two letters of the uids.
#These uids start with either 'EP', 'CN', 'WO', 'JP', 'DE' or 'US'.
#So, I have considered 6 unique uids.
#The next four blocks corresponds to showing dendogram, extracted_problems and their cluster numbers and a scatter plot of K-means clustering for uids staring with CN.


#specifying mpnet as embedding technique. Another option the user can select is bert.
#caution: running this will download some trained weights for performing the sentence embedding operations
embed = Sentence_Embedding(df, 'CN', 'mpnet')
uid_problem_pair = embed.uid_problem_pair()
uid_sentence_embed_pair = embed.sentence_embed(uid_problem_pair)

#shows the dendogram
clustering = Clustering_Visualization(uid_sentence_embed_pair)
clustering.dendrogram_draw()

#shows the cluster number of the extracted_problems of the uids starting with 'CN'
labels_out = clustering.k_means_labels(uid_problem_pair)

#Shows the scatter plot of K-means clustering
clustering.k_means_clustering(labels = labels_out[0], clf = labels_out[1])
