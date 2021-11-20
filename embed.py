from sentence_transformers import SentenceTransformer
#this class creates the sentence embeddings of the extracted_problems belonging to a particular label/uid
class Sentence_Embedding():

  def __init__(self, df, uid, model):
    #uid will be either WO, DE, EP, US, JP or CN at a time
    self.df = df
    self.uid = uid
    self.model = model

  def uid_problem_pair(self):
    #creating a dictionary of {uid (uid staring with either WO, DE, EP, US, JP or CN), extracted_problem}
    dict_problems = {}
    for i in range(len(self.df['uid'])):
      if self.df['uid'][i][0:2] == self.uid:
        dict_problems[self.df['uid'][i]] = self.df['extracted_problems'][i]
    print('Number of records under uid starting with ' + self.uid + ' is ' + str(len(list(dict_problems.keys()))))
    return dict_problems

  def sentence_embed(self, dict_problems):
    #sentence embedding using either BERT or MPNet
    #a sentence is first converted to 512 tokens
    #each token generates a vector of 768 values
    #finally a pooling layer will convert a 512x768 tensor to (1x768) vector for each sentence

    if self.model == 'bert':
      model = SentenceTransformer('bert-base-nli-mean-tokens')
      sentence_embeddings = model.encode(list(dict_problems.values()))

    elif self.model == 'mpnet':
      model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
      sentence_embeddings = model.encode(list(dict_problems.values()))

    print("Shape of " + str(len(list(dict_problems.keys()))) + " embedded sentences is " + str(sentence_embeddings.shape))
    return self.uid, sentence_embeddings
