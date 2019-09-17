# steps:
# 1) get training info from user
# 2) parse out into usable from
# 3) create features (POS, position in tweet, langauge, some combo)
# 4) rank according to log(sense1Prob/sense2Prob) (using Laplace smoothing where needed)
# 5) create and use descion list


# sample input:
# Sample command line: py 'C:\Users\Alwin Hollebrandse\Dropbox\NLP\assign4\decision-list.py' 'C:\Users\Alwin Hollebrandse\Dropbox\NLP\assign4\line-train.txt' 'C:\Users\Alwin Hollebrandse\Dropbox\NLP\assign4\line-test.txt' > 'C:\Users\Alwin Hollebrandse\Dropbox\NLP\assign4\my-line-answers.txt'
# the first argument is the txt file that contains the ambigious word training data. The second argument is the txt file that contains the test data. After these arguments, there should be "> [the txt file with the predications]" so that the results can be scored

# py .\SentiMix.py .\spanglish_trial_release.txt
# py .\SentiMix.py ..\..\spanglish_trial_release.json
# py .\SentiMix.py ..\..\tweets_train.json


# pip install pytorch-pretrained-bert

from __future__ import print_function
# import torch
import numpy as np
import argparse
import json
from torchtext import data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# BERT stuff
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import matplotlib.pyplot as plt


def pytorchstuff(spanglishData):
   TEXT = data.Field(tokenize = None) # TODO chcek if tokenize None works
   LABEL = data.LabelField(dtype = torch.float)
   # train_data, test_data = spanglishData.(TEXT, LABEL) #TODO use sklearn to split?

   # print(f'Number of training examples: {len(train_data)}')
   # print(f'Number of testing examples: {len(test_data)}')


def parseData(dataFileName): #TODO will we be given a test set to as a final proect? like should both train and test be possible params?
   # used for training
   X = []
   y = []

   spanglishData = []

   with open(dataFileName, "r", encoding="utf8") as json_file:
      
      data = json.load(json_file)
   
      posSenti = []
      negSenti = []
      neutSenti = []

      

      for jsonLine in data:
         # jsonLine = json.loads(line) # note really json, its a python dictionary
         spanglishData.append(jsonLine)
         X.append(jsonLine["tokens"]) #TODO this is a list. should it be joined?
         # y.append(jsonLine["sentiment"])


         if (jsonLine["sentiment"] == "positive"):
            y.append(2)
            posSenti.append(jsonLine)
         elif (jsonLine["sentiment"] == "negative"):
            negSenti.append(jsonLine)
            y.append(1)
         elif (jsonLine["sentiment"] == "neutral"):
            neutSenti.append(jsonLine)
            y.append(0)

      # print(spanglishData)

      #NOTE that the sentiment-organized-lists are not returned/used for anything. idk if they will be
      print("number of all sentiments: ", len(spanglishData))
      print("number of positive sentiments: ", len(posSenti))
      print("number of negative sentiments: ", len(negSenti))
      print("number of neutral sentiments: ", len(neutSenti))

      # print("X : ", X[0])
      # print("y : ", y[0])

   return spanglishData, X, y, len(posSenti), len(negSenti), len(neutSenti)


# TODO pass in the cleaned data I suppose. maybe probaly clean based on given tokens and then combine the tokens here?
# NOTE This is one my favorite method names of all time
def prepareForBert(data):
   # Load pre-trained model tokenizer (vocabulary)
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 

   # Load pre-trained model (weights)
   model = BertModel.from_pretrained('bert-base-uncased')

   # Put the model in "evaluation" mode, meaning feed-forward operation.
   model.eval() # TODO does this go here? or should it be in the loop? 

   firstRun = True

   bertPreparedTweets = []
   bertSentenceReprenstations = []
   for line in data:
      text = " ".join(line["tokens"])
      marked_text = "[CLS] " + text + " [SEP]" # TODO This is assuming a whole tweet is a single sentence rn. fix dis shiz homz

      tokenized_text = tokenizer.tokenize(marked_text)

      indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

      segments_ids = [1] * len(tokenized_text)

      # Convert inputs to PyTorch tensors
      tokens_tensor = torch.tensor([indexed_tokens])
      segments_tensors = torch.tensor([segments_ids])

      # Predict hidden states features for each layer
      with torch.no_grad(): # NOTE alledgly "(we don't need gradients or backpropagation since we're just running a forward pass)." but why? 
         encoded_layers, _ = model(tokens_tensor, segments_tensors) # NOTE I think this is the word embedding for the sentence (rn the whole tweet) TODO chcek TODO make per sentence, not per tweet

         # Convert the hidden state embeddings into single token vectors

         # Holds the list of 12 layer embeddings for each token
         # Will have the shape: [# tokens, # layers, # features]
         token_embeddings = [] 

         # For each token in the sentence...
         for token_i in range(len(tokenized_text)):
         
            # Holds 12 layers of hidden states for each token 
            hidden_layers = [] 
            
            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers)):

                # For each batch in the sentence...
               for batch_i in range(len(encoded_layers[layer_i])): # NOTE this was not in the tutorial, but without this "batch_i" is undefined, so I guess it's needed TODO check if this is in the correct spot

                  # Lookup the vector for `token_i` in `layer_i`
                  vec = encoded_layers[layer_i][batch_i][token_i]
                  
                  hidden_layers.append(vec)
               
            token_embeddings.append(hidden_layers)

            # The upshot being that, again, the correct pooling strategy (mean, max, concatenation, etc.) and layers used (last four, all, last layer, etc.) is dependent on the application. This discussion of pooling strategies applies both to entire sentence embeddings and individual ELMO-like token embeddings.
            # it almost looks like I can either use word vectors or sentences vectors. In theroy, we have to test both versions
            # at a guess, sentence reps will be faster to do, but words would be more accurate. but im not sure about that 

         sentence_embedding = torch.mean(encoded_layers[11], 1)

         bertSentenceReprenstations.append(sentence_embedding[0])
         
         if (firstRun):
            print ("Our final sentence embedding vector of shape: ", sentence_embedding[0].shape[0])

      # only print the first iteration of the loop as a sanity check
      if (firstRun):
         print("")
         print (tokenized_text)
         for tup in zip(tokenized_text, indexed_tokens):
            print (tup)
         print (segments_ids)

         print ("Number of layers:", len(encoded_layers))
         layer_i = 0

         print ("Number of batches:", len(encoded_layers[layer_i]))
         batch_i = 0

         print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
         token_i = 0

         print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

         # For the 5th token in our sentence, select its feature values from layer 5.
         token_i = 5
         layer_i = 5
         vec = encoded_layers[layer_i][batch_i][token_i]

         # # Plot the values as a histogram to show their distribution.
         # plt.figure(figsize=(10,10))
         # plt.hist(vec, bins=200)
         # plt.show()

         # Sanity check the dimensions:
         print ("Number of tokens in sequence:", len(token_embeddings))
         print ("Number of layers per token:", len(token_embeddings[0]))
         print("")
      firstRun = False
      

      bertPreparedTweets.append(tokenized_text)

   return bertPreparedTweets

# class BertForSequenceClassification(nn.Module):
  
#     def __init__(self, num_labels=2):
#         super(BertForSequenceClassification, self).__init__()
#         self.num_labels = num_labels
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         nn.init.xavier_normal_(self.classifier.weight)
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         return logits

def getUniqueTokens(data):
   uniqueTokens = {}

   for line in data: # needs to be strings
      for token in line['tokens']:
         if token not in uniqueTokens: 
            uniqueTokens[token] = 0
         uniqueTokens[token] += 1

   print("number of unique tokens: ", len(uniqueTokens))

   return uniqueTokens


def cleanData(spanglishData): #TODO pass in tokens
   #TODO go all lower case, do stemming and leminzation, fix typos, removing stop words, replace all urls and hashtags with "URL" and "hashtag" (or remove them), etc
   return


def splitData(X, y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)  #TODO what was the split %? #NOTE random_State is a seed. used for debugging
   
   # print("X train: ", X_train[0])
   # print("X test: ", X_test[0])
   # print("y train: ", y_train[0])
   # print("y test: ", y_test[0])

   print("number of X train: ", len(X_train))
   print("number of X test: ", len(X_test))
   print("number of y train: ", len(y_train))
   print("number of y test: ", len(y_test))

   return X_train, X_test, y_train, y_test


# TODO should this be based off of the test set? what happens when X-fold happens? I think im gonna use the whole dataset for this.
def getBaselinePredicitions(numOfPosSenti, numOfNegSenti, numOfNeutSenti, y_true):
   
   # create y_pred to be init with the most frequent sentiment seen in the data
   sentiToPredict = -1
   mostFrequentSenti = max(numOfPosSenti, numOfNegSenti, numOfNeutSenti)
   if (mostFrequentSenti == numOfPosSenti):
      sentiToPredict = 2
   elif (mostFrequentSenti == numOfPosSenti):
      sentiToPredict = 1
   elif (mostFrequentSenti == numOfPosSenti):
      sentiToPredict = 0
   y_pred = [sentiToPredict] * len(y_true)

   scorer(y_true, y_pred)


def scorer(y_true, y_pred):
   accuracy = accuracy_score(y_true, y_pred)
   error = 1.0 - accuracy
   cm = confusion_matrix(y_true, y_pred)
   averagePrecisonRecallFScore = precision_recall_fscore_support(y_true, y_pred, average='macro') #TODO what average to use?
   perLabel = precision_recall_fscore_support(y_true, y_pred, average=None,
      labels=[0,1,2])

   print(cm)
   print("accuracy: ", accuracy)
   print("error: ", error)
   print("averagePrecisonRecallFScore: ", averagePrecisonRecallFScore)
   print("perLabel preciosn, recall, and fscore: ", perLabel)


# main
def main():
   # x = torch.rand(5, 3)
   # print(x)
   # y = torch.cuda.is_available()
   # print(y)

   # gets the command line args
   parser = argparse.ArgumentParser(description='Process some SentiMix data files')
   parser.add_argument("dataFileName", type=str, help="the name/path of the file that has the SentiMix training data")
   # parser.add_argument("testFileName", type=str, help="the name/path of the file that has the SentiMix test data")
   args = parser.parse_args()
   dataFileName = args.dataFileName
   # testFileName = args.testFileName

   spanglishData, X, y, numOfPosSenti, numOfNegSenti, numOfNeutSenti = parseData(dataFileName)

   # getBaselinePredicitions(numOfPosSenti, numOfNegSenti, numOfNeutSenti, y) # see results at the bottom of this. TODO add to the top in the summary.

   getUniqueTokens(spanglishData)

   # bertPreparedTweets = prepareForBert(spanglishData)

   # tfidf(spanglishData[:2])
   X_train, X_test, y_train, y_test = splitData(X, y)
   # naiveBayes(X_train, X_test, y_train, y_test)
   # pytorchstuff(spanglishData)

   # instanceList = []
   # n = 1
   # minRequiredApperances = 2

   # # default = parseTrain(instanceList, trainFileName)

   # nGrams = createNgrams(instanceList, n) # TODO why couldnt I use a global dict? the results werent returning? wtf?

   # sortedDecisionList = decisionList(nGrams, minRequiredApperances)
   # # print(sortedDecisionList) # print for log file

   # parseTestAndPredict(instanceList, default, n, sortedDecisionList, testFileName)

   # https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa
   # Both have their own advantages and disadvantages. According to Mikolov, Skip Gram works well with small amount of data and is found to represent rare words well.
   # On the other hand, CBOW is faster and has better representations for more frequent words.

   #BERT tutorial https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
   #TODO for BERT, each token list must start with '[CLS]' and '[SEP]' at the end. TODO how does that work with the 1-2 sentence option?
   #uses 30k most common english "words" TODO can that be combined with spanish?
   # In order to get the individual vectors we will need to combine some of the layer vectors…but which layer or combination of layers provides the best representation? The BERT authors tested this by feeding different vector combinations as input features to a BiLSTM used on a named entity recognition task and observing the resulting F1 scores.
   # For out of vocabulary words that are composed of multiple sentence and character-level embeddings, there is a further issue of how best to recover this embedding. Averaging the embeddings is the most straightforward solution (one that is relied upon in similar embedding models with subword vocabularies like fasttext), but summation of subword embeddings and simply taking the last token embedding (remember that the vectors are context sensitive) are acceptable alternative strategies.

if __name__ == '__main__':
   main()


# baseline results: 
# where class 2 is pos, class 1 is neutral, and class 0 is negative
# [[   0    0 4968]
#  [   0    0 2529]
#  [   0    0 7503]]
# accuracy:  0.5002
# error:  0.4998

# from a failed sklearn attempt. worst case ill go back to this if bert is also a failure
# def tfidf(text):
#    # # list of text documents
#    # text = ["The quick brown fox jumped over the lazy dog.",
#    #       "The dog.",
#    #       "The fox"]

#    joinedText = []

#    for line in text: # needs to be strings
#       s = " ".join(line["tokens"])
#       print("s: ", s)
#       joinedText.append(s)

#    # version A
#    # create the transform
#    vectorizer = TfidfVectorizer()
#    # tokenize and build vocab
#    vectorizer.fit(joinedText)
#    # summarize
#    print(vectorizer.vocabulary_)
#    print(vectorizer.idf_)
#    # encode document
#    vector = vectorizer.transform([joinedText[0]])
#    # summarize encoded vector
#    print(vector.shape)
#    print(vector.toarray())

#    #version B
#    test = TfidfVectorizer(min_df=1, max_df=5, ngram_range=(1,2)) # TODO play with params
#    features = test.fit_transform(joinedText)
#    print("features: ", features)

#    #TODO which version to use?


# def naiveBayes(X_train, X_test, y_train, y_test):
#    nb = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', MultinomialNB()),
#               ])
#    nb.fit(X_train, y_train)

#    # %%time
#    y_pred = nb.predict(X_test)

#    print('accuracy %s' % accuracy_score(y_pred, y_test))
#    print(classification_report(y_test, y_pred,target_names=my_tags))
