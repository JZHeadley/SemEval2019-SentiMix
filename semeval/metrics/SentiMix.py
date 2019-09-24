# sample input:
# Sample command line: py 'C:\Users\Alwin Hollebrandse\Dropbox\NLP\assign4\decision-list.py' 'C:\Users\Alwin Hollebrandse\Dropbox\NLP\assign4\line-train.txt' 'C:\Users\Alwin Hollebrandse\Dropbox\NLP\assign4\line-test.txt' > 'C:\Users\Alwin Hollebrandse\Dropbox\NLP\assign4\my-line-answers.txt'
# the first argument is the txt file that contains the ambigious word training data. The second argument is the txt file that contains the test data. After these arguments, there should be "> [the txt file with the predications]" so that the results can be scored

# py .\SentiMix.py .\spanglish_trial_release.txt
# py .\SentiMix.py ..\..\spanglish_trial_release.json
# py .\SentiMix.py ..\..\tweets_train.json
# py .\SentiMix.py ..\output_tweets.json


from __future__ import print_function
# import torch
import numpy as np
import argparse


#TODO I think everything needed for the classifer is in here: https://github.com/google-research/bert/blob/master/run_classifier.py
#TODO maybe follow this : https://ireneli.eu/2019/07/05/deep-learning-17-text-classification-with-bert-using-pytorch/ and just split all the data after cleaning and then run these?


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

   # spanglishData, X, y, numOfPosSenti, numOfNegSenti, numOfNeutSenti = parseData(dataFileName)

   # getBaselinePredicitions(numOfPosSenti, numOfNegSenti, numOfNeutSenti, y) # see results at the bottom of this. TODO add to the top in the summary.

   # getUniqueTokens(spanglishData)

   # createTSVFiles(spanglishData)

   # bertPreparedTweets = prepareForBert(spanglishData)

   # tfidf(spanglishData[:2])
   # X_train, X_test, y_train, y_test = splitData(X, y)

   # createSplitTSVFiles(X_train, X_test, y_train, y_test)

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
