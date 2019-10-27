import json
import sys
import time
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser, get_shutdown_parser
from pre_processing import cleaning
from metrics import metrics
from processing import processing
from machine_learning import machine_learning
from progress.spinner import Spinner
import csv 
import numpy as np
import argparse

# python3 main.py bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4
# python3 ./semeval/main.py bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=1
# Zephyr Just a helper method for code cleanliness
def run_spinner(label,duration):
    spinner = Spinner(label)
    for i in range(0,duration*10):
        time.sleep(.1)
        spinner.next()
    spinner.finish()

# Zephyr needed for unpacking  and packing the json
def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

# Zephyr writes embeddings as a csv for input into better processing libraries (or custom written cuda knn...)
def write_csv_embeddings(embeddings):
    csv_embeddings = []
    for i,embedding in enumerate(embeddings):
        csv_embeddings.append(embedding['embedding'])
        csv_embeddings[i]=np.append(csv_embeddings[i],0 if embedding['sentiment'] == 'negative' else 1 if embedding['sentiment'] =='neutral' else 2)
    # print(csv_embeddings[:2])
    with open('data/embeddings.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_embeddings)
    return csv_embeddings
# Zephyr - this function is from the bert-as-a-service library.
# I made a small change to the first line to force it to ignore args it doesn't know
# This allowed me to make it accept command line args of my own and nicely work with 2 different arg parsers

def get_run_args(parser_fn=get_args_parser, printed=True):
    args,_ = parser_fn().parse_known_args()
    if printed:
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    return args

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", help="Clean the data", action="store_true")
    parser.add_argument("--embeddings", help="Get the embeddings for the data", action="store_true")
    parser.add_argument("--csv", help="write embeddings to a csv file", action="store_true")
    parser.add_argument("--ml", help="Runs the machine learning models on the data", action="store_true")

    args, _ = parser.parse_known_args()
    return args
    

if __name__ =='__main__':
    # parse_conll_to_json('train_conll_spanglish.txt','tweets_train.json')
    args = parse_arguments()
    print('args: ', args)
    if args.embeddings: 
        bert_args = get_run_args()
        print(bert_args)
        server = BertServer(bert_args)
        server.start()
        run_spinner('Loading Bert as a service\t',20)
    if args.clean:
        with open('data/tweets_train.json') as json_file:
            data = json.load(json_file)
            # data = data[:2000]
        data = cleaning.clean_tweets(data)
        data = cleaning.lowercase(data)
        data = cleaning.remove_stop_words(data)
        data = cleaning.lemmatize(data)
        with open('data/output_tweets.json', 'w') as fp:
            json.dump(data, fp)
    else:
        with open('data/output_tweets.json', 'r') as fp:
            data = json.load(fp)

    # data = data[:3]
    # capTweets, nonCappedTweets = metrics.splitTweetsByCaps(data)
    # data = capTweets
    # numOfPosSenti, numOfNeutSenti, numOfNegSenti = metrics.getSentimentCounts(data)

    # # NOTE in theory, breaks the embedding stuff. Until they are combined, leave it commented out. except for the NOTE line (potentially)
    # # # preprocessing
    # emojiTweets, nonEmojiTweets = metrics.splitTweetsByEmoji(data) # NOTE use nonEmojiTweets for other methods, combinig the emoji appraoch? TODO
    # data = emojiTweets
    # numOfPosSenti, numOfNeutSenti, numOfNegSenti = metrics.getSentimentCounts(data)
    # print("number of positive sentiments: ", numOfPosSenti)
    # print("number of neutral sentiments: ", numOfNeutSenti)
    # print("number of negative sentiments: ", numOfNegSenti)
    # mostFrequentSentiment = metrics.getMostFreqSentiment(numOfPosSenti, numOfNegSenti, numOfNeutSenti)

    # # # splitting data
    # x = [x['tweet'] for x in data]
    # y = [ 0 if data['sentiment'] == 'negative' else 1 if data['sentiment'] =='neutral' else 2 for data in data]
    # x_train,x_test,y_train,y_test = processing.splitData(x,y)

    # # # emoji map
    # emoji_sentiments = metrics.calculate_emoji_sentiments(x_train, y_train)
    # # print('emoji_sentiments: ', emoji_sentiments)
    # emojiPredictions = metrics.get_emoji_baseline(x_test, mostFrequentSentiment, emoji_sentiments)
    # metrics.scorer(y_test, emojiPredictions)

    if args.embeddings:
        _, embeddings = processing.get_word_embeddings(data)
        with open('data/whole_tweet_embeddings.json', 'w', encoding="utf8") as fp:
            # print(embeddings)
            json.dump(embeddings,fp,default=default)
        if csv:
            write_csv_embeddings(embeddings)
    
        shut_args = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])
        BertServer.shutdown(shut_args)
    else:
        with open('data/whole_tweet_embeddings.json') as fp:
            embeddings = json.load(fp)
    x = [embedding['embedding'] for embedding in embeddings]
    y = [ 0 if embedding['sentiment'] == 'negative' else 1 if embedding['sentiment'] =='neutral' else 2 for embedding in embeddings]

    # x_train,x_test,y_train,y_test = processing.splitData(np.array(x),np.array(y))
    # processing.torch_split(data)

    processing.convert_to_numpy(y)
    x_train,x_test,y_train,y_test = processing.splitData(x,y)
    if args.ml:
        # linear models
        # logisticRegression
        logisticRegressionPredictions = machine_learning.logisticRegression(x_train,x_test,y_train,y_test)
        metrics.scorer(y_test, logisticRegressionPredictions)

        # non-linear models
        # knn
        knn_predictions = machine_learning.knn(x_train,x_test,y_train,y_test)
        metrics.scorer(y_test, knn_predictions)

        # decisionTreeClassifier
        decisionTreeClassifier_predictions = machine_learning.decisionTreeClassifier(x_train,x_test,y_train,y_test)
        metrics.scorer(y_test, decisionTreeClassifier_predictions)

        # gaussianNB
        gaussianNB_predictions = machine_learning.gaussianNB(x_train,x_test,y_train,y_test)
        metrics.scorer(y_test, gaussianNB_predictions)

        # supportVectorClassification
        svm_predictions = machine_learning.supportVectorClassification(x_train,x_test,y_train,y_test)
        metrics.scorer(y_test, svm_predictions)
        
        # linearDiscriminantAnalysis
        linearDiscriminantAnalysis_predictions = machine_learning.linearDiscriminantAnalysis(x_train,x_test,y_train,y_test)
        metrics.scorer(y_test, linearDiscriminantAnalysis_predictions)

        # copmutes the optimitzed hyper params
        # Set the parameters by cross-validation
        # params = machine_learning.paramOptimizer(x_train,y_train,x_test,y_test)
        #paramOptimizer