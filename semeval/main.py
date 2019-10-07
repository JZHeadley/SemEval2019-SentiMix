import json
import sys
import time
from bert_serving.server import BertServer
from bert_serving.server.helper import get_run_args,get_shutdown_parser
from pre_processing import cleaning
from metrics import metrics
from processing import processing
from machine_learning import machine_learning
from progress.spinner import Spinner
import csv 
import numpy as np

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

if __name__ =='__main__':
    # parse_conll_to_json('train_conll_spanglish.txt','tweets_train.json')

    CLEAN_DATA = True
    RUN_PROCESSING = True
    WRITE_CSV = True
    GET_EMBEDDINGS = True
    if GET_EMBEDDINGS: 
        args = get_run_args()
        server = BertServer(args)
        server.start()
        run_spinner('Loading Bert as a service\t',20)
    if CLEAN_DATA:
        with open('data/tweets_train.json') as json_file:
            data = json.load(json_file)
            # data = data[:1000]
        cleaned = cleaning.clean_tweets(data)
        lowered = cleaning.lowercase(cleaned)
        stopped = cleaning.remove_stop_words(lowered)
        lemmatized = cleaning.lemmatize(stopped)
        emoji_sentiments = metrics.calculate_emoji_sentiments(lemmatized)
        metrics.get_emoji_baseline(data,emoji_sentiments)
        with open('data/output_tweets.json', 'w') as fp:
            json.dump(lemmatized, fp)
    else:
        with open('data/output_tweets.json', 'r') as fp:
            lemmatized = json.load(fp)


    if GET_EMBEDDINGS:
        _, embeddings = processing.get_word_embeddings(lemmatized)
        with open('data/whole_tweet_embeddings.json', 'w', encoding="utf8") as fp:
            print(embeddings)
            json.dump(embeddings,fp,default=default)
        if WRITE_CSV:
            write_csv_embeddings(embeddings)
    
        shut_args = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])
        BertServer.shutdown(shut_args)
    else:
        with open('data/whole_tweet_embeddings.json') as fp:
            embeddings = json.load(fp)
    x = [embedding['embedding'] for embedding in embeddings]
    y = [ 0 if embedding['sentiment'] == 'negative' else 1 if embedding['sentiment'] =='neutral' else 2 for embedding in embeddings]

    # x_train,x_test,y_train,y_test = processing.splitData(np.array(x),np.array(y))
    # processing.torch_split(lemmatized)

    processing.convert_to_numpy(y)
    x_train,x_test,y_train,y_test = processing.splitData(x,y)

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