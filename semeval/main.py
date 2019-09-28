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

import numpy as np

# python3 main.py bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4 

def run_spinner(label,duration):
    spinner = Spinner(label)
    for i in range(0,duration*10):
        time.sleep(.1)
        spinner.next()
    spinner.finish()

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

if __name__ =='__main__':
    # parse_conll_to_json('train_conll_spanglish.txt','tweets_train.json')
    
    GET_EMBEDDINGS = False
    CLEAN_DATA = False

    if GET_EMBEDDINGS: 
        # starts a bert as a service
        args = get_run_args()
        server = BertServer(args)
        server.start()
        run_spinner('Loading Bert as a service\t',20)
    
    if CLEAN_DATA:
        with open('data/tweets_train.json') as json_file:
            data = json.load(json_file)
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
    
        shut_args = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])
        BertServer.shutdown(shut_args)
    
    # TODO return this to an else?
    with open('data/whole_tweet_embeddings.json') as fp:
        embeddings = json.load(fp)
    x = [embedding['embedding'] for embedding in embeddings]
    processing.convert_to_numpy(x)
    y = [ 0 if embedding['sentiment'] == 'negative' else 1 if embedding['sentiment'] =='neutral' else 2 for embedding in embeddings]
    processing.convert_to_numpy(y)
    x_train,x_test,y_train,y_test = processing.splitData(x,y)

    # # linear models
    # logisticRegression
    # logisticRegressionPredictions = machine_learning.logisticRegression(x_train,x_test,y_train,y_test)
    # metrics.scorer(y_test, logisticRegressionPredictions)

    # # linearDiscriminantAnalysis
    # linearDiscriminantAnalysis_predictions = machine_learning.linearDiscriminantAnalysis(x_train,x_test,y_train,y_test)
    # metrics.scorer(y_test, linearDiscriminantAnalysis_predictions)

    # # non-linear models
    # # knn
    # knn_predictions = machine_learning.knn(x_train,x_test,y_train,y_test)
    # metrics.scorer(y_test, knn_predictions)

    # # decisionTreeClassifier
    # decisionTreeClassifier_predictions = machine_learning.decisionTreeClassifier(x_train,x_test,y_train,y_test)
    # metrics.scorer(y_test, decisionTreeClassifier_predictions)

    # # gaussianNB
    # gaussianNB_predictions = machine_learning.gaussianNB(x_train,x_test,y_train,y_test)
    # metrics.scorer(y_test, gaussianNB_predictions)

    # # supportVectorClassification
    # supportVectorClassification_predictions = machine_learning.supportVectorClassification(x_train,x_test,y_train,y_test)
    # metrics.scorer(y_test, supportVectorClassification_predictions)


# baseline results: 
# where class 2 is pos, class 1 is neutral, and class 0 is negative
# [[   0    0 4968]
#  [   0    0 2529]
#  [   0    0 7503]]
# accuracy:  0.5002
# error:  0.4998


# sklearn logisticRegression: solver = 'lbfgs' max_iter = 5000 accuracy:  0.386
# sklearn logisticRegression: solver = 'saga' max_iter = 5000 accuracy:  0.4911111111111111
# sklearn logisticRegression: solver = 'sag' max_iter = 5000 accuracy:  0.4911111111111111
# sklearn logisticRegression: solver = 'newton-cg' max_iter = 5000 accuracy:  0.4902222222222222


# sklearn LinearDiscriminantAnalysis: solver='svd' accuracy:  0.4922222222222222 
# sklearn LinearDiscriminantAnalysis: solver='lsqr' accuracy:  0.49955555555555553
# sklearn LinearDiscriminantAnalysis: solver='eigen' accuracy:  0.49133333333333334

# sklearn knn: k = 2 accuracy:  0.386
# sklearn knn: k = 5 accuracy:  0.45222222222222225
# sklearn knn: k = 10 accuracy:  0.46155555555555555
# sklearn knn: k = 15 accuracy:  0.4731111111111111
# sklearn knn: k = 20 accuracy:  0.4706666666666667

# sklearn DecisionTreeClassifier: default all accuracy:  0.412

# sklearn GaussianNB: default all accuracy:  0.4424444444444444

# sklearn Support Vector Classification: gamma='auto'  accuracy:  0.5008888888888889