import json
import sys
import time
from bert_serving.server import BertServer
from bert_serving.server.helper import get_run_args,get_shutdown_parser
from pre_processing import cleaning
from metrics import metrics
from processing import processing
from progress.spinner import Spinner

# C:\Users\Alwin Hollebrandse\AppData\Local\Programs\Python\Python37\Lib\site-packages 

import numpy as np

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

    # TODO maybe comment these out
    # starts a bert as a service, because fuck windows path
    args = get_run_args()
    server = BertServer(args)
    server.start()
    # server.join()
    run_spinner('Loading Bert as a service\t',20)
    print("still running")
    with open('tweets_train.json') as json_file:
        data = json.load(json_file)
        cleaned = cleaning.clean_tweets(data)
        lowered = cleaning.lowercase(cleaned)
        stopped = cleaning.remove_stop_words(lowered)
        lemmatized = cleaning.lemmatize(stopped)
        emoji_sentiments = metrics.calculate_emoji_sentiments(lemmatized)
        metrics.get_emoji_baseline(data,emoji_sentiments)
        embeddings,sentiment_embeddings = processing.get_word_embeddings(lemmatized)
        with open('output_tweets.json', 'w') as fp:
            json.dump(lemmatized, fp)
        
        with open('whole_tweet_embeddings.json', 'w', encoding="utf8") as fp:
            print(sentiment_embeddings)
            json.dump(sentiment_embeddings,fp,default=default)
    shut_args = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])
    BertServer.shutdown(shut_args)

