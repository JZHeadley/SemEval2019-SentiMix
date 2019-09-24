import json
import sys

from bert_serving.server import BertServer
from bert_serving.server.helper import get_run_args
from pre_processing import cleaning
from metrics import metrics
from processing import processing

# C:\Users\Alwin Hollebrandse\AppData\Local\Programs\Python\Python37\Lib\site-packages 

if __name__ =='__main__':
    # parse_conll_to_json('train_conll_spanglish.txt','tweets_train.json')

    # TODO maybe comment these out
    # starts a bert as a service, because fuck windows path
    args = get_run_args()
    server = BertServer(args)
    server.start()
    server.join()

    print("still running")
    with open('tweets_train.json') as json_file:
        data = json.load(json_file)
        cleaned = cleaning.clean_tweets(data)
        lowered = cleaning.lowercase(cleaned)
        stopped = cleaning.remove_stop_words(lowered)
        lemmatized = cleaning.lemmatize(stopped)
        emoji_sentiments = metrics.calculate_emoji_sentiments(lemmatized)
        metrics.get_emoji_baseline(data,emoji_sentiments)
        processing.get_word_embeddings(lemmatized)
        with open('output_tweets.json', 'w') as fp:
            json.dump(lemmatized, fp)