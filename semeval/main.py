import json
from pre_processing import cleaning

if __name__ =='__main__':
    # parse_conll_to_json('train_conll_spanglish.txt','tweets_train.json')
    with open('tweets_train.json') as json_file:
        data = json.load(json_file)
        cleaned = cleaning.clean_tweets(data)
        stopped = cleaning.remove_stop_words(cleaned)
        lemmatized = cleaning.lemmatize(stopped)
        with open('output_tweets.json', 'w') as fp:
            json.dump(lemmatized, fp)

