from bert_serving.client import BertClient
from progress.bar import ChargingBar
from sklearn.model_selection import train_test_split

import numpy as np

# https://github.com/hanxiao/bert-as-service

# TODO is it worth noramlizing the embeddings? https://stats.stackexchange.com/questions/177905/should-i-normalize-word2vecs-word-vectors-before-using-them

# ALWIN - splits the data into a 60-40 split via sklearn. Includes a 'random_state' to allow for result compariosn of different machine learning models
def splitData(x,y):
    #NOTE random_State is a seed. used for debugging and comparing performance of different ml models
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state = 42) 
    # print("number of X train: ", len(x_train))
    # print("number of X test: ", len(x_test))
    # print("number of y train: ", len(y_train))
    # print("number of y test: ", len(y_test))

    return x_train, x_test, y_train, y_test


# ZEPHYR
# need to run this before you run the code
# the dependencies can be found here https://github.com/hanxiao/bert-as-service
# 
# bert-serving-start -model_dir tmp/ -num_worker=2 
# Zephyr method to get embeddings of the tweets
def get_word_embeddings(data):
    bc = BertClient()
    # bc.encode(['First do it', 'then do it right', 'then do it better'])

    embeddings = []
    sentiment_embeddings = []
    bar = ChargingBar('Calculating tweet embeddings\t\t\t', max=len(data))
    for instance in data:
        # should encode the join of the tokens array instead
        # kinda a hacky fix to an empty tokens array
        if len(instance['tokens']) == 0:
            embedding = bc.encode([instance['tweet']])
        else: 
            embedding = bc.encode([' '.join(instance['tokens'])])
        embeddings.append(embedding)
        sentiment_embeddings.append({
            "embedding": embedding[0],
            "sentiment": instance['sentiment']
        })
        bar.next()

    bar.finish()
    # print(embeddings)
    # print(len(embeddings), len(embeddings[0]),len(embeddings[0][0]))
    return embeddings,sentiment_embeddings


# checks if using bert for the whole tweet gives a different result than averaging each word embedding
# TODO if you get a different embedding, make a json file of the averaged word embedding
# TODO try a different combo method other than averaging
# ALWIN - an other version of bert, per word and not per tweet
def compare_word_embeddings(data): 
    data = data[:1]
    bc = BertClient()

    embeddings = []
    sentiment_embeddings = []
    # word embedding average embedding 
    bar = ChargingBar('Calculating tweet embeddings\t\t\t', max=len(data))
    for instance in data:
        if len(instance['tokens']) == 0:
            embedding = bc.encode([instance['tweet']])
        else: 
            embedding = bc.encode([' '.join(instance['tokens'])])

        print("tweet embedding: " + embedding)
        bar.next()
    bar.finish()

    # whole tweet embedding
    bar = ChargingBar('Calculating word average embeddings\t\t\t', max=len(data))
    for instance in data:
        if len(instance['tokens']) == 0:
            embedding = bc.encode([instance['tweet']])
        else:
            tweet_embedding = []
            word_embeddings = []
            for word in instance['tokens']:
                word_embeddings.append(bc.encode(word))

            # for each feature in the embedding, calucalute the avg of said feature for each word
            # all word embeddings should be the same size because thats how embeddings work
            # TODO add a check, probably
            for feature in range(len(word_embeddings)):
                featureAvg = 0
                for word_embedding in word_embeddings:
                    featureAvg += word_embedding[feature]
                featureAvg /= len(word_embeddings)
                tweet_embedding.append(featureAvg)
                
        print("word average embedding: " + tweet_embedding)
        bar.next()
    bar.finish()
    return 

# ALWIN - converts a list to numpy
def convert_to_numpy(data):
    np_data = np.array(data)

    # print statements are here to ensure that teh outcome is the proper shapes, also ensures embeddings worked
    # print("np_data shape: ")
    # print(np_data.shape)

    # print("np_data[0]")
    # print(np_data[0])

    # print("data[0]")
    # print(data[0])
    return np_data

# Zephyr splitting for use in pytorch
def torch_split(dataset):  
    TOKENS = data.Field()
    LANGID = data.Field()
    SENTIMENT = data.Field()
    fields = {
        "tokens":('tokens',TOKENS), 
        "langid":("langid",LANGID),
        "sentiment":("sentiment",SENTIMENT)
    }
    train_data,test_data = data.TabularDataset.splits(
        path='data',
        train = 'train.json',
        test = 'test.json',
        format = 'json',
        fields = fields
    )
    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data), 
        batch_size = BATCH_SIZE,
        device = device)