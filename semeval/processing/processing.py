from bert_serving.client import BertClient
from progress.bar import ChargingBar
from sklearn.model_selection import train_test_split

from torchtext import data, datasets
# https://github.com/hanxiao/bert-as-service

def splitData(X, y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  #, random_state = 42)  #TODO what was the split %? #NOTE random_State is a seed. used for debugging

   print("number of X train: ", len(X_train))
   print("number of X test: ", len(X_test))
   print("number of y train: ", len(y_train))
   print("number of y test: ", len(y_test))

   return X_train, X_test, y_train, y_test


# need to run this before you run the code
# the dependencies can be found here https://github.com/hanxiao/bert-as-service
# 
# bert-serving-start -model_dir tmp/ -num_worker=2 
def get_word_embeddings(data):
    bc = BertClient()
    # bc.encode(['First do it', 'then do it right', 'then do it better'])
    embeddings = []
    sentiment_embeddings = []
    # data = data[9268:9272]
    bar = ChargingBar('Calculating word embeddings\t\t\t', max=len(data))
    for instance in data:
        # should encode the join of the tokens array instead
        #kinda a hacky fix to an empty tokens array
        if len(instance['tokens']) == 0:
            embedding = bc.encode([instance['tweet']])
        else: 
            embedding = bc.encode([' '.join(instance['tokens'])])
        embeddings.append(embedding)
        sentiment_embeddings.append({
            "embedding":embedding[0],
            "sentiment":instance['sentiment']
        })
        bar.next()

    bar.finish()
    # print(embeddings)
    # print(len(embeddings), len(embeddings[0]),len(embeddings[0][0]))
    return embeddings,sentiment_embeddings

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

