from bert_serving.client import BertClient
from progress.bar import ChargingBar
from sklearn.model_selection import train_test_split

# https://github.com/hanxiao/bert-as-service

def splitData(X, y):
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  #, random_state = 42)  #TODO what was the split %? #NOTE random_State is a seed. used for debugging

   print("number of X train: ", len(X_train))
   print("number of X test: ", len(X_test))
   print("number of y train: ", len(y_train))
   print("number of y test: ", len(y_test))

   return X_train, X_test, y_train, y_test


def get_word_embeddings(data):
    bc = BertClient()
    # bc.encode(['First do it', 'then do it right', 'then do it better'])
    embeddings = []
    data = data[:10]
    bar = ChargingBar('Calculating word embeddings\t\t\t', max=len(data))
    with open('whole_tweet_embeddings.json', 'w', encoding="utf8") as fp:
        fp.write("{")
        fp.write("instances: [")

        for instance in data[:10]:
            # should encode the join of the tokens array instead
            embedding = bc.encode([' '.join(instance['tokens'])])
            embeddings.append(embedding)
            fp.write("{\"embedding\": " + embedding + ", \"sentiment\":" + instance["sentiment"] + "}")
            bar.next()

        fp.write("]")
        fp.write("}")

    bar.finish()
    # print(embeddings)
    print(len(embeddings), len(embeddings[0]),len(embeddings[0][0]))
    return embeddings