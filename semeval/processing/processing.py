from bert_serving.client import BertClient
from progress.bar import ChargingBar

# need to run this before you run the code
# the dependencies can be found here https://github.com/hanxiao/bert-as-service
# 
# bert-serving-start -model_dir tmp/ -num_worker=2 
def get_word_embeddings(data):
    bc = BertClient()
    # bc.encode(['First do it', 'then do it right', 'then do it better'])
    embeddings = []
    bar = ChargingBar('Calculating word embeddings\t\t\t', max=len(data))
    for instance in data[:10]:
        # should encode the join of the tokens array instead
        embeddings.append(bc.encode([' '.join(instance['tokens'])]))
        bar.next()
    bar.finish()
    # print(embeddings)
    print(len(embeddings), len(embeddings[0]),len(embeddings[0][0]))
    return embeddings