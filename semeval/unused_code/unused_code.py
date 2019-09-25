import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from torchtext import data
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

# TODO pass in the cleaned data I suppose. maybe probaly clean based on given tokens and then combine the tokens here?
# NOTE This is one my favorite method names of all time
def prepareForBert(data):
   # Load pre-trained model tokenizer (vocabulary)
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 

   # Load pre-trained model (weights)
   model = BertModel.from_pretrained('bert-base-uncased')

   # Put the model in "evaluation" mode, meaning feed-forward operation.
   model.eval() # TODO does this go here? or should it be in the loop? 

   firstRun = True

   bertPreparedTweets = []
   bertSentenceReprenstations = []
   for line in data:
      text = " ".join(line["tokens"])
      marked_text = "[CLS] " + text + " [SEP]" # TODO This is assuming a whole tweet is a single sentence rn. fix dis shiz homz

      tokenized_text = tokenizer.tokenize(marked_text)

      indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

      segments_ids = [1] * len(tokenized_text)

      # Convert inputs to PyTorch tensors
      tokens_tensor = torch.tensor([indexed_tokens])
      segments_tensors = torch.tensor([segments_ids])

      # Predict hidden states features for each layer
      with torch.no_grad(): # NOTE alledgly "(we don't need gradients or backpropagation since we're just running a forward pass)." but why? 
         encoded_layers, _ = model(tokens_tensor, segments_tensors) # NOTE I think this is the word embedding for the sentence (rn the whole tweet) TODO chcek TODO make per sentence, not per tweet

         # Convert the hidden state embeddings into single token vectors

         # Holds the list of 12 layer embeddings for each token
         # Will have the shape: [# tokens, # layers, # features]
         token_embeddings = [] 

         # For each token in the sentence...
         for token_i in range(len(tokenized_text)):
         
            # Holds 12 layers of hidden states for each token 
            hidden_layers = [] 
            
            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers)):

                # For each batch in the sentence...
               for batch_i in range(len(encoded_layers[layer_i])): # NOTE this was not in the tutorial, but without this "batch_i" is undefined, so I guess it's needed TODO check if this is in the correct spot

                  # Lookup the vector for `token_i` in `layer_i`
                  vec = encoded_layers[layer_i][batch_i][token_i]
                  
                  hidden_layers.append(vec)
               
            token_embeddings.append(hidden_layers)

            # The upshot being that, again, the correct pooling strategy (mean, max, concatenation, etc.) and layers used (last four, all, last layer, etc.) is dependent on the application. This discussion of pooling strategies applies both to entire sentence embeddings and individual ELMO-like token embeddings.
            # it almost looks like I can either use word vectors or sentences vectors. In theroy, we have to test both versions
            # at a guess, sentence reps will be faster to do, but words would be more accurate. but im not sure about that 

         sentence_embedding = torch.mean(encoded_layers[11], 1)

         bertSentenceReprenstations.append(sentence_embedding[0])
         
         if (firstRun):
            print ("Our final sentence embedding vector of shape: ", sentence_embedding[0].shape[0])

      # only print the first iteration of the loop as a sanity check
      if (firstRun):
         print("")
         print (tokenized_text)
         for tup in zip(tokenized_text, indexed_tokens):
            print (tup)
         print (segments_ids)

         print ("Number of layers:", len(encoded_layers))
         layer_i = 0

         print ("Number of batches:", len(encoded_layers[layer_i]))
         batch_i = 0

         print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
         token_i = 0

         print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

         # For the 5th token in our sentence, select its feature values from layer 5.
         token_i = 5
         layer_i = 5
         vec = encoded_layers[layer_i][batch_i][token_i]

         # # Plot the values as a histogram to show their distribution.
         # plt.figure(figsize=(10,10))
         # plt.hist(vec, bins=200)
         # plt.show()

         # Sanity check the dimensions:
         print ("Number of tokens in sequence:", len(token_embeddings))
         print ("Number of layers per token:", len(token_embeddings[0]))
         print("")
      firstRun = False
      

      bertPreparedTweets.append(tokenized_text)

   return bertPreparedTweets


def pytorchstuff(spanglishData):
   TEXT = data.Field(tokenize = None) # TODO chcek if tokenize None works
   LABEL = data.LabelField(dtype = torch.float)
   # train_data, test_data = spanglishData.(TEXT, LABEL) #TODO use sklearn to split?

   # print(f'Number of training examples: {len(train_data)}')
   # print(f'Number of testing examples: {len(test_data)}')


# class BertForSequenceClassification(nn.Module):
  
#     def __init__(self, num_labels=2):
#         super(BertForSequenceClassification, self).__init__()
#         self.num_labels = num_labels
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         nn.init.xavier_normal_(self.classifier.weight)
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         return logits



def createTSVFiles(spanglishData):
   with open('tsv_tweets.tsv', 'w', encoding="utf8") as fp:
      fp.write("sentence\tlabel")

      for jsonLine in spanglishData:
         fp.write(jsonLine["tweet"] + "\t" + jsonLine["sentiment"])

   # with open('tsv_tweets.tsv', 'r', encoding="utf8") as tsv_file:
   #    for line in tsv_file:
   #       print(line)


def createSplitTSVFiles(X_train, X_test, y_train, y_test):
   if ((len(X_train) != len(y_train)) or (len(X_test) != len(y_test))):
      print("something went wrong when splitting the data")
      return

   with open('train.tsv', 'w', encoding="utf8") as fp:
      fp.write("sentence\tlabel")

      for i,line in enumerate(X_train):
         fp.write(" ".join(line) + "\t" + str(y_train[i]))

   with open('dev.tsv', 'w', encoding="utf8") as fp:
      fp.write("sentence\tlabel")

      for i,line in enumerate(X_test):
         fp.write(" ".join(line) + "\t" + str(y_test[i]))