# Project Description
This project aims to predict if a spanglish tweet has a "positive", "negative", or "neutral" sentiment. This is achieved by taking in tweets (see 'train_conll_spanglish.txt' in the data folder and 'Sample Training Data'.) These given tweets are converted into a more usable JSON format and then preprocessed. The preprocessing is composed of cleaning the tweets (removing URLS, @'s, and #'s), removing both English and Spanish stop words, lemmetizing words to either the base English or Spanish word, and finally converting all characters to lowercase. 

Once this is completed for each tweet, each tweet gets embedded. This occurs in two methods. The first apporach is to pass the whole ‘tweet’ to Bert-as-a-Service. The second is to pass each word of a given tweet to Bert and then combine the word embeddings for said tweet by averaging the values together. Both of these approaches results with a word embedding of 756 features (see 'Sample Cleaning and Embedding Process' for an example of this in action.) In addition to these two embedding files, there are another two. The embeddings are duplicated and appended with emoji features. These features are computed by going through all the data and getting a list of emojis that appear in the Twitter data. When all of these steps are completed, the result is four different JSON files.  

It should be noted that main includes flags that enable to the user to turn off the cleaning and embedding process. These should be enabled if either there already exists a file of the cleaned tweets in JSON format, and/or if there already exists a JSON file of the embeddings. 

Additionally, one could easily further this process by commenting out the ‘remove stopwords’ line and by changing the names of the resulting embedding JSON files. If this is done in conjunction with the stopword-less version, one should have created eight different embedding JSON files. To use all eight of these files in the machine learning portion, one would need to add the stopword-less JSON file names to ‘embeddingFileNames’.  

Once all the embeddings are created and/or retrieved, and if the machine learning flag (‘--ml’) is added in the command line, this program will attempt to create predictions. This is done by going through each of the ‘embeddingFileNames’. Each JSON file is then opened and split into ‘x’ and ‘y’ data. These data sets are then split into a 60-40 training, testing split using sklearn. This split was done with a defined 'random_state', so that each classification method would use the same data splits. These enables a more reliable comparison of results. These data splits are then passed to different machine learning models. Currently, this project optimizes the hyper parameters for two different models: “Decision Tree Classifier” and “Support Vector Classifier.” These models also use cross validation to improve their results. This will display the best parameters to use for each model and the highest f-score of the model with a given embedding file. We have already implemented the skeleton to also optimize the following models: Logistic Regression,  Linear Discriminant Analysis, knn, and  Gaussian NB. However, these were not fully completed. The results of these models, with limited trials experimenting with hyper parameters can be found in 'metrics/results.txt.'

Inorder to compute a baseline for the results to be compared against, we determined which sentiment had the most frequent occurrences. Our baseline was the accuracy of a model that would predict the most frequent sentiment for each tweet.

# Emoji Map
This project also attempted a naive approach to classifying tweets via emojis. This consisted of creating a map of all emojis that appeared in the given tweets. Each of these emojis would have a ‘positive’, ‘neutral’, and ‘negative’ frequency counter. These would be computed by looping through each tweet, and for each emoji in said tweet, incrementing its dictionary value based on said tweet’s sentiment.  Predictions would occur by summing up the collective value of each emoji that appeared in a given tweet and classifying said tweet as as the sentiment that matched the frequency of emoji sentiments. For tweets that contained no emojis, the baseline approach was used.

This project assumes that a 'positive' sentiment gets assigned a numeric value of '2','negative' gets assigned to '1,' and the 'neutral' sentiment gets to the value of '0.'

## Installation of Dependencies
If you don't have pipenv you can download it [here](https://github.com/pypa/pipenv).  In the root directory of this project run `pipenv install --skip-lock`, this will install all of the dependencies that the project needs and will create the virtual environment.  
#### Downloading the models
##### The script way
Run this command in the root of the directory and it will download the model and extract it correctly.
```bash
wget -qO- https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip | bsdtar -xvf-
```

To install the Spacy models run 
```bash
pipenv run python -m spacy download en_core_web_sm
pipenv run python -m spacy download es_core_news_sm
```
###### Long way
Once we have this the bert model that we want to use needs to be downloaded and extracted into a directory.  Since we're working on a mix of languages we're going to use the bert mixed model which can be downloaded [here](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip).  To make things easier, I'd recommend moving that downloaded zip to the root of the project directory and unzipping it there.  This will unzip into a folder named `multi_cased_L-12_H-768_A-12/` which contains all the files for the pretrained model.

## Running the code
~~To be able to use our dependencies in the virtual environment you'll need to run `pipenv shell` to get a session within it.  To make our word embeddings easier we're using [bert as a service](https://github.com/hanxiao/bert-as-service) which needs to be started in a separate shell with `bert-serving-start -model_dir ./multi_cased_L-12_H-768_A-12/ -num_worker=4 `~~ Swapped to running the server with code so don't need to start it separately anymore

To run the code you'll first have to enter into a pipenv virtual environment with `pipenv shell` this should give you all the packages you installed earlier with `pipenv install` and will allow you to run everything.  From here  have some command line arguments that enable or disable certain parts of the code, but for the first run you'll want to start with:
```
python semeval/main.py -model_dir <path to model dir> -num_worker=2 --clean --embeddings --ml
```
  That will run the cleaning of the tweets, get the word embeddings, and run the machine learning models on that.  When you clean the tweets and get the embeddings intermediate files are written out so you don't have to wait forever again, which is why we have them optioned this way so we can easily disable them while developing further portions. 

An example filled in command to run the project would be:
```
python semeval/main.py --clean --embeddings -model_dir tmp/ -num_worker=1 --ml --csv    
```
#### Sample Training Data
```conll
meta	1	positive
So	lang1
that	lang1
means	lang1
tomorrow	lang1
cruda	lang2
segura	lang2
lol	lang1
```
Please note that lang1 is English and lang2 is Spanish. On the first line, '1' is the tweet id and 'positive' is the sentiment of the tweet.


#### Sample Cleaning and Embedding Process
Original
meta	17	positive
@Gaaybee	other
lo	lang2
levantaste	lang2
con	lang2
una	lang2
de	lang2
tus	lang2
bombas	lang2
lmao	lang1

JSON
```json
{"tokens": ["@Gaaybee", "lo", "levantaste", "con", "una", "de", "tus", "bombas", "lmao"], "langid": ["other", "lang2", "lang2", "lang2", "lang2", "lang2", "lang2", "lang2", "lang1"], "tweet": " @Gaaybee lo levantaste con una de tus bombas lmao", "tweetid": 17, "sentiment": "positive"}
```
Cleaned
```json
{"tweetid": 17, "tweet": " @Gaaybee lo levantaste con una de tus bombas lmao", "tokens": ["levantar", "bombo", "lmao"], "langid": ["lang2", "lang2", "lang1"], "sentiment": "positive"}
```
Embedding
```json
{"embedding": [0.2900029122829437, 0.017484011128544807, -0.10647160559892654, 0.6051885485649109, 0.24973852932453156, 0.021043820306658745,etc (goes for 756 features)]}
```