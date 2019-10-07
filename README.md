## Installation of Dependencies
In the root directory of this project run `pipenv install`, this will install all of the dependencies that the project needs and will create the virtual environment.  If you don't have pipenv you can download it [here](https://github.com/pypa/pipenv).   


#### Downloading the model
##### The script way
Run this command in the root of the directory and it will download the model and extract it correctly.
```bash
wget -qO- https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip | bsdtar -xvf-
```
###### Long way
Once we have this the bert model that we want to use needs to be downloaded and extracted into a directory.  Since we're working on a mix of languages we're going to use the bert mixed model which can be downloaded [here](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip).  To make things easier, I'd recommend moving that downloaded zip to the root of the project directory and unzipping it there.  This will unzip into a folder named `multi_cased_L-12_H-768_A-12/` which contains all the files for the pretrained model.

## Running the code
To be able to use our dependencies in the virtual environment you'll need to run `pipenv shell` to get a session within it.  To make our word embeddings easier we're using [bert as a service](https://github.com/hanxiao/bert-as-service) which needs to be started in a separate shell with `bert-serving-start -model_dir ./multi_cased_L-12_H-768_A-12/ -num_worker=4 ` 

Once the bert as a service server is running we can run our code, so get a new virtual environment session in a new shell and from the root directory run `python semeval/main.py` and everything should start working

#### Project Description
This project aims to predict if a spanglish tweet has a "positive", "negative", or "neutral" sentiment. This is achieved by taking in tweets (see 'train_conll_spanglish.txt' in the data folder and 'Sample Training Data'.) These giving tweets are converted into a more usbale JSON format and then preprocessed. The preprocessing is composed of cleaning the tweets (removing URLS, @'s, and #'s), removing both english and spanish stop words, lemmetizing words to either the base English or Spanish word, and finally converting all characters to lowercase. Once this is completed for each tweet, each whole tweet the gets passed to 'Bert-as-a-service.' This computes a word embedding of 756 features (see 'Sample Cleaning and Embedding Process' for an example of this in action.) It should be noted that main includes flags that enable to the user to turn off the cleaning and embedding process. These should be enabled if eithee there already exists a file of the cleaned tweets in JSON format, and/or if there already exists a JSON file of the embeddings. These embeddings are then split into a 60-40 training, testing split using sklearn. This split was done with a defined 'random_state', so that each classifaction method would use the same data splits. These enabels a more reliable comparison of results. These data splits are then passed to different machine learning models. Currently, this project uses two linear models and four non-linear models, all from sklearn. The linear models are: Logistic Regression and Linear Discriminant Analysis. The non-linear models include: knn, Decision Tree Classifier, Gaussian NB, and Support Vector Classification. The results of these models, with limited trials experemienteing with hyperparameters can be found in 'metrics/results.txt.'

Inorder to compute a baseline for the results to be compared against, we deteremined which sentiment had the most frequent occurances. Our baseline was the accuracy of a model that would predict the most frequent sentiment for each tweet.

This project also attempted a naive approach to classifying tweets via emojis. This consisted of creating a map of all emojis that appeared in the given tweets. Then by looping through by each tweet, everytime an emoji was seen, increment its dictionary value if the tweet was 'positive' and decrementing the value if the tweet had a sentiment of 'negative.' 'Neutral tweets had no impact on an emoji's value. Predicitions would occur by summing up the collective value of each emoji that appeared in a agiven tweet and classifying said tweet as 'positive' if the emoji score was above zero, 'negative' if the score was below zero, and 'neutral' if the sum was zero. For tweets that contained no emojis, the baseline approach was used.


#### Sample Training Data
meta	1	positive
So	lang1
that	lang1
means	lang1
tomorrow	lang1
cruda	lang2
segura	lang2
lol	lang1

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
{"tokens": ["@Gaaybee", "lo", "levantaste", "con", "una", "de", "tus", "bombas", "lmao"], "langid": ["other", "lang2", "lang2", "lang2", "lang2", "lang2", "lang2", "lang2", "lang1"], "tweet": " @Gaaybee lo levantaste con una de tus bombas lmao", "tweetid": 17, "sentiment": "positive"}

Cleaned
{"tweetid": 17, "tweet": " @Gaaybee lo levantaste con una de tus bombas lmao", "tokens": ["levantar", "bombo", "lmao"], "langid": ["lang2", "lang2", "lang1"], "sentiment": "positive"}

Embedding
{"embedding": [0.2900029122829437, 0.017484011128544807, -0.10647160559892654, 0.6051885485649109, 0.24973852932453156, 0.021043820306658745,etc (goes for 756 features)]}