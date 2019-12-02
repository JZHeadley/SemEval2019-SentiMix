import re
import regex
import emoji
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

from progress.bar import ChargingBar


# https://stackoverflow.com/a/43146653/5472958
# Alwin, Zephyr Simple regex pattern that extracts emojis and flags.  Stolen from the above link
def extract_emojis(text):
   allchars = [str for str in text]
   list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
   return list


# Alwin, Zephyr Calculates the emoji map and which emojis should be positive, negative, or neutral
# TODO read add some importance threshold?
def calculate_emoji_sentiments(data):
   regex = re.compile(r'\d+(.*?)[\u263a-\U0001f645]')
   emoji_sentiments = {}
   bar = ChargingBar('Calculating Emoji Map\t\t\t', max=len(data))

   for instance in data:
      emojis = extract_emojis(instance['tweet'])

      for emoji in emojis:
         if emoji not in emoji_sentiments:
            emoji_sentiments[emoji] = {}
            emoji_sentiments[emoji]['positive'] = 0
            emoji_sentiments[emoji]['neutral'] = 0
            emoji_sentiments[emoji]['negative'] = 0
         
         if (instance['sentiment'] == 'positive'):
            emoji_sentiments[emoji]['positive'] += 1
         elif (instance['sentiment'] == 'neutral'):
            emoji_sentiments[emoji]['neutral'] += 1
         elif (instance['sentiment'] == 'negative'):
            emoji_sentiments[emoji]['negative'] += 1
         
      bar.next()
   bar.finish()
   return emoji_sentiments


def get_all_emojis_used(data):
   allEmojisInData = {}
   bar = ChargingBar('Finding all emojis used in data\t\t', max=len(data))
   for instace in data:
      tweet = instace['tweet']
      emojis = extract_emojis(tweet)

      for emoji in emojis:
         if emoji not in allEmojisInData:
            allEmojisInData[emoji] = 0
      bar.next()

   bar.finish()
   return allEmojisInData


def get_emojis_of_tweet(tweet, allEmojisInData):
   # print('allEmojisInData: ', allEmojisInData)
   emojis = extract_emojis(tweet)

   for emoji in emojis: #  TODO does this cahnge all emojis in data?
      allEmojisInData[emoji] += 1

   emojiFeatureList = []
   for emoji in allEmojisInData:
      emojiFeatureList.append(allEmojisInData[emoji])

   # print('emojiFeatureList: ', emojiFeatureList)
   result = np.array(emojiFeatureList)
   result =result.reshape((1, len(result)))
   return np.array(emojiFeatureList)


# Alwin, Zephyr Uses the emoji maps produced by the above method to calculate accuracy of the map
def get_emoji_baseline(data, mostFrequentSentiment, emoji_sentiments):
   predictions =[]
   bar = ChargingBar('Calculating Emoji Map Accuracy\t\t', max=len(data))
   for instance in data:
      emojis = extract_emojis(instance['tweet'])

      if (len(emojis) > 0):
         emojiSentiment = getEmojiSentiment(emojis, emoji_sentiments)
         predictedSentiment = emojiSentiment
      else:
         predictedSentiment = mostFrequentSentiment

      predictions.append(predictedSentiment)
      bar.next()

   bar.finish()
   return predictions


# Alwin - given a list of emojis, determines which sentiment to predict based solely on emojis
def getEmojiSentiment(emojis, emoji_sentiments):
   if (len(emojis) == 0):
      print('must pass in emojis')
      return

   positiveScore = 0
   neutralScore = 0
   negativeScore = 0

   for emoji in emojis:
      if (emoji in emoji_sentiments):
         positiveScore += emoji_sentiments[emoji]['positive']
         neutralScore += emoji_sentiments[emoji]['neutral']
         negativeScore += emoji_sentiments[emoji]['negative']

   maxScore = max(positiveScore, neutralScore, negativeScore)

   if (maxScore == positiveScore):
      return 2 # 'positive'
   elif (maxScore == neutralScore):
      return 1 # 'neutral'
   else:
      return 0 # 'negative'
         

# ALWIN - predicts the most frequent class for each tweet and computes its score
def getBaselinePredicitions(mostFrequentSentiment, y_true):
   y_pred = [mostFrequentSentiment] * len(y_true)
   scorer(y_true, y_pred)


# ALWIN - create y_pred to be init with the most frequent sentiment seen in the data
def getMostFreqSentiment(numOfPosSenti, numOfNeutSenti, numOfNegSenti):
   sentiToPredict = -1
   mostFrequentSenti = max(numOfPosSenti, numOfNeutSenti, numOfNegSenti)
   if (mostFrequentSenti == numOfPosSenti):
      sentiToPredict = 2
   elif (mostFrequentSenti == numOfNeutSenti):
      sentiToPredict = 1
   elif (mostFrequentSenti == numOfNegSenti):
      sentiToPredict = 0
   return sentiToPredict


# ALWIN - computes the score of a given prediciton list, given the correct answers
def scorer(y_true, y_pred):
   print("accuracy: ", accuracy_score(y_true, y_pred))
   print(confusion_matrix(y_true, y_pred))
   print(classification_report(y_true, y_pred))

   f1 = f1_score(y_true, y_pred, average='micro')
   return f1


# ALWIN - returns a list of all tweets that have emojis, and another list of the tweets that dont
def splitTweetsByEmoji(data):
   emojiTweets = []
   nonEmojiTweets = []
   for instance in data:
      tweet = instance['tweet']
      emojis = extract_emojis(tweet)
      if (len(emojis) > 0):
         emojiTweets.append(instance)
      else:
         nonEmojiTweets.append(instance)

   print('len(emojiTweets): ', len(emojiTweets))
   print('len(nonEmojiTweets): ', len(nonEmojiTweets))
   return emojiTweets, nonEmojiTweets 


# ALWIN - returns a list of all tweets that have more capital letters than tokens, and another list of the tweets that dont
def splitTweetsByCaps(data):
   capTweets = []
   nonCapTweets = []
   for instance in data:
      tweet = instance['tweet']
      if (checkIfMoreCapsThanTokens(tweet)):
         capTweets.append(instance)
      else:
         nonCapTweets.append(instance)

   print('len(capTweets): ', len(capTweets))
   print('len(nonCapTweets): ', len(nonCapTweets))
   return capTweets, nonCapTweets 


# Alwin - checks if a list of tokens has >= upper cased letters than number of tokens
def checkIfMoreCapsThanTokens(tokens):
   numCaps = sum(1 for c in tokens if c.isupper())
   print(numCaps, len(tokens))
   if (numCaps >= len(tokens)):
      return True
   return False


# ALWIN - gets a dictionary of each unique token that appears in the data, along with the frequency count
def getUniqueTokens(data):
   uniqueTokens = {}
   for line in data: # needs to be strings
      for token in line['tokens']:
         if token not in uniqueTokens: 
            uniqueTokens[token] = 0
         uniqueTokens[token] += 1

   print("number of unique tokens: ", len(uniqueTokens))
   return uniqueTokens


# ALWIN - returns the data split according to their sentiments
def getSentimentCounts(data): 
   posSenti = []
   negSenti = []
   neutSenti = []
   spanglishData = []

   for instance in data:
      spanglishData.append(instance)

      if (instance["sentiment"] == "positive"):
         posSenti.append(instance)
      elif (instance["sentiment"] == "negative"):
         negSenti.append(instance)
      elif (instance["sentiment"] == "neutral"):
         neutSenti.append(instance)

   return len(posSenti), len(neutSenti), len(negSenti)


# ALWIN - returns X (the tweet data) and y (the sentiment)
def getXandy(data): 
   X = []
   y = []

   for instance in data:
      X.append(instance["tokens"]) 

      if (instance["sentiment"] == "positive"):
         y.append(2)
      elif (instance["sentiment"] == "negative"):
         y.append(1)
      elif (instance["sentiment"] == "neutral"):
         y.append(0)

   return X, y