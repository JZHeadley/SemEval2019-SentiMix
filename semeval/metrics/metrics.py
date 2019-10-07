import re
import emoji
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from progress.bar import ChargingBar


# https://stackoverflow.com/a/43146653/5472958
# Zephyr Simple regex pattern that extracts emojis.  Stolen from the above link
def extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

# Zephyr Calculates the emoji map and which emojis should be positive, negative, or unimportant
def calculate_emoji_sentiments(data, threshold=10):
    regex = re.compile(r'\d+(.*?)[\u263a-\U0001f645]')
    emoji_sentiments={}
    bar = ChargingBar('Calculating Emoji Map\t\t\t', max=len(data))
    for instance in data:
        emojis=extract_emojis(instance['tweet'])
        sentiment = 1 if  instance['sentiment']=='positive' else -1 if instance['sentiment']=='negative' else 0 
        for emoji in emojis:
            if emoji not in emoji_sentiments:
                emoji_sentiments[emoji] = sentiment
            else:
                emoji_sentiments[emoji] = emoji_sentiments[emoji] + sentiment
        bar.next()
    sortedEmojis = (sorted((value, key) for (key,value) in emoji_sentiments.items()))
    # print(sortedEmojis)
    thresholded_emojis = [x for x in sortedEmojis if x[0] > threshold or x[0] <  -1 * threshold]
    # print(thresholded_emojis)
    emoji_map = {}
    for emoji in thresholded_emojis:
        emoji_map[emoji[1]] = 1 if emoji[0] > 0 else -1
    # mapped_emojis = [('positive',emoji[1]) if emoji[0] > 0 else ('negative',emoji[1])  for emoji in thresholded_emojis]
    # print(emoji_map)
    bar.finish()
    return emoji_map

# Zephyr Uses the emoji map produced by the above method to calculate accuracy of the map
def get_emoji_baseline(data,emoji_map):
    predictions =[]
    emoji_tweet_labels =[]
    bar = ChargingBar('Calculating Emoji Map Accuracy\t\t', max=len(data))
    for tweet in data:
        score=0
        emojis = extract_emojis(tweet['tweet'])
        bar.next()
        if len(emojis) > 0:
            emoji_tweet_labels.append(-1 if tweet['sentiment'] == 'negative' else 1 if tweet['sentiment'] =='positive' else 0)
        else:
            continue
        for emoji in emojis:
            if emoji in emoji_map:
                score = score + emoji_map[emoji]
        label = 1 if score > 0 else -1 if score < 0 else 0
        predictions.append(label)
        # print(tweet['sentiment'],label)
    bar.finish()
    print('Accuracy of emoji map is:',accuracy_score(predictions,emoji_tweet_labels))


# TODO should this be based off of the test set? what happens when X-fold happens? I think im gonna use the whole dataset for this.
def getBaselinePredicitions(numOfPosSenti, numOfNegSenti, numOfNeutSenti, y_true):
   
   # create y_pred to be init with the most frequent sentiment seen in the data
   sentiToPredict = -1
   mostFrequentSenti = max(numOfPosSenti, numOfNegSenti, numOfNeutSenti)
   if (mostFrequentSenti == numOfPosSenti):
      sentiToPredict = 2
   elif (mostFrequentSenti == numOfPosSenti):
      sentiToPredict = 1
   elif (mostFrequentSenti == numOfPosSenti):
      sentiToPredict = 0
   y_pred = [sentiToPredict] * len(y_true)

   scorer(y_true, y_pred)


def scorer(y_true, y_pred):
   print("accuracy: ",accuracy_score(y_true, y_pred))
   print(confusion_matrix(y_true, y_pred))
   print(classification_report(y_true, y_pred))


def getUniqueTokens(data):
   uniqueTokens = {}

   for line in data: # needs to be strings
      for token in line['tokens']:
         if token not in uniqueTokens: 
            uniqueTokens[token] = 0
         uniqueTokens[token] += 1

   print("number of unique tokens: ", len(uniqueTokens))

   return uniqueTokens


def getDataBySentiment(data): 
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

      print("number of all sentiments: ", len(spanglishData))
      print("number of positive sentiments: ", len(posSenti))
      print("number of negative sentiments: ", len(negSenti))
      print("number of neutral sentiments: ", len(neutSenti))

   return posSenti, negSenti, neutSenti


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