import re
import emoji
from sklearn.metrics import accuracy_score
from progress.bar import ChargingBar

# https://stackoverflow.com/a/43146653/5472958
def extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

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
