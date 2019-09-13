import json
from pre_processing import cleaning
import re
import emoji

# https://stackoverflow.com/a/43146653/5472958
def extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

def calculate_emoji_sentiments(data, threshold=10):
    regex = re.compile(r'\d+(.*?)[\u263a-\U0001f645]')
    emoji_sentiments={}
    for instance in data:
        emojis=extract_emojis(instance['tweet'])
        sentiment = 1 if  instance['sentiment']=='positive' else -1 if instance['sentiment']=='negative' else 0 
        for emoji in emojis:
            if emoji not in emoji_sentiments:
                emoji_sentiments[emoji] = sentiment
            else:
                emoji_sentiments[emoji] = emoji_sentiments[emoji] + sentiment
    sortedEmojis = (sorted((value, key) for (key,value) in emoji_sentiments.items()))
    # print(sortedEmojis)
    thresholded_emojis = [x for x in sortedEmojis if x[0] > threshold or x[0] <  -1 * threshold]
    # print(thresholded_emojis)
    mapped_emojis = [('positive',emoji[1]) if emoji[0] > 0 else ('negative',emoji[0])  for emoji in thresholded_emojis]
    print(mapped_emojis)
    return mapped_emojis

if __name__ =='__main__':
    # parse_conll_to_json('train_conll_spanglish.txt','tweets_train.json')
    with open('tweets_train.json') as json_file:
        data = json.load(json_file)
        cleaned = cleaning.clean_tweets(data)
        lowered = cleaning.lowercase(cleaned)
        stopped = cleaning.remove_stop_words(lowered)
        lemmatized = cleaning.lemmatize(stopped)
        emoji_sentiments = calculate_emoji_sentiments(lemmatized)
        with open('output_tweets.json', 'w') as fp:
            json.dump(lemmatized, fp)

