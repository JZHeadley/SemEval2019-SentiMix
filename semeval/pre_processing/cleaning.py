
import re
import spacy
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

from progress.bar import ChargingBar
# Zephyr lemmatizing the tweets
def lemmatize(data):
    output=[]
    lemmatizerEn = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)    
    lemmatizerEs = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES,lookup=spacy.lang.es.LOOKUP)    
    bar = ChargingBar('Lemmatizing\t\t\t\t', max=len(data))
    for instance in data:
        new_tweet = {}
        new_tweet['tweetid']=instance['tweetid']
        new_tweet['tweet'] = instance['tweet']
        new_tweet['tokens']=[]
        new_tweet['langid']=instance['langid']
        new_tweet['sentiment']=instance['sentiment']  
        for i,word in enumerate(instance['tokens']):
            if (instance['langid'][i] == 'lang1'):
                new_tweet['tokens'].append(lemmatizerEn.lookup(word))
            elif (instance['langid'][i] == 'lang2'):
                new_tweet['tokens'].append(lemmatizerEs.lookup(word))
            else:
                new_tweet['tokens'].append( word)

            # new_tweet['tokens'].append(lemmatizerEn.lookup(word))
        output.append(new_tweet)
        new_tweet = {}
        new_tweet['tweetid']=instance['tweetid']
        new_tweet['tweet'] = instance['tweet']
        new_tweet['tokens']=[]
        new_tweet['langid']=[]
        new_tweet['sentiment']=instance['sentiment']  
        bar.next()
    bar.finish()
    return output
# Zephyr cleaning  urls and @mentions out of the tweets
def clean_tweets(data):
    output=[]
    urlrx = re.compile(r'^https?:\/\/.*[\r\n]*')
    mentionrx = re.compile(r'^@[a-zA-Z0-9]+')
    bar = ChargingBar('Cleaning Tweets\t\t\t\t', max=len(data))
    for tweet in data:
        new_tweet = {}
        new_tweet['tweetid']=tweet['tweetid']
        new_tweet['tweet'] = tweet['tweet']
        new_tweet['tokens']=[]
        new_tweet['langid']=[]
        new_tweet['sentiment']=tweet['sentiment']
        for i in range(0,len(tweet['tokens'])):
            if tweet['langid'][i] == 'other':
                text = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet['tokens'][i], flags=re.MULTILINE)
                if mentionrx.search(tweet['tokens'][i]):
                    continue
                elif len(text) != 0:
                    new_tweet['tokens'].append(tweet['tokens'][i])
                    new_tweet['langid'].append(tweet['langid'][i])
            else:
                new_tweet['tokens'].append(tweet['tokens'][i])
                new_tweet['langid'].append(tweet['langid'][i])
        output.append(new_tweet)
        new_tweet = {}
        new_tweet['tweetid']=tweet['tweetid']
        new_tweet['tweet'] = tweet['tweet']
        new_tweet['tokens']=[]
        new_tweet['langid']=[]
        new_tweet['sentiment']=tweet['sentiment']
        bar.next()
    bar.finish()
    return output
# Zephyr Removing english and spanish stop words
def remove_stop_words(data):
    spacy_enstopwords = spacy.lang.en.stop_words.STOP_WORDS
    spacy_esstopwords = spacy.lang.es.stop_words.STOP_WORDS
    output=[]
    bar = ChargingBar('Removing Stop Words\t\t\t', max=len(data))
    for instance in data:
        new_tweet = {}
        new_tweet['tweetid']=instance['tweetid']
        new_tweet['tweet'] = instance['tweet']
        new_tweet['tokens']=[]
        new_tweet['langid']=[]
        new_tweet['sentiment']=instance['sentiment']  
        for i,word in enumerate(instance['tokens']):
            if instance['langid'][i] == 'lang1':
                if word not in spacy_enstopwords:
                    new_tweet['tokens'].append(word)
                    new_tweet['langid'].append(instance['langid'][i])
            elif instance['langid'][i] == 'lang2':
                if word not in spacy_esstopwords:
                    new_tweet['tokens'].append(word)
                    new_tweet['langid'].append(instance['langid'][i])
            else:
                new_tweet['tokens'].append(word)
                new_tweet['langid'].append(instance['langid'][i])       
        output.append(new_tweet)
        new_tweet = {}
        new_tweet['tweetid']=instance['tweetid']
        new_tweet['tweet'] = instance['tweet']
        new_tweet['tokens']=[]
        new_tweet['langid']=[]
        new_tweet['sentiment']=instance['sentiment']  
        bar.next()
    bar.finish()
    return output
    
# Zephyr lowercasing everything
def lowercase(data):
    output=[]
    bar = ChargingBar('Lowercasing tokens\t\t\t', max=len(data))
    for instance in data:
        new_tweet = {}
        new_tweet['tweetid']=instance['tweetid']
        new_tweet['tweet'] = instance['tweet']
        new_tweet['tokens']= [x.lower() if instance['langid'][i] != 'other' else x for i, x in enumerate(instance['tokens']) ]
        new_tweet['langid']=instance['langid']
        new_tweet['sentiment']=instance['sentiment']  
        output.append(new_tweet)
        bar.next()
        
    bar.finish()
    return output
