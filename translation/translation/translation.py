from googletrans import Translator
# from google.cloud import translate_v3beta1 as translate

import json
import time
import re

import spacy
from spacy.lang.en import English
from spacy.lang.es import Spanish
import en_core_web_md
import es_core_news_md

from progress.bar import ChargingBar

#import xx_ent_wiki_sm


def parse_conll_to_json():
    with open('train_conll_spanglish.txt') as fp:
        output =[]
        count= 0
        instance={}
        instance['tokens']=[]
        instance['langid']=[]
        instance['tweet']=""
        for line in fp.readlines():
            if re.search(r'^meta\b\t[0-9]',line):
                meta = line.replace('\n','').split('\t')
                # print(meta)
                instance['tweetid'] = int(meta[1].strip())
                instance['sentiment'] = meta[2].strip()
            elif line == '\n':
                count+=1
                output.append(instance)
                instance={}
                instance['tokens']=[]
                instance['langid']=[]
                instance['tweet']=""
                # print('found boundary')
            # print(line)
            else:
                parts = line.replace('\n','').split('\t')
                # print(parts)
                instance['tweet'] = "%s %s" % (instance['tweet'], parts[0])
                instance['tokens'].append(parts[0])
                instance['langid'].append(parts[1])
        print(output)
        with open('tweets_train.json', 'w') as fp:
            json.dump(output, fp)

def translate_from_spanish(string_to_translate, translator):
    # return translator.translate(string_to_translate, source_language='es')
    return translator.translate(string_to_translate, src='es',dest='en')

def translate_text_hacky():
    translator = Translator()
    #with open('spanglish_trial_release.json') as json_file:
    with open('translatedTweets.json') as json_file:
            data = json.load(json_file)
            output = []
            no_more_trans=False	
            for tweet in data:
                print(tweet['tweetid'])
                new_tweet = {}
                new_tweet['tweetid']=tweet['tweetid']
                new_tweet['tweet'] = tweet['tweet']
                new_tweet['tokens']=[]
                new_tweet['langid']=[]
                for i in range(0,len(tweet['tokens'])):
                    if tweet['langid'][i] == 'lang2' and not no_more_trans:
                        try:
                            translation = translate_from_spanish(tweet['tokens'][i],translator).text
                            new_tweet['langid'].append('lang1')    
                        except Exception:
                            print("couldn't translate moving on...")
                            no_more_trans=True	
                            new_tweet['tokens'].append(tweet['tokens'][i])
                            new_tweet['langid'].append(tweet['langid'][i])    
                    else:
                        new_tweet['tokens'].append(tweet['tokens'][i])
                        new_tweet['langid'].append(tweet['langid'][i])
                new_tweet['sentiment']=tweet['sentiment']
                output.append(new_tweet)
            with open('translatedTweets.json', 'w') as fp:
                json.dump(output, fp)

def translate_google(data):
    client = translate.Client()
    output = []
    for tweet in data:#[data[59]]:
        print(tweet['tweetid'])
        new_tweet = {}
        new_tweet['tweetid']=tweet['tweetid']
        new_tweet['tweet'] = tweet['tweet']
        new_tweet['tokens']=[]
        new_tweet['langid']=[]
        # print(tweet)
        for i in range(0,len(tweet['tokens'])):
            if tweet['langid'][i] == 'lang2':
                try:
                    translation = translate_from_spanish(tweet['tokens'][i],client)
                    # print(translation)
                    new_tweet['tokens'].append(translation['translatedText'])
                    new_tweet['langid'].append('lang1')
                except Exception:
                    new_tweet['tokens'].append(tweet['tokens'][i])
                    new_tweet['langid'].append(tweet['langid'][i])
            else:
                new_tweet['tokens'].append(tweet['tokens'][i])
                new_tweet['langid'].append(tweet['langid'][i])
        new_tweet['sentiment']=tweet['sentiment']

        # print(tweet)
        # print(new_tweet)
        time.sleep(5)
        output.append(new_tweet)
        with open('translatedTweets.json', 'w') as fp:
            json.dump(output, fp)

def spacy_pos_tagging(data):
    spacy.prefer_gpu()
    nlpEn = en_core_web_md.load()
    nlpEs = es_core_news_md.load()
    # nlpXX=xx_ent_wiki_sm.load()
    output = []
    for tweet in data[:2]:
        print(tweet['tweetid'])
        new_tweet = {}
        new_tweet['tweetid']=tweet['tweetid']
        new_tweet['tweet'] = tweet['tweet']
        new_tweet['tokens']=tweet['tokens']
        new_tweet['langid']=tweet['langid']
        new_tweet['pos']=[]
        new_tweet['sentiment']=tweet['sentiment']
        doc=nlpEn(tweet['tweet'])
        print([(w.text, w.pos_) for w in doc])

        for i in range(0,len(tweet['tokens'])):
            pass
        with open('posTweets.json', 'w') as fp:
            json.dump(output, fp)

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



if __name__ =='__main__':
    # with open('spanglish_trial_release.json') as json_file:
    # print("Loading Spacy")
    # nlpEn = en_core_web_md.load()
    # nlpEs = es_core_news_md.load()
    # print("Finished loading Spacy")
    with open('tweets_train.json') as json_file:
        data = json.load(json_file)
        cleaned = clean_tweets(data)
        stopped = remove_stop_words(cleaned)
        # translate_text_hacky()
        # spacy_pos_tagging(data)
    # parse_conll_to_json()
        with open('output_tweets.json', 'w') as fp:
            json.dump(stopped, fp)


