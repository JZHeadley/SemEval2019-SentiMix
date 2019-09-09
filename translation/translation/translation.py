from googletrans import Translator
# from google.cloud import translate_v3beta1 as translate

import json
import time

import spacy
#import en_core_web_md
#import es_core_news_md
#import xx_ent_wiki_sm

def translate_from_spanish(string_to_translate, translator):
    # return translator.translate(string_to_translate, source_language='es')
    return translator.translate(string_to_translate, src='es',dest='en')
def translate_text_hacky():
    translator = Translator()
    with open('spanglish_trial_release.json') as json_file:
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
                # print(tweet['tokens'])
                # print(tweet['langid'])
                # print(tweet)
                for i in range(0,len(tweet['tokens'])):
                    if tweet['langid'][i] == 'lang2' and not no_more_trans:
                        try:
                            translation = translate_from_spanish(tweet['tokens'][i],translator).text
                            new_tweet['langid'].append('lang1')    
                        except Exception:
                            print("couldn't translate moving on...")
                            no_more_trans=True
                            # break
                            new_tweet['tokens'].append(tweet['tokens'][i])
                            new_tweet['langid'].append(tweet['langid'][i])    
                    else:
                        new_tweet['tokens'].append(tweet['tokens'][i])
                        new_tweet['langid'].append(tweet['langid'][i])
                new_tweet['sentiment']=tweet['sentiment']
                # if no_more_trans:
                #     break
                # print(tweet)
                # print(new_tweet)
                if not no_more_trans:
                    time.sleep(2)
                output.append(new_tweet)
            with open('translatedTweets.json', 'w') as fp:
                json.dump(output, fp)

def translate_google():
    client = translate.Client()
    with open('spanglish_trial_release.json') as json_file:
        data = json.load(json_file)
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

def spacy_pos_tagging():
    spacy.prefer_gpu()
    nlpEn = en_core_web_md.load()
    # nlpEs = es_core_news_md.load()
    # nlpXX=xx_ent_wiki_sm.load()
    with open('spanglish_trial_release.json') as json_file:
        data = json.load(json_file)
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
            # print(tweet)
            doc=nlpEn(tweet['tweet'])
            print([(w.text, w.pos_) for w in doc])

            for i in range(0,len(tweet['tokens'])):
                pass
        with open('posTweets.json', 'w') as fp:
            json.dump(output, fp)
if __name__ =='__main__':
    translate_text_hacky()
    # spacy_pos_tagging()
