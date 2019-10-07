from googletrans import Translator
# from google.cloud import translate_v3beta1 as translate

import time

import spacy
# Zephyr Trying to translate things from spanish to english
def translate_from_spanish(string_to_translate, translator):
    # return translator.translate(string_to_translate, source_language='es')
    return translator.translate(string_to_translate, src='es',dest='en')

# Zephyr a hacky version of translating
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

# Zephyr A better way of translating using official google client. 
# Unfortunately this way would cost money so is not an option
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

