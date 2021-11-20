from typing import final
from nltk.tokenize.destructive import NLTKWordTokenizer
import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f=open('chatbot.txt','r',errors='ignore')
raw_doc=f.read()
raw_doc=raw_doc.lower()#converts the whole text to lowercase
nltk.download('punkt') #using the Punkt tokenizer
nltk.download('wordnet')
sent_tokens=nltk.sent_tokenize(raw_doc)
word_tokens=nltk.word_tokenize(raw_doc)

word_tokens[:2]

#text preprocessing
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Greeting input and output responses
Greet_INPUTS = ("hello","hi","greetings","what's up","hey","namastay","hola","garcias")
GREET_RESPONSES =["Hi","Hey","Hello there","hello","Hi, I'm your personal chatbot.","Namstay, What can, I help you with.."]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in Greet_INPUTS:
            return random.choice(GREET_RESPONSES)

#response
def response(user_response):
    robo1_response= ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo1_response=robo1_response+"I'm sorry! I don't unserstand you."
        return robo1_response
    else:
        robo1_response=robo1_response+sent_tokens[idx]
        return robo1_response

#start and end
def get_res(msg):
    
    word_tokens=nltk.word_tokenize(raw_doc)
    user_response=msg.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you'):
            flag=False
            return "Thank you for talking to me."
        else:
            if(greet(user_response)!=None):
                return greet(user_response)
            else:
                sent_tokens.append(user_response)
                word_tokens=word_tokens+nltk.word_tokenize(user_response)
                final_words=list(set(word_tokens))
                return response(user_response)
                
    else:
        flag=False
        return "Goodbye!"
