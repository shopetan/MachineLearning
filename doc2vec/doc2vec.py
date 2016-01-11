#coding: UTF-8

import os 
import re
import nltk
import csv

from gensim import corpora, models, similarities,utils
from collections import defaultdict

HAMS = './data/easy_ham'
SPAMS = './data/spam_2'
TRAINING = 1000
TEST = 500
count = 0
first_empty = True

hams = os.listdir(HAMS)
spams = os.listdir(SPAMS)

hams_sentences = [[]]
spams_sentences = [[]]
tmp_list = []


def isStringEmpty(s):
    if s  in '\n':
        return True
    else:
        return False

def deleteAsciiList(list):
    for word in list:
        try:
            word.encode('ascii', 'strict')
            word.encode('utf-8').decode('utf-8')
        except UnicodeDecodeError:
            list.remove(word)


def deleteAsciiWord(word):
    try:
        word.encode('ascii', 'strict')
        word.encode('utf-8').decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False

    
def deleteWord(word,list):
    while word in list:
        list.remove(word)

        
def containDeleteWord(word,list):
    containList = []
    
    for contain_word in list:
        if re.search(word,contain_word):
            containList.append(contain_word)

    for contain_word in containList:
        list.remove(contain_word)
        

def containReplaceWord(word,list):
    containList = []
    
    for contain_word in list:

        if contain_word == "\(":
            if re_check(word,contain_word):
                list.remove(contain_word)
                list.append(contain_word[1:])
                
        elif contain_word == "\)":
            if re_check(word,contain_word):
                list.remove(contain_word)
                #list.append(contain_word[:-1])
                
        else:
            if re_check(word,contain_word):
                contain_word.strip()
                list.remove(contain_word)
            
    #print "===================="

def re_check(word,contain_word):
    try:
        return re.search(word,contain_word)
    except re.error:
        pass
    return False


# Get Spams sentences(1000)


os.chdir(HAMS)
for ham in hams:
    if(count < TRAINING):
        if(ham != None):
            f = open('%(ham)s' % locals(), 'r')
            for line in f:
                line.rstrip()
                if(isStringEmpty(line) & first_empty):
                    first_empty = False
                if(not first_empty):
                    tmp_list += line.split(" ")
                    
            first_empty = True
            f.close()

        deleteAsciiList(tmp_list)
        deleteWord("\n",tmp_list)
        deleteWord("|",tmp_list)
        deleteWord("{",tmp_list)
        deleteWord("}",tmp_list)
        deleteWord("",tmp_list)
        containDeleteWord("_",tmp_list)
        containDeleteWord("=",tmp_list)
        containDeleteWord("-",tmp_list)
        containDeleteWord(">",tmp_list)
        containDeleteWord("<",tmp_list)
        containDeleteWord("@",tmp_list)
        containDeleteWord("http",tmp_list)
        containDeleteWord("1",tmp_list)
        containDeleteWord("2",tmp_list)
        containDeleteWord("3",tmp_list)
        containDeleteWord("4",tmp_list)
        containDeleteWord("5",tmp_list)
        containDeleteWord("6",tmp_list)
        containDeleteWord("7",tmp_list)
        containDeleteWord("8",tmp_list)
        containDeleteWord("9",tmp_list)
        containDeleteWord("0",tmp_list)
        containDeleteWord(u"’",tmp_list)
        containReplaceWord("\n",tmp_list)
        containReplaceWord("\(",tmp_list)
        containReplaceWord("\)",tmp_list)

        norn = []
        for word in tmp_list:
            if not deleteAsciiWord(word):
                tmp_list.remove(word)
            else:
                word = nltk.word_tokenize(word)
                tokens = nltk.pos_tag(word)
                try:
                    if tokens[0][1] == "NN":
                        norn.append(tokens[0][0])
                except IndexError:
                    pass

        print count

        hams_sentences.append(norn)
        tmp_list = []
        count += 1
count = 0

print hams_sentences

f = open('hams_sentences1000.csv', 'w')
writer = csv.writer(f)
for topic in hams_sentences:
    for line in topic:
        writer.writerow(line)
f.close()


# 辞書追加，コーパス作成とLSIモデル生成

dictionary = corpora.Dictionary(hams_sentences)
print dictionary
dictionary.save('deerwester.dict')

corpus = [dictionary.doc2bow(text) for text in hams_sentences]
corpora.MmCorpus.serialize('deerwester.mm', corpus)

print corpus

lsi = models.LsiModel(corpus=corpus, id2word=dictionary, num_topics=200)
lsi.save('hams_lsi_norn_topics200.model')

for i in range(0, 200):
    print lsi.show_topic(i,topn = 5)


while True:
    query = raw_input('Enter your input word:')
    
    for i in range(0, 200):
        for j in range(0, 5):
            if query in lsi.show_topic(i,topn = 5)[j][0]:
                print lsi.show_topic(i,topn = 5)

