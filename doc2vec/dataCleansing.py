import sys
import os 
import re
#from gensim.models.doc2vec import *

HAMS = './data/easy_ham'
SPAMS = './data/spam'
txt = '.txt'

hams = os.listdir(HAMS)
spams = os.listdir(SPAMS)

os.chdir('./data/spam')

for spam in spams:
    if(spam != None):
        if(txt not in spam):
            os.rename('%(spam)s' % locals() ,'%(spam)s' % locals() + '%(txt)s' % locals())
            
