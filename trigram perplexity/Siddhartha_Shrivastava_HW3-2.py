#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter
from collections import defaultdict
import numpy as np
import re
import string 


# In[18]:


#Importing the training data and punctuating


# In[70]:


text = open('/Users/owner/Documents/CSCI-5832/HW3/hobbit-train.txt').read()
text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))


# In[71]:


# addinf <s> at both the end of the sentence 


# In[72]:


def Refined_text(ptext):
    ptext=ptext.lower()
    ptext=re.sub(r'\n|\n\n','',ptext)

    sentences=re.split(r'(?:(?<=\.)|(?<=\?)|(?<=\!)|(?<=\!”)|(?<=\?”))',ptext)  # We split the text on basis of special character

    textWithMarkers=""                                  # Adding <s> on both the terminals of a complete sentence
    for sentence in sentences:
        textWithMarkers +='<s> <s> '+sentence+' </s> '

    wordsToBeProcessed=textWithMarkers.split()
    tokens=[]
    for word in wordsToBeProcessed:
        for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
            if word.endswith(suffix):
                word = word[:-len(suffix)]
                break
        tokens.append(word)
    return tokens


# In[73]:


#word2index and default dict for the split text


# In[74]:


tokens = Refined_text(text)
rawFrequencies=Counter(tokens)
word2index = defaultdict(lambda: len(word2index))
UNK = word2index["<UNK>"]
[word2index[word] for word, freq in rawFrequencies.items() if (freq > 1) or (word=='<s>') or (word=='</s>') ];
word2index = defaultdict(lambda: UNK, word2index)


# In[75]:


#Making Unigrams, Bigrams and Trigrams


# In[76]:



unigrams = [word2index[word] for word in tokens]
unigramFreqs = Counter(unigrams)

bigrams = [ (word2index[tokens[i-1]], word2index[tokens[i]]) for i in range(1, len(tokens)) ]
bigramFreqs = Counter(bigrams)

trigrams = [ (word2index[tokens[i-2]], word2index[tokens[i-1]],word2index[tokens[i]]) for i in range(2, len(tokens)) ]
trigramFreqs = Counter(trigrams)


# In[77]:


#Making Unigrams, Bigrams and Trigrams probablity


# In[78]:


def UnigramProb(unigram):
    return (unigramFreqs[unigram])/(sum(unigramFreqs.values())+len(unigramFreqs))
def BigramProb(bigram):
    return (bigramFreqs[bigram])/(unigramFreqs[bigram[0]]+len(unigramFreqs))


def TrigramProb(trigram):
    res=(trigramFreqs[trigram])/(bigramFreqs[(trigram[0],trigram[1])]+len(unigramFreqs))
    res = 0.1*res+0.1*getBigramProb((trigram[1],trigram[2])) + 0.8*getUnigramProb(trigram[2])
    return res    


# In[79]:


# Finding out the lograthmic probablity


# In[80]:


def Log_Trigram_Sentence_Prob(text):
    words = preprocessText(text)
    trigramProb = np.sum([np.log(TrigramProb((word2index[words[i-2]], word2index[words[i-1]],word2index[words[i]]))) for i in range(2, len(words))])
    value=trigramProb
    return value


# In[81]:


# Perplexity of the Trigram sentence


# In[82]:


def Trigram_Sentence_Perplexity(text):
    words=text.split()
    prob = Log_Trigram_Sentence_Prob(text)
    return (np.exp(-prob/len(words)))


# In[83]:


# Using the test data


# In[84]:


test_data=[]
with open('/Users/owner/Documents/CSCI-5832/HW3/hw-test.txt',"r+",encoding="utf8") as fp:
    for line in fp:
        test_data.append(line)


# In[85]:


test_data


# In[86]:


for i in range(0,len(test_data)):
        text = test_data[i]
        print("%d , Log probability=%f"%(i+1,Log_Trigram_Sentence_Prob(text)))
        print("%d , perplexity=%f "%(i+1, Trigram_Sentence_Perplexity(text)))
        print('----------------------------------')


# In[ ]:




