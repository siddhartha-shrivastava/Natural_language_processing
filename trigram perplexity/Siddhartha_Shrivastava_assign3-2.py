#!/usr/bin/env python
# coding: utf-8

# In[14]:


from collections import Counter
from collections import defaultdict
import numpy as np


# In[15]:


text = open('/Users/owner/Documents/CSCI-5832/HW3/hobbit-train.txt')
tolkin_text = (text.read())


# In[23]:


words = tolkin_text.split()
rawfreqs = Counter(words)
rawfreqs


# In[18]:


word2index = defaultdict(lambda: len(word2index))
UNK = word2index["<UNK>"]


# In[19]:


word2index["<UNK>"]


# In[20]:


[word2index[word] for word, freq in rawfreqs.items() if freq > 1]


# In[21]:


word2index["Gandalf"]


# In[50]:


#Finding trigrams, bigrams and Unigrams Counter and Frequencies

#Unigrmas
unigrams = [word2index[word] for word in words]
unigramFreqs = Counter(unigrams)
#bigrams
bigrams = [ (word2index[words[i-1]], word2index[words[i]]) for i in range(1, len(words)) ]
bigramFreqs = Counter(bigrams)

#trigrams
trigrams = [(word2index[words[i-2]], word2index[words[i-1]],word2index[words[i]]) for i in range(2, len(words)) ]
trigramFreqs = Counter(trigrams)


# In[63]:


def UnigramProb(unigram):
    return float(unigramFreqs[unigram]+1)/(sum(unigramFreqs.values())+len(unigramFreqs))

def BigramProb(bigram):
    return float(bigramFreqs[bigram]+1)/(unigramFreqs[bigram[0]]+len(unigramFreqs))

def trigramProb(trigram):
    return float(trigramFreqs[trigram]+1)/(bigramFreqs[(trigram[0],trigram[1])]+len(unigramFreqs))


# In[70]:


def Trigram_Sentence_Log_Prob(text):
    words = text.split()
    #print((word2index[words[0]], word2index[words[1]]))
    print(BigramProb((word2index[words[0]], word2index[words[1]])))
    value = np.sum([np.log(BigramProb((word2index[words[0]], word2index[words[1]]))),np.log(UnigramProb(word2index[words[0]]))])
    log_trigramProb = np.sum([np.log(trigramProb((word2index[words[i-2]], word2index[words[i-1]],word2index[words[i]]))) for i in range(2, len(words))])
    value=value+log_trigramProb
    return value


# In[71]:


def Trigram_Sentence_Perplexity(text):
    words=text.split()
    prob=Trigram_Sentence_Log_Prob(text)
    return (np.exp(-prob/len(words)))


# In[72]:


test_data=[]
with open('/Users/owner/Documents/CSCI-5832/HW3/hw-test.txt') as fp:
    for line in fp:
        test_data.append(line)


# In[73]:


test_data


# In[75]:


with open('/Users/owner/Documents/CSCI-5832/HW3/HW3_test_reults.txt',"w") as results:
    for i in range(0,len(test_data)):
        text = test_data[i]
        perplexity = Trigram_Sentence_Perplexity(text)
        results.write("%f\n" % perplexity)


# In[ ]:




