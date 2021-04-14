#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import json
import pickle
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import nltk
from nltk.stem import WordNetLemmatizer


# In[2]:


lemmatizer = WordNetLemmatizer()
a = json.loads(open('jsonData.json').read())


# In[3]:


words = []
classes = []
documents = []
ignore_letters = ['!','?','.',',']


# In[4]:


for intent in a['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


# In[5]:


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))


# In[6]:


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


# In[7]:


training = []
empty_op = [0] * len(classes)


# In[8]:


for document in documents:
    bag=[]
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    op_rows = list(empty_op)
    op_rows[classes.index(document[1])] = 1
    training.append([bag, op_rows])


# In[9]:


random.shuffle(training)

training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])


# In[10]:


model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(train_y[0]), activation='softmax'))


sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
print('done')


# In[ ]:





# In[ ]:




