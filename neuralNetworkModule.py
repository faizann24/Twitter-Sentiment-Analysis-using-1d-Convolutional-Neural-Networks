import sys
import os
import numpy as np
import random
import pandas as pd
from nltk.corpus import words
from keras.preprocessing import text
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten,Bidirectional,Input,Reshape,Masking,Embedding,Conv1D, MaxPooling1D
plt.style.use('seaborn-whitegrid')

'''
DEFINE PARAMETERS
'''
vocabularyThreshold = 25	#words having count less than this threshold will not be in our vocabulary
maxWordsInATweet = 35	#number of max words to keep in a tweet
minWordsInATweet = 5	#tweets containing words less than this value will be ignored while training

'''
FUNCTIONS
'''
def cleanTweet(tweet):
	initialCleanedTweet = tweet.split(" ")
	initialCleanedTweet = [w for w in initialCleanedTweet if "@" not in w and "#" not in w and "/" not in w and w.lower() != 'rt']
	initialCleanedTweet = ''.join(str(e) + " " for e in initialCleanedTweet)
	words = text.text_to_word_sequence(initialCleanedTweet, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
	return words

def loadData(filepath):
	#load tweets data
	#columns -> sentiment,tweetid,date,query,user,actualtweet
	trainData = pd.read_csv(filepath,",")
	tweets = list(trainData["actualtweet"])
	tweetsSentiment = list(trainData["sentiment"])
	return tweets,tweetsSentiment


'''
DATA LOADING
'''
filepath = "data/train.csv"
tweets,tweetsSentiment = loadData(filepath)
print('-- Raw data loaded')


'''
CREATING WORD INDEXES
'''
#creating word indexes
errors = 0
wordsDict = {}
for i in range(0,len(tweets)):
	try:
		tweet = tweets[i]
		#clean tweet and convert into a list of words
		words = cleanTweet(tweet)
		for w in words:
			if w not in wordsDict:
				wordsDict[w] = 1
			else:
				wordsDict[w] = wordsDict[w] + 1
	except Exception as e:
		errors = errors + 1

print('-- Words indexes created')
print('-- Errors in creating word indexes',errors)

'''
CREATING VOCABULARY DICTIONARY
'''
#dictionary containing our final vocabulary words
finalWordsDict = {}
currentWordCount = 1
for word in wordsDict:
	if wordsDict[word] > vocabularyThreshold:
		finalWordsDict[word] = currentWordCount
		currentWordCount = currentWordCount + 1

print('-- Final and Initial Vocabulary sizes are',len(finalWordsDict),len(wordsDict))
#saving the vocabulary dictionary for testing purposes
np.save('vocabulary_dictionary/vocab_dict.npy',finalWordsDict)


'''
DATA PREPARATION FOR MODEL
'''
#generating features and labels from data
features = []
labels = []
for i in range(0,len(tweets)):
	tweet = tweets[i]
	words = cleanTweet(tweet)
	label = int(tweetsSentiment[i])
	tempFeatures = []
	wordsFound = len([w for w in words if w in finalWordsDict])

	#if words are less than minimum number of tweets to be in a tweet, skip this tweet
	if wordsFound <= minWordsInATweet:
		continue

	for w in words:
		if w in finalWordsDict:
			#adding index of the word from the vocabulary. This index will later be converted into an embedding
			tempFeatures.append(finalWordsDict[w])
		else:
			tempFeatures.append(0)

	#keeping feature vectors to a constant size
	if len(tempFeatures) >= maxWordsInATweet:
		tempFeatures = tempFeatures[:maxWordsInATweet]
	while len(tempFeatures) < maxWordsInATweet:
		tempFeatures.append(0)

	features.append(tempFeatures)
	if label == 0:
		labels.append([1,0])
	else:
		labels.append([0,1])

print('-- Labels distribution of positive and negative labels is',labels.count([0,1]),labels.count([1,0]))
features = np.array(features)
labels = np.array(labels)
print('-- Data shape is',features.shape,labels.shape)

'''
MODEL ARCHITECTURE
'''
model = Sequential()

#embedding layer to convert word indexes into word embeddings
model.add(Embedding(input_dim=len(finalWordsDict)+1,output_dim=64,input_shape=features.shape[1:]))

#1d convolutional layers
model.add(Conv1D(128,kernel_size=5,strides=1,padding='same',kernel_initializer='random_uniform',activation='relu'))
model.add(Conv1D(128,kernel_size=4,strides=1,padding='same',kernel_initializer='random_uniform',activation='relu'))
model.add(Conv1D(128,kernel_size=3,strides=1,padding='same',kernel_initializer='random_uniform',activation='relu'))

#dropout for reducing overfitting
model.add(Dropout(0.5))
model.add(Flatten())

#final dense layer
model.add(Dense(128))
model.add(Dropout(0.5))

#output layer
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


'''
MODEL TRAINING
'''
epochsToRun = 10
batchSize = 100
for i in range(0,epochsToRun):
	#training the model
	model.fit(features,labels,verbose=1,batch_size=batchSize,epochs=1,validation_split=0.10,shuffle=True)

	#saving the checkpoint
	model.save('model_checkpoints/model_checkpoint_'+str(i)+'.h5')



