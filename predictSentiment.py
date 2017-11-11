import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing import text

'''
Usage: python predictSentiment.py textfile.txt outputFile.txt

- Textfile.txt should contain tweets line by line
- After the code has run, outputfile.txt will contain all tweets with their sentiment labels
''' 

'''
MODEL AND VOCABULARY DICTIONARY LOADING
'''
#loading model and vocabulary dictionary and 
model = load_model('model_checkpoints/model_checkpoint_1.h5')
wordsDictionary = np.load('vocabulary_dictionary/vocab_dict.npy').item()
print('-- Vocabulary length is',len(wordsDictionary))


'''
PARAMETERS
'''
#defining parameters. Please don't change this
maxWordsInATweet = 35	#number of max words to keep in a tweet
minWordsInATweet = 5	#tweets containing words less than this value will be ignored while training


'''
LOADING DATA
'''
#file paths
inputFile = sys.argv[1]
outFile = sys.argv[2]

#opening files
outputFile = open(outFile,"w")
outputFile.write("Sentence,Label,LabelProbability\n")
data = open(inputFile,"r").readlines()


'''
PROCESSING AND LABELING
'''
#processing data and predicting stuff
for row in data:
	row = row.strip("\n")
	initialWords = row.split(" ")
	#cleaning tweet
	initialWords = [w for w in initialWords if "@" not in w and "#" not in w and "/" not in w and w.lower() != 'rt']
	initialWords = ''.join(str(e) + " " for e in initialWords)
	
	#tokenizing
	words = text.text_to_word_sequence(initialWords, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
	wordsFound = [w for w in words if w in wordsDictionary]
	
	#generating feature vector
	featuresList = []
	for w in words:
		if w in wordsDictionary:
			wordId = wordsDictionary[w]
			featuresList.append(wordId)
		else:
			featuresList.append(0)

	#making the size of the feature vector consistent
	if len(featuresList) >= maxWordsInATweet:
		featuresList = featuresList[:maxWordsInATweet]

	while len(featuresList) < maxWordsInATweet:
		featuresList.append(0)

	#wrapping the vector in a list
	featuresList = [featuresList]
	featuresList = np.array(featuresList)

	#predicting label for the tweet
	prediction = model.predict(featuresList)

	#saving in file
	if prediction[0][0] > prediction[0][1]:
		outputFile.write('"'+str(row).replace('"',"")+'"'+","+str('"Negative"')+","+str(prediction[0][0])+"\n")
	else:
		outputFile.write('"'+str(row).replace('"',"")+'"'+","+str('"Positive"')+","+str(prediction[0][1])+"\n")

#closing file
outputFile.close()