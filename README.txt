TRAINING
- Start the neuralNetworkModule and run it. Before running it, please make two folders in the same directory named vocabulary_dictionary and model_checkpoints. After the training phase is done, you can use the model checkpoints to predict.

PREDICTION SERVICE
- To predict sentiment for tweets, put them in a text file line by line and run the following script. In case of a single tweet, just put it in a line in a text file and run the script below.

python predictSentiment.py inputFile.txt outputFile.txt

inputFile.txt -> This will contain tweets, one tweet per line.
outputFile.txt -> The classifier will predict sentiment for tweets and will save them in this file with headers = tweet, predicted sentiment, probability of predicted sentiment
