TRAINING
- The model has already been trained on our data set. You do not need to train it again.

PREDICTION SERVICE
- To predict sentiment for tweets, put them in a text file line by line and run the following script. In case of a single tweet, just put it in a line in a text file and run the script below.

python predictSentiment.py inputFile.txt outputFile.txt

inputFile.txt -> This will contain tweets, one tweet per line.
outputFile.txt -> The classifier will predict sentiment for tweets and will save them in this file with headers = tweet, predicted sentiment, probability of predicted sentiment
