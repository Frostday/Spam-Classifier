import os
import io
import numpy
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':  # ignore header and only take body
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)


data = DataFrame({'message': [], 'class': []})

data = data.append(dataFrameFromDirectory('emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('emails/ham', 'ham'))

print(data.head())

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)
#  splits message into words and counts how many time each word occurs
#  counts contains how many times each word occurs in an email
targets = data['class'].values
#  targets contains the classification data for each email

classifier = MultinomialNB()
print(classifier.fit(counts, targets))

print("Enter email content: ")
x = input()
y = list()
y.append(x)
y_counts = vectorizer.transform(y)
prediction = classifier.predict(y_counts)

if prediction == 'ham':
    print("not spam")
else:
    print("spam")