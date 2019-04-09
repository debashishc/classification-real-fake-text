#!/Users/dc/anaconda3/bin/python3
from sklearn.model_selection import test_train_split

# classifiers
from sklearn.naive_bayes import GaussianNB

# csv file name
CSV_name = ''

# need labelled data
train = list()
train_labels = list()
test = list()
test_labels = list()

# Initialize classifier
gnb = GaussianNB()

# Train classifier
model = gnb.fit(train, train_labels)

# Make predictions
preds = gnb.predict(test)

# Evaluate model
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(test_labels, preds))
print(accuracy_score(test_labels, preds))


if __name__ == "__main__":
    pass
