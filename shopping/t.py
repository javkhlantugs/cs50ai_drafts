import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

month_to_int = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "June": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
}
TEST_SIZE = 0.4


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")
    
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    
def load_data(filename):


    raw_data = pd.read_csv(filename, header=0).values.tolist()

    #duplicate to evidence from raw_data with last item from every list removed
    evidence = [row[:-1] for row in raw_data]
    
    #duplicate to label from raw_data and change from boolean to int
    label = [int(row[-1]) for row in raw_data]

    #change evidence items to floats and integers
    for row in evidence:
        row[1] = float(row[1])
        row[3] = float(row[3])
        row[5] = float(row[5])
        row[6] = float(row[6])
        row[7] = float(row[7])
        row[8] = float(row[8])
        row[9] = float(row[9])
        row[10] = month_to_int[row[10]]
        row[-2]=1 if row[-2] == "Returning_Visitor" else 0
        row[-1] = 1 if row[-1] is True else 0

    return (evidence, label)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(evidence, labels)
    return neigh

def evaluate(labels, predictions):
    positive, negative, false_positive, false_negative, true_positive, true_negative, sensitivity, specificity = 0, 0, 0, 0, 0, 0, 0, 0

    for label in labels:
        if label == 1:
            positive += 1
        elif label == 0:
            negative += 1

    for label, prediction in zip(labels, predictions):
        if label == 1 and prediction == 1:
            true_positive += 1
        elif label == 0 and prediction == 0:
            true_negative += 1
        elif label == 1 and prediction == 0:
            false_negative += 1
        elif label == 0 and prediction == 1:
            false_positive += 1
            
    sensitivity = true_positive / positive
    specificity = true_negative / negative

    return (sensitivity, specificity)



if __name__ == "__main__":
    main()