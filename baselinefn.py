import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

"""
baselineNB: function for Naive Bayes Baseline

:param x_train: training data
:param y_train: training labels
:param x_test: testing data
:param y_test: testing labels
:param dist: set distribution
             0: Bernoulli
             1: Gaussian
             2: Multinomial


"""

def baselineNB(x_train, y_train, x_test, y_test, dist):
    # Generate Model

    if dist == 0:
        model = BernoulliNB()
        print("Running Bernoulli Naive Bayes \n")
    elif dist == 1:
        model = GaussianNB()
        print("Running Gaussian Naive Bayes \n")
    else:
        model = MultinomialNB()
        print("Running Multinomial Naive Bayes \n")

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    train_acc = (metrics.accuracy_score(y_test, y_pred)) * 100
    print("Training Accuracy:", format(train_acc, '.3f'))

    # Report Test Accuracy, Classification Report, and Confusion Matrix

    test_acc = (model.score(x_test, y_test)) * 100
    print("Test Accuracy:", format(test_acc, ".3f"))

    print("\nClassification Report")
    report = metrics.classification_report(y_test, y_pred)
    print(report)

    cm = confusion_matrix(y_pred, y_test)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig("Matrix/Confusion_matrix_NB")
    print("Confusion Matrix saved")

    return ()

def baselineDT():

    return()
