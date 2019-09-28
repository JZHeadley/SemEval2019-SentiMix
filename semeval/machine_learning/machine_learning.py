# these models and implentations came from https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
	

# TODO change the part 1 portion
def machine_learning(x_train, x_test, y_train, y_test):
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    # data split validation for part 1
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        #NOTE random_State is a seed. used for debugging and comparing performance of different ml models
        kfold = model_selection.KFold(n_splits=10, random_state=42) 
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # cross fold validation for part 2
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


# TODO play with all hyper params
# sklearn models
# linear models
def logisticRegression(x_train, x_test, y_train, y_test):
    print("LogisticRegression")
    # Make predictions on validation dataset
    # NOTE for solver param: For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; 
    # NOTE solver = solver : str, {‘newton-cg’, ‘lbfgs’, ‘sag’, ‘saga’}
    clf = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=5000)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    # TODO should I just add the metrics stuff here?
    return predictions


def linearDiscriminantAnalysis(x_train, x_test, y_train, y_test):
    print("LinearDiscriminantAnalysis")
    # Make predictions on validation dataset
    # NOTE solver = solver : svd, lsqr, eigen
    lda = LinearDiscriminantAnalysis(solver='eigen')
    lda.fit(x_train, y_train)
    predictions = lda.predict(x_test)
    # TODO should I just add the metrics stuff here?
    return predictions


# non-linear models
def knn(x_train, x_test, y_train, y_test):
    print("knn")
    # Make predictions on validation dataset
    k = 15
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    # TODO should I just add the metrics stuff here?
    return predictions


def decisionTreeClassifier(x_train, x_test, y_train, y_test):
    print("DecisionTreeClassifier")
    # Make predictions on validation dataset
    # TODO theres lots of hyper params to try
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    predictions = dtc.predict(x_test)
    # TODO should I just add the metrics stuff here?
    return predictions


def gaussianNB(x_train, x_test, y_train, y_test):
    print("GaussianNB")
    # Make predictions on validation dataset
    # TODO theres lots of hyper params to try
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    predictions = gnb.predict(x_test)
    # TODO should I just add the metrics stuff here?
    return predictions


def supportVectorClassification(x_train, x_test, y_train, y_test):
    print("Support Vector Classification")
    # Make predictions on validation dataset
    # TODO theres lots of hyper params to try
    svc = SVC(gamma='auto')
    svc.fit(x_train, y_train)
    predictions = svc.predict(x_test)
    # TODO should I just add the metrics stuff here?
    return predictions