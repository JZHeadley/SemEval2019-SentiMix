# these models and implentations came from https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from alive_progress import alive_bar


# ALWIN - runs  Logistic Regression, Linear Discriminant Analysis, knn, Decision Tree Classifier, Gaussian NB, and Support Vector Classification with cross fold validation
# TODO change the part 1 portion
def machine_learning(X_train, X_test, Y_train, Y_test):
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))

    scoring = None # TODO change this, so its actaully usable

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
        kfold = model_selection.KFold(n_splits=10, random_state=4)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


# sklearn models
# linear models
# ALWIN - performs Logistic Regression on data
def logisticRegression(x_train, x_test, y_train, y_test):
    print("LogisticRegression")
    # Make predictions on validation dataset
    # list of possible params
    # NOTE for solver param: For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
    # NOTE solver = solver : str, {‘newton-cg’, ‘lbfgs’, ‘sag’, ‘saga’}
    clf = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=5000)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    return predictions


# ALWIN - performs Logistic Regression on data
def linearDiscriminantAnalysis(x_train, x_test, y_train, y_test):
    print("LinearDiscriminantAnalysis")
    # list of possible params:
    # NOTE solver = solver : svd, lsqr, eigen
    lda = LinearDiscriminantAnalysis(solver='eigen')
    lda.fit(x_train, y_train)
    predictions = lda.predict(x_test)
    return predictions


# non-linear models
# ALWIN - creates a knn model using the data
def knn(x_train, x_test, y_train, y_test):
    print("knn")
    k = 15
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    return predictions


# ALWIN - creates a Decision Tree Classifier model using the data
def decisionTreeClassifier(x_train, x_test, y_train, y_test):
    print("DecisionTreeClassifier")
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    predictions = dtc.predict(x_test)
    return predictions


# ALWIN - creates a Gaussian NB model using the data
def gaussianNB(x_train, x_test, y_train, y_test):
    print("GaussianNB")
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    predictions = gnb.predict(x_test)
    return predictions


# ALWIN - creates a Support Vector Classification model using the data
def supportVectorClassification(x_train, x_test, y_train, y_test):
    print("Support Vector Classification")
    svc = SVC(gamma='auto')
    svc.fit(x_train, y_train)
    predictions = svc.predict(x_test)
    return predictions
# KARUN - optimizes logistic regression model using the data
def logregOptimizer(x_train,y_train,x_test,y_test):
    # Create hyperparameter options
    tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty'=['l1', 'l2', 'elasticnet']}]
    scores = ['f1_weighted']
    # Create grid search using 5-fold cross validation
    clf = GridSearchCV(linear_model.LogisticRegression(), hyperparameters, cv=5, scoring='f1_weighted')
    clf.fit(x_train, y_train)
    print("Best parameters set found on development set: ", clf.best_params_)
	print("Grid scores on development set:\n")
	means = clf.cv_results_["mean_test_score"]
	stds = clf.cv_results_["std_test_score"]
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	print('\nDetailed classification report:\n')
	y_true, y_pred = y_test, clf.predict(x_test)
	print(classification_report(y_true, y_pred))
	return clf.best_params_
# KARUN - optimizes Support Vector Classification model using the data
def svcOptimizer(x_train,y_train,x_test,y_test):
	#with alive_bar(len(scores)) as bar:
		#bar()
	print("Start of Optimzations: ")
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.0001,0.001, 0.01, 0.1],'C': [1,10,100,1000]}]
	scores = ['f1_weighted']
	clf = GridSearchCV(SVC(), tuned_parameters, cv=2,scoring='f1_weighted')
	clf.fit(x_train, y_train)
	print("Best parameters set found on development set: ", clf.best_params_)
	print("Grid scores on development set:\n")
	means = clf.cv_results_["mean_test_score"]
	stds = clf.cv_results_["std_test_score"]

	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

	print('\nDetailed classification report:\n')
	y_true, y_pred = y_test, clf.predict(x_test)
	print(classification_report(y_true, y_pred))
	return clf.best_params_
