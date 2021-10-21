import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier  

inno = 46
if inno == 1:
    X,y = mglearn.datasets.make_forge()
    mglearn.discrete_scatter(X[:,0],X[:,1],y)
    plt.legend(["Class 0", "Class 1"], loc=4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    print("X.shape: {}".format(X.shape))
    plt.show()
if inno == 2:
    X,y = mglearn.datasets.make_wave(n_samples=40)
    plt.plot(X,y,"o")
    plt.ylim(-3,3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.show()
if inno ==3:
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    print("cancer.key(): \n{}".format(cancer.keys()))
          
    print("Shape of cancer data: {}".format(cancer.data.shape))
    
    print("Sample counts per class: \n{}".format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
    
    print("Feature names:\n{}".format(cancer.feature_names))
if inno == 7:
    from sklearn.datasets import load_boston
    boston = load_boston()
    print("Data shape: {}".format(boston.data.shape))
    
    X, y = mglearn.datasets.load_extended_boston()
    print("X.shape: {}".format(X.shape))
    
    mglearn.plots.plot_knn_classification(n_neighbors=3)
    plt.show()
if inno == 11:
    from sklearn.model_selection import train_test_split
    X, y = mglearn.datasets.make_forge()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    
    clf.fit(X_train, y_train)
    
    print("Test set prediction: {}".format(clf.predict(X_test)))
    
    print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
    
    fig, axes = plt.subplots(1,3,figsize=(10,3))

    for n_neighbors, ax in zip([1,3,9], axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
        ax.set_title("{} neighbor(s)".format(n_neighbors))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
    axes[0].legend(loc=3)
    plt.show()
if inno == 17:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=66)
        
    training_accuracy = []
    test_accuracy = []
    neighbors_settings = range(1,11)
    
    for n_neighbors in neighbors_settings:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        
        training_accuracy.append(clf.score(X_train, y_train))
        
        test_accuracy.append(clf.score(X_test, y_test))
        
    plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()
if inno == 18:
    mglearn.plots.plot_knn_regression(n_neighbors=3)
    plt.show()
if inno == 20:
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    X, y = mglearn.datasets.make_wave(n_samples=40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    reg = KNeighborsRegressor(n_neighbors=3)
    reg.fit(X_train, y_train)
    
    print("Test set predictions: \n{}".format(reg.predict(X_test)))
    
    print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    line = np.linspace(-3, 3, 1000).reshape(-1,1)
    for n_neighbors, ax in zip([1,3,9], axes):
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)
        ax.plot(line, reg.predict(line))
        ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
        ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
        
        ax.set_title(
            "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
                n_neighbors, reg.score(X_train, y_train),
                reg.score(X_test, y_test)))
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
    axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")
    plt.show()
if inno == 24:   
    mglearn.plots.plot_linear_regression_wave()
    plt.show()
if inno == 25:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    X, y = mglearn.datasets.make_wave(n_samples=60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    lr = LinearRegression().fit(X_train, y_train)
    
    print("lr.coef_: {}".format(lr.coef_))
    print("lr.intercept_: {}".format(lr.intercept_))
    
    print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
if inno == 28:  # 2.3.3 線形モデル
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr = LinearRegression().fit(X_train, y_train)
    
    print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
if inno == 30:  # 2.3.3.3リッジ回帰
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge

    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    lr = LinearRegression().fit(X_train, y_train)

    ridge = Ridge().fit(X_train, y_train)
    print("Alpha=1")
    print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
   
    ridge10 = Ridge(alpha=10).fit(X_train, y_train)
    print("Alpha=10")
    print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
    print("Alpha=0.1")
    print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
    print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
    
    plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
    plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
    plt.plot(ridge01.coef_, 'v', label="Ridge alpha=01")
    
    plt.plot(lr.coef_, 'o', label="LinearRegression")
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.hlines(0, 0, len(lr.coef_))
    plt.ylim(-25,25)
    plt.legend()
    plt.show()
 
    mglearn.plots.plot_ridge_n_samples()
    plt.show()
if inno == 35:  # 2.3.3.4 Lasso    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso

    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    lasso = Lasso().fit(X_train, y_train)
    print("lasso")
    print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
    print("Training set score: {:.2f}".format(lasso.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
   
    lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
    print("lasso001")
    print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
    print("Training set score: {:.2f}".format(lasso001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))
    
    lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
    print("lasso00001")
    print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
    print("Training set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
    print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))
    
    plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
    plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
    plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

    plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
    plt.legend(ncol=2, loc=(0, 1.05))
    plt.ylim(-25, 25)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.show()
if inno == 39:  # 2.3.3.5 クラス分類のための線形モデル
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    
    X, y = mglearn.datasets.make_forge()
    fig, axes = plt.subplots(1,2, figsize=(10,3))
    
    for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
        clf = model.fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                        ax=ax, alpha=.7)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{}".format(clf.__class__.__name__))
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    axes[0].legend()
    plt.show()
    
    mglearn.plots.plot_linear_svc_regularization()
    plt.show()
if inno == 41:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=42)
    logreg = LogisticRegression(max_iter=10000).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
    print("Teshst set score: {:.3f}".format(logreg.score(X_test, y_test)))
    
    print("\nlogreg100")
    logreg100 = LogisticRegression(C=100, max_iter=10000).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
    print("Teshst set score: {:.3f}".format(logreg100.score(X_test, y_test)))  
    
    print("\nlogreg001")
    logreg001 = LogisticRegression(C=0.001, max_iter=10000).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
    print("Teshst set score: {:.3f}".format(logreg001.score(X_test, y_test)))
    
    plt.plot(logreg.coef_.T, 'o', label="C=1")
    plt.plot(logreg100.coef_.T, '^', label="C=100")
    plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.ylim(-5, 5)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.show()
            
    for C, marker in zip([0.001, 1, 100], ['o', '^','v']):
        lr_l1 = LogisticRegression(C=C, penalty="l1", solver='liblinear').fit(X_train, y_train)
        print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
            C, lr_l1.score(X_train, y_train)))
        print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
        plt.plot(lr_l1.coef_.T, marker, label="C{:.3f}".format(C))
        
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    
    plt.ylim(-5, 5)
    plt.legend(loc=3)
    plt.show()
if inno == 46:  # 2.3.3.6 線形モデルによるタクラス分類
    from sklearn.datasets import make_blobs
    from sklearn.svm import LinearSVC
    
    X, y = make_blobs(random_state=42)
    mglearn.discrete_scatter(X[:,0], X[:,1], y)
    plt.xlabel("feature 0")
    plt.ylabel("feature 1")
    plt.legend(["Class 0", "Class 1", "Class 2"])
#    plt.show()
    
    linear_svm = LinearSVC().fit(X, y)
    print("Coefficient shape: ", linear_svm.coef_.shape)
    print("Intercept shape: ", linear_svm.intercept_.shape)
    
    mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    line = np.linspace(-15, 15)
    for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
        plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.ylim(-10, 15)
    plt.xlim(-10, 8)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 
                'Line class 1', 'Line class 2'], loc=(1.01, 0.3))
    plt.show()
    