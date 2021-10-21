import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import graphviz 
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

inno = 118
if inno == 103: 
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_circles
    from sklearn.model_selection import train_test_split
    
    X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
    
    y_named = np.array(["blue", "red"])[y]
   
    X_train, X_test, y_train_named, y_test_named, y_train, y_test =\
        train_test_split(X, y_named, y, random_state=0)
    
    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train_named)
    
    print("X_test.shape: {}".format(X_test.shape))
    print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
    print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6]))
    
    print("Thresholded decision function:\n{}".format(
        gbrt.decision_function(X_test)>0))
    print("Prediction:\n{}".format(gbrt.predict(X_test)))
    
    greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)
    pred = gbrt.classes_[greater_zero]
    print("pred is equal to predictions: {}".format(np.all(pred == gbrt.predict(X_test))))

    decision_function = gbrt.decision_function(X_test)
    print("Decision function minimum: {:.2f} maximum: {:.2f}".format(
        np.min(decision_function), np.max(decision_function)))
        
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,
                                    fill=True, cm=mglearn.cm2)
    scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],
                                                alpha=.4, cm=mglearn.ReBl)
    
    for ax in axes:
        mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    cbar = plt.colorbar(scores_image, ax=axes.tolist())
    axes[0].legend(["Test class 0", "Test class 1", "Train class 0", "Train class 1"], ncol=4, loc=(.1, 1.1))
    plt.show()
    
    print("shape of probabilities: {}".format(gbrt.predict_proba(X_test).shape))
    
    # predict_probaの出力の最初の数行を見る
    print("Predicted probabilities:\n{}".format(
        gbrt.predict_proba(X_test[:6])))
if inno == 112: 
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_circles
    from sklearn.model_selection import train_test_split
    
    X, y = make_circles(noise=0.25, factor=0.5, random_state=1)
    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X, y)
    
    y_named = np.array(["blue", "red"])[y]
    X_train, X_test, y_train_named, y_test_named, y_train, y_test =\
        train_test_split(X, y_named, y, random_state=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,
                                    fill=True, cm=mglearn.cm2)
    scores_image = mglearn.tools.plot_2d_scores(
        gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')
    
    for ax in axes:
        mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    cbar = plt.colorbar(scores_image, ax=axes.tolist())
    axes[0].legend(["Test class 0", "Test class 1", "Train class 0", "Train class 1"], ncol=4, loc=(.1, 1.1))
    plt.show()
if inno == 113:
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X_train, X_test, y_train, y_test =\
        train_test_split(iris.data, iris.target, random_state=0)
    gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
    gbrt.fit(X_train, y_train)
    
    print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
    print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))
    
    print("Argmax of decision function:\n{}".format(
        np.argmax(gbrt.decision_function(X_test), axis=1)))
    print("Predictions:\n{}".format(gbrt.predict(X_test)))

    print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6]))
    print("Sums: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))
if inno == 118:
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    
    iris = load_iris()
    X_train, X_test, y_train, y_test =\
        train_test_split(iris.data, iris.target, random_state=0)
    gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
    gbrt.fit(X_train, y_train)
    
    print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
    print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))
    
    print("Argmax of decision function:\n{}".format(
        np.argmax(gbrt.decision_function(X_test), axis=1)))
    print("Predictions:\n{}".format(gbrt.predict(X_test)))

    print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6]))
    print("Sums: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))
    
    logreg = LogisticRegression()
    named_target = iris.target_names[y_train]
    logreg.fit(X_train, named_target)
    print()
    print("unique classes in training data: {}".format(logreg.classes_))
    print("predictions: {}".format(logreg.predict(X_test)[:10]))
    
    argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1)
    print("argmax of decision function: {}".format(argmax_dec_func[:10]))
    print("argmax combined with classes_: {}".format(
        logreg.classes_[argmax_dec_func][:10]))
            