import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier
#import graphviz 
   
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

inno = 83
if inno == 74:       # 2.3.7 カーネル法を用いたサポートベクタマシン
    from sklearn.datasets import make_blobs
    
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()  
if inno == 76:
    from sklearn.svm import LinearSVC
    from sklearn.datasets import make_blobs
    from mpl_toolkits.mplot3d import Axes3D, axes3d
    
    X, y = make_blobs(centers=4, random_state=8)
    X_new = np.hstack([X, X[:, 1:]**2])
    y = y % 2

    linear_svm_3d = LinearSVC().fit(X_new, y)
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
    
    figure = plt.figure()
    ax = Axes3D(figure, elev=-152, azim=-26)
    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
    
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (coef[0] * XX + coef[1] * YY + intercept)/-coef[2]
    ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
    mask = y == 0
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
               cmap=mglearn.cm2, s=60)
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
               cmap=mglearn.cm2, s=60)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.set_zlabel("Feature 1**2")
    plt.show()  
if inno == 78:
    from sklearn.svm import LinearSVC
    from sklearn.datasets import make_blobs
    from mpl_toolkits.mplot3d import Axes3D, axes3d
    
    X, y = make_blobs(centers=4, random_state=8)
    X_new = np.hstack([X, X[:, 1:]**2])
    y = y % 2

    linear_svm_3d = LinearSVC().fit(X_new, y)

    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
    
    XX, YY = np.meshgrid(xx, yy)
    ZZ = YY ** 2
    
    dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
                 cmap=mglearn.cm2, alpha=0.5)
    mglearn.discrete_scatter(X[:,0], X[:, 1], y)
    
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
if inno == 79:
    from sklearn.svm import SVC
    X, y = mglearn.tools.make_handcrafted_dataset()
    svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
    mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    sv = svm.support_vectors_
    sv_labels = svm.dual_coef_.ravel() > 0
    mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
if inno == 80:
    from sklearn.svm import SVC
    
    X, y = mglearn.tools.make_handcrafted_dataset()
    svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
    mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, C in zip(axes, [-1, 0, 3]):
        for a, gamma in zip (ax, range(-1, 2)):
            mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
    axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],
                      ncol=4, loc=(.9, 1.2))
    plt.show()
if inno == 81:
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()    
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    
    svc = SVC()
    svc.fit(X_train, y_train)
    
    print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
    print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
    
    plt.plot(X_train.min(axis=0), 'o', label="min")
    plt.plot(X_train.max(axis=0), '^', label="max")
    plt.legend(loc=4)
    plt.xlabel("Feature index")
    plt.ylabel("Feature magnitude")
    plt.yscale("log")
    plt.show()
if inno == 83:
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()    
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
   
    min_on_training = X_train.min(axis=0)
    range_on_training = (X_train - min_on_training).max(axis=0)
    
    X_train_scaled = (X_train - min_on_training) / range_on_training
    print("minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
    print("maximum for each feature\n{}".format(X_train_scaled.max(axis=0)))
    
    X_test_scaled = (X_test - min_on_training) / range_on_training

    svc = SVC(C=1000)
    svc.fit(X_train_scaled, y_train)
    
    print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))
    print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
    
    exit(0)
    plt.plot(X_train.min(axis=0), 'o', label="min")
    plt.plot(X_train.max(axis=0), '^', label="max")
    plt.legend(loc=4)
    plt.xlabel("Feature index")
    plt.ylabel("Feature magnitude")
    plt.yscale("log")
    plt.show()
    
    
    