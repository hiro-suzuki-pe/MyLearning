import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import graphviz 
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier
   
inno = 112
if inno == 87:       # 2.3.8 Neural network (Deep learning)
    from sklearn.tree import export_graphviz

    display(mglearn.plots.plot_logistic_regression_graph())
    display(mglearn.plots.plot_single_hidden_layer_graph())
if inno == 89: 
    line = np.linspace(-3, 3, 100)
    plt.plot(line, np.tanh(line), label="tanh")
    plt.plot(line, np.maximum(line, 0), label="relu")
    plt.legend(loc="best")
    plt.xlabel("x")
    plt.ylabel("relu(x), tanh(x)")
    plt.show()
if inno == 91: 
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    
    mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X_train, y_train)
    mglearn.plots.plot_2d_classification(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
if inno == 92: 
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    
    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_classification(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
if inno == 93: 
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    mlp = MLPClassifier(solver='lbfgs', activation="tanh",
                        random_state=0, hidden_layer_sizes=[10, 10])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_classification(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
if inno == 94: 
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    
    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
if inno == 95: 
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for axx, n_hidden_nodes in zip(axes, [10, 100]):
        for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
            mlp = MLPClassifier(solver='lbfgs', random_state=0,
                                hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                                alpha=alpha, max_iter=1000)
            mlp.fit(X_train, y_train)
            mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
            mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
            ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(
                n_hidden_nodes, n_hidden_nodes, alpha)) 
    plt.show()
if inno == 96: 
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, ax in enumerate(axes.ravel()):
            mlp = MLPClassifier(solver='lbfgs', random_state=i,
                                hidden_layer_sizes=[100, 100])
            mlp.fit(X_train, y_train)
            mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
            mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
    plt.show()
if inno == 97: 
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    cancer = load_breast_cancer()    
    print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))
    
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    mean_on_train = X_train.mean(axis=0)
    std_on_train = X_train.std(axis=0)
    X_train_scaled = (X_train - mean_on_train) / std_on_train
    X_test_scaled = (X_test - mean_on_train) / std_on_train
    mlp = MLPClassifier(random_state=42, max_iter=1000)
    mlp.fit(X_train_scaled, y_train)
    print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
    print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
    
    plt.figure(figsize=(20, 5))
    plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
    plt.yticks(range(30), cancer.feature_names)
    plt.xlabel("Columns in weight matrix")
    plt.colorbar()
    plt.show()
    