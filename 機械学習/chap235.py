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

inno = 70
if inno == 49:
    mglearn.plots.plot_animal_tree()
    plt.show()
if inno == 56:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import export_graphviz
    
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target,random_state=42)
    tree = DecisionTreeClassifier(random_state=0, max_depth=4)
    tree.fit(X_train, y_train)
    print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
    
    export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
                    feature_names=cancer.feature_names, impurity=False, filled=True)
    
    import graphviz
    with open("tree.dot", "rt") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
    
    print("Feature importances:\n{}".format(tree.feature_importances_))
    
    plot_feature_importances_cancer(tree)
    plt.show()
    
    tree = mglearn.plots.plot_tree_not_monotone()
    display(tree)
    plt.show()
if inno == 63:
    import os
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
    
    ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH,
                                          "ram_price.csv"))
    
    plt.semilogy(ram_prices.date, ram_prices.price)
    plt.xlabel("Year")
    plt.ylabel("Price in $/Mbyte")
    plt.show()
    
    data_train = ram_prices[ram_prices.date < 2000]
    data_test = ram_prices[ram_prices.date >= 2000]
    
    X_train = data_train.date[:, np.newaxis]
    y_train = np.log(data_train.price)
    
    tree = DecisionTreeRegressor().fit(X_train, y_train)
    linear_reg = LinearRegression().fit(X_train, y_train)
    
    X_all = ram_prices.date[:, np.newaxis]
    
    pred_tree = tree.predict(X_all)
    pred_lr = linear_reg.predict(X_all)
    
    price_tree = np.exp(pred_tree)
    price_lr = np.exp(pred_lr)
    
    plt.semilogy(data_train.date, data_train.price, label="Training data")
    plt.semilogy(data_test.date, data_test.price, label="Test data")
    plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
    plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
    plt.legend()
    plt.show()
if inno == 66:      # 決定木のアンサンブル法   
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_moons   
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    forest = RandomForestClassifier(n_estimators=5, random_state=2)
    forest.fit(X_train, y_train)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        ax.set_title("Tree {}".format(i))
        mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

    mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
                                    alpha=.4)
    axes[-1, -1].set_title("Random Forest")
    mglearn.discrete_scatter(X_train[:,0], X_train[:, 1], y_train)
    plt.show()
if inno == 68:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_moons   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer
    
    cancer = load_breast_cancer()
    
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
    forest = RandomForestClassifier(n_estimators=100, random_state=0)
    forest.fit(X_train, y_train)
    
    print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
    
    plot_feature_importances_cancer(forest)
    plt.show()
if inno == 70:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_moons   
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=0)
#    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
#    gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
    gbrt.fit(X_train, y_train)
   
    print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
    print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))
  
    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt.fit(X_train, y_train)
    plot_feature_importances_cancer(gbrt)
    plt.show()
    