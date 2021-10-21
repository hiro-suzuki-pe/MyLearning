import os
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

inno = 11
if inno == 1:
    adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
    data = pd.read_csv(
        adult_path, header=None, index_col=False,
        names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 
        'income'])
    
    data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
                 'occupation', 'income']]
    display(data.head())
    
    print(data.gender.value_counts())
    
    print("Original features:\n", list(data.columns), "\n")
    data_dummies = pd.get_dummies(data)
    print("Features after get_dummies:\n", list(data_dummies.columns))
    
    print(data_dummies.head())
    
    features = data_dummies.loc[:, 'age': 'occupation_ Transport-moving']
    
    X = features.values
    y = data_dummies['income_ >50K'].values
    print("X.shape: {}  y.shape: {}".format(X.shape, y.shape))
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))
    
    demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                            'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
    display(demo_df)
    
    print(pd.get_dummies(demo_df))
    
    demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
    p = pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])
    print(p)
if inno == 10:   # ビニング，離散化，線形モデル，決定木
    X, y = mglearn.datasets.make_wave(n_samples=100)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
    plt.plot(line, reg.predict(line), label="decision tyree")
    
    reg = LinearRegression().fit(X,y) 
    plt.plot(line, reg.predict(line), label="linear regression")
 
    plt.plot(X[:, 0], y, 'o', c='k')
    plt.ylabel("Regression output")
    plt.xlabel("Input feature")
    plt.legend(loc="best")
    plt.show()
if inno == 11:
    X, y = mglearn.datasets.make_wave(n_samples=100)

    bins = np.linspace(-3, 3, 11)
    print("bins: {}".format(bins))
    
    # inno==12
    which_bin = np.digitize(X, bins=bins)
    print('shape of X: {}'.format(X.shape))
    print("\nData points:\n", X[:5])
    print("\nBin membership for data points:\n", which_bin[:5])
    
    
        