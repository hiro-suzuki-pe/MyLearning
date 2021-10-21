import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE

inno = 41
print(">> Executing In[{}]:".format(inno))
if inno == 1:
    mglearn.plots.plot_scaling()
    plt.show()
if inno == 2:
    cancer = load_breast_cancer()
    
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                        random_state=1)
    print(X_train.shape)
    print(X_test.shape)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    print("transformed shape: {}".format(X_train_scaled.shape))
    print("per-feature minimum before scalling:\n{}".format(X_train.min(axis=0)))
    print("per-feature maximum before scalling:\n{}".format(X_train.max(axis=0)))
    print("per-feature minimum after scalling:\n{}".format(X_train_scaled.min(axis=0)))
    print("per-feature maximum after scalling:\n{}".format(X_train_scaled.max(axis=0)))
    
    X_test_scaled = scaler.transform(X_test)
    print("per-feature minimum after scalling:\n{}".format(X_test_scaled.min(axis=0)))
    print("per-feature maximum after scalling:\n{}".format(X_test_scaled.max(axis=0)))
if inno == 7:
    X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
    X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(X_train[:,0], X_train[:,1], c=mglearn.cm2(0), label="Training set", s=60)
    axes[0].scatter(X_train[:,0], X_train[:,1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
    axes[0].legend(loc='upper left')
    axes[0].set_title("Original Data")
    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
    axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c=mglearn.cm2(1), label="test set", s=60)
    axes[1].set_title("Scaled Data")
    
    test_scaler = MinMaxScaler()
    test_scaler.fit(X_test)
    X_test_scaled_badly = test_scaler.transform(X_test)
    
    axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
    axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^', c=mglearn.cm2(1), label="test set", s=60)
    axes[2].set_title("Improperly Scaled Data")
    
    for ax in axes:
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    plt.show()
if inno == 9:
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    svm = SVC(C=100)
    svm.fit(X_train, y_train)
    print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))
    
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm.fit(X_train_scaled, y_train)
    
    print("Scaled test set accuracy: {:.2f}".format(
        svm.score(X_test_scaled, y_test)))

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm = SVC(C=100)
    svm.fit(X_train_scaled, y_train)
    
    print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))
if inno == 13:      # 次元削減，特徴量抽出，多様体学習    
    mglearn.plots.plot_pca_illustration()
    plt.show()
    
    cancer = load_breast_cancer()
    fig, axes = plt.subplots(15, 2, figsize=(10, 20))
    malignant = cancer.data[cancer.target == 0]
    benign = cancer.data[cancer.target == 1]
    
    ax = axes.ravel()
    
    for i in range(30):
        _, bins = np.histogram(cancer.data[:,i], bins=50)
        ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=0.5)
        ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=0.5)
        ax[i].set_title(cancer.feature_names[i])
        ax[i].set_yticks(())
    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(["malignant", "benign"], loc="best")
    fig.tight_layout()
    plt.show()
if inno == 14:
    cancer = load_breast_cancer()
    scaler = StandardScaler()
    scaler.fit(cancer.data)
    X_scaled = scaler.transform(cancer.data)
    
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    X_pca = pca.transform(X_scaled)
    print("Original shape: {}".format(str(X_scaled.shape)))
    print("Reduced shape: {}".format(str(X_pca.shape)))
    print("PCA component shape: {}".format(pca.components_.shape))
    print("PCA components:\n{}".format(pca.components_))
    
    plt.figure(figsize=(8, 8))
    mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
    plt.legend(cancer.target_names, loc="best")
    plt.gca().set_aspect("equal")
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()
    
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(cancer.feature_names)),
               cancer.feature_names, rotation=60, ha="left")
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()
if inno == 20:
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape
    
    fix, axes = plt.subplots(2, 5, figsize=(15, 8), 
                             subplot_kw={'xticks': (), 'yticks': ()})
    for target, image, ax in zip(people.target, people.images, axes.ravel()):
        ax.imshow(image)
        ax.set_title(people.target_names[target])
        print("people.images.shape: {}".format(people.images.shape))
        print("Number of classes: {}".format(len(people.target_names)))
    plt.show()
    
    counts = np.bincount(people.target)
    for i, (count, name) in enumerate(zip(counts, people.target_names)):
        print("{0:25} {1:3}".format(name, count), end='   ')
        if (i + 1) % 3 == 0:
            print()
    print()
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
        mask
        X_people = people.data[mask]
        y_people = people.target[mask]
        X_people = X_people / 255.0
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_people, y_people, stratify=y_people, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))
    
    mglearn.plots.plot_pca_whitening()
    plt.show()
if inno == 26:
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    X_people = X_people / 255

    y_people = people.target[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X_people, y_people, stratify=y_people, random_state=0)

    pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
    X_train_pca = pca.transform(X_train)
    x_test_pca = pca.transform(X_test)
    print("X_train_pca.shape: {}".format(X_train_pca.shape))
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print("Test set accuracy of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))
    
    print("pca.components_.shape: {}".format(pca.components_.shape))
    fix, axes = plt.subplots(3, 5, figsize=(15, 12), 
                             subplot_kw={"xticks": (), "yticks": ()})
    for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap='viridis')
        ax.set_title("{}. component".format((i + 1)))
    plt.show()
    
    mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)  # Fig3-1 が出ない。
    plt.show()
if inno == 31:
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask] 
    y_people = people.target[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X_people, y_people, stratify=y_people, random_state=0)

    pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
    X_train_pca = pca.transform(X_train)
    x_test_pca = pca.transform(X_test)
    mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()
if inno == 32:
    mglearn.plots.plot_nmf_illustration()
    plt.show()
if inno == 33:      # ???
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask] / 255
    y_people = people.target[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X_people, y_people, stratify=y_people, random_state=0)
    mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
    plt.show()
if inno == 34:      
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask] / 255
    y_people = people.target[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X_people, y_people, stratify=y_people, random_state=0)
    
    nmf =NMF(n_components=15, random_state=0)
    nmf.fit(X_train)
    X_train_nmf = nmf.transform(X_train)
    X_test_nmf = nmf.transform(X_test)
    fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={"xticks": (), "yticks": ()})
    for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape))
        ax.set_title("{}. component".format(i))
                                                                 
    plt.show()   
    
    compn = 3
    inds = np.argsort(X_train_nmf[:, compn])[::-1]
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        ax.imshow(X_train[ind].reshape(image_shape))
    plt.show()
    
    compn = 7
    inds = np.argsort(X_train_nmf[:, compn])[::-1]
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        ax.imshow(X_train[ind].reshape(image_shape))
    plt.show()
if inno == 36:
    S = mglearn.datasets.make_signals()
    print("Shape of S: {}".format(S.shape))
    plt.figure(figsize=(6, 1))
    plt.plot(S, '-')
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.show()
    
    A = np.random.RandomState(0).uniform(size=(100, 3))
    X = np.dot(S, A.T)
    print("Shape of measurements: {}".format(X.shape))
    
    nmf = NMF(n_components=3, random_state=42)
    S_ = nmf.fit_transform(X)
    print("Recovered signal shape: {}".format(S_.shape))
    
    pca = PCA(n_components=3)
    H = pca.fit_transform(X)
    
    models = [X, S, S_, H]
    names = ['Observations (first three measurements)', 
             'True sources', 
             'NMF recovered signals',
             'PCA recovered signals']
    fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5},
                             subplot_kw={'xticks': (), 'yticks': ()})
    
    for model, name, ax in zip(models, names, axes):
        ax.set_title(name)
        ax.plot(model[:, :3], '-')
    ax.legend(loc='upper right')
    plt.show()
if inno == 41:
    digits = load_digits()
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})
    for ax, img in zip(axes.ravel(), digits.images):
        ax.imshow(img)
    plt.show()
    
    pca = PCA(n_components=2)
    pca.fit(digits.data)
    digits_pca = pca.transform(digits.data)
    colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
              "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
    plt.figure(figsize=(10, 10))
    plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
    plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
    for i in range(len(digits.data)):
        plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), 
                 color = colors[digits.target[i]], 
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()

    tsne = TSNE(random_state=42)
    digits_tsne = tsne.fit_transform(digits.data)
    
    plt.figure(figsize=(10, 10))
    plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())
    plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())
    for i in range(len(digits.data)):
        plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), 
                color = colors[digits.target[i]], 
                fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel("t-SNE First principal component")
    plt.ylabel("t-SNE Second principal component")
    plt.show()