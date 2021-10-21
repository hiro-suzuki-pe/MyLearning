import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score

inno = 84
print(">> Executing In[{}]:".format(inno))

if inno == 45:
    mglearn.plots.plot_kmeans_algorithm()
    plt.show()
    
    mglearn.plots.plot_kmeans_boundaries()
    plt.show()
if inno == 47:
    X, y = make_blobs(random_state=1)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    print("Cluster memberships:\n{}".format(kmeans.labels_))
    print(kmeans.predict(X))
    
    mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
    mglearn.discrete_scatter(
        kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], [0, 1, 2], 
        markers='^', markeredgewidth=2)
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    assignments = kmeans.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])
    
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    assignments = kmeans.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])
    plt.show()
if inno == 52:
    X_varied, y_varied = make_blobs(n_samples=200,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=170)
    y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)
    mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
    plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc='best')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
if inno == 53:
    X, y = make_blobs(random_state=170, n_samples=600)
    rng = np.random.RandomState(74)
    
    transformation = rng.normal(size=(2, 2))
    X = np.dot(X, transformation)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    
    plt.scatter(X[:,0], X[:,1], c=y_pred, cmap=mglearn.cm3)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
                marker='^', c=[0, 1, 2], s=100, linewidth=2, cmap=mglearn.cm3)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
    

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
if inno == 54:
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    
    plt.scatter(X[:,0], X[:,1], c=y_pred, cmap=mglearn.cm2, s=60)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                marker='^', c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidth=2)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
if inno == 55:
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask] / 255
    y_people = people.target[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X_people, y_people, stratify=y_people, random_state=0)
    
    nmf =NMF(n_components=100, random_state=0)
    nmf.fit(X_train)
    pca = PCA(n_components=100, random_state=0)
    pca.fit(X_train)
    kmeans = KMeans(n_clusters=100, random_state=0)
    kmeans.fit(X_train)
    
    X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
    X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]
    X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)
    
    fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks':()})
    fig.suptitle("Extracted Components")
    for ax, comp_kmeans, comp_pca, comp_nmf in zip(
            axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
        ax[0].imshow(comp_kmeans.reshape(image_shape))
        ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
        ax[2].imshow(comp_nmf.reshape(image_shape))
    
    axes[0,0].set_ylabel("kmeans")
    axes[1,0].set_ylabel("pca")
    axes[2,0].set_ylabel("nmf")
    
    fig, axes = plt.subplots(4, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks':()})
    fig.suptitle("Reconstructions")
    for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(
            axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca, X_reconstructed_nmf):
        ax[0].imshow(orig.reshape(image_shape))
        ax[1].imshow(rec_kmeans.reshape(image_shape))
        ax[2].imshow(rec_pca.reshape(image_shape))
        ax[3].imshow(rec_nmf.reshape(image_shape))
    
    axes[0,0].set_ylabel("original")
    axes[1,0].set_ylabel("kmeans")
    axes[2,0].set_ylabel("pca")
    axes[3,0].set_ylabel("nmf")
                                                                 
    plt.show()   
if inno == 57:
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    
    plt.scatter(X[:,0], X[:,1], c=y_pred, s=60, cmap='Paired')
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
                s=60, marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
    
    print("Cluster memberships:\n{}".format(y_pred))
    distance_features = kmeans.transform(X)
    print("Distance feature shape: {}".format(distance_features.shape))
    print("Distance features:\n{}".format(distance_features))
if inno == 59:
    mglearn.plots.plot_agglomerative_algorithm()
    plt.show()
if inno == 60:
    X, y = make_blobs(random_state=1)
    agg = AgglomerativeClustering(n_clusters=3)
    assignment = agg.fit_predict(X)
    
    mglearn.discrete_scatter(X[:,0], X[:,1], assignment)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
if inno == 61:
    mglearn.plots.plot_agglomerative()
    plt.show()
if inno == 62:    
    X, y = make_blobs(random_state=0, n_samples=12)
    linkage_array = ward(X)
    dendrogram(linkage_array)
    
    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [7.25, 7.25], '--', c='k')
    ax.plot(bounds, [4, 4], '--', c='k')
    ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
    ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
    plt.xlabel("Sample index")
    plt.ylabel("Cluster distance")
    plt.show()
if inno == 63:      # 3.5.3 DBSCAN
    X, y = make_blobs(random_state=0, n_samples=12)
    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(X)
    print("Cluster memberships:\n{}".format(clusters))
    
    mglearn.plots.plot_dbscan()
    plt.show()
if inno == 65:
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(X_scaled)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
    plt.xlabel("Feature 0")
    plt.ylabel("feature 1")       
    plt.show()
if inno == 66:
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
    algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),
                  DBSCAN()]
    
    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X))
    
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,cmap=mglearn.cm3, s=60)
    axes[0].set_title("Random assignment - ARI: {:.2f}".format( 
                                                adjusted_rand_score(y, random_clusters)))
    
    for ax, algorithm in zip(axes[1:], algorithms):
        clusters = algorithm.fit_predict(X_scaled)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
        ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__, 
                                               adjusted_rand_score(y, clusters)))
    plt.show()
    
    clusters1 = [0, 0, 1, 1, 0]
    clusters2 = [1, 1, 0, 0, 1]
    
    print("Accuracy: {:.2f}".format(accuracy_score(clusters1, clusters2)))
    print("ARI: {:.2f}".format(adjusted_rand_score(clusters1, clusters2)))
if inno == 68:
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
    algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]
    
    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X))
    
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,cmap=mglearn.cm3, s=60)
    axes[0].set_title("Random assignment - ARI: {:.2f}".format( 
        silhouette_score(X_scaled, random_clusters)))
    
    for ax, algorithm in zip(axes[1:], algorithms):
        clusters = algorithm.fit_predict(X_scaled)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
        ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__, 
            silhouette_score(X_scaled, clusters)))
    plt.show()
if inno == 69:    
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask] / 255
    y_people = people.target[mask]

    pca = PCA(n_components=100, whiten=True, random_state=0)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)
    
    dbscan = DBSCAN(min_samples=3, eps=15)
    labels = dbscan.fit_predict(X_pca)
    print("Unique labels: {}".format(np.unique(labels)))
    print("Number of points per cluster: {}".format(np.bincount(labels + 1)))
    
    noise = X_people[labels==-1]
    fig, axes = plt.subplots(3, 9, subplot_kw={'xticks': (), 'yticks': ()})
    for image, ax in zip(noise, axes.ravel()):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
    plt.show()
    
    for eps in [1, 3, 5, 7, 9, 11, 13]:
        print("\neps={}".format(eps))
        dbscan = DBSCAN(eps=eps, min_samples=3)
        labels = dbscan.fit_predict(X_pca)
        print("Clusters present: {}".format(np.unique(labels)))
        print("Cluster sizes: {}".format(np.bincount(labels + 1)))
if inno == 76:    
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask] / 255
    y_people = people.target[mask]

    pca = PCA(n_components=100, whiten=True, random_state=0)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)
        
    dbscan = DBSCAN(min_samples=3, eps=7)
    labels = dbscan.fit_predict(X_pca)
    for cluster in range(max(labels) + 1):
        mask = labels == cluster
        n_images = np.sum(mask)
        fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4),
                                 subplot_kw={'xticks': (), 'yticks': ()})
        for image, label, ax in zip(X_people[mask], y_people[mask], axes):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1])
    plt.show()
if inno == 77:
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask] / 255
    y_people = people.target[mask]

    pca = PCA(n_components=100, whiten=True, random_state=0)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)


    km = KMeans(n_clusters=10, random_state=0)
    labels_km = km.fit_predict(X_pca)
    print("Cluster sizes k-means: {}".format(np.bincount(labels_km)))
    
    fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(12, 4))
    for center, ax in zip(km.cluster_centers_, axes.ravel()):
        ax.imshow(pca.inverse_transform(center).reshape(image_shape), vmin=0, vmax=1)
    plt.show()
    
    mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)
    plt.show()
if inno == 80:
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask] / 255
    y_people = people.target[mask]

    pca = PCA(n_components=100, whiten=True, random_state=0)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)

    km = KMeans(n_clusters=10, random_state=0)
    labels_km = km.fit_predict(X_pca)
    
    agglomerative = AgglomerativeClustering(n_clusters=10)
    labels_agg = agglomerative.fit_predict(X_pca)
    print("Cluster sizes agglomerative clustering: {}".format(np.bincount(labels_agg)))
    print("ARI: {:.2f}".format(adjusted_rand_score(labels_agg, labels_km)))    
    
    linkage_array = ward(X_pca)
    
    plt.figure(figsize=(20, 5))
    dendrogram(linkage_array, p=7, truncate_mode='level', no_labels=True)
    plt.xlabel("Sample index")
    plt.ylabel("Cluster distance")
    plt.show()
    
    n_clusters = 10
    for cluster in range(n_clusters):
        mask = labels_agg == cluster
        fig, axes = plt.subplots(1, 10, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(15, 8))
        axes[0].set_ylabel(np.sum(mask))
        for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1],
                        fontdict={'fontsize': 9})
    plt.show()    
if inno == 84:
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask] / 255
    y_people = people.target[mask]

    pca = PCA(n_components=100, whiten=True, random_state=0)
    pca.fit_transform(X_people)
    X_pca = pca.transform(X_people)

    km = KMeans(n_clusters=10, random_state=0)
    labels_km = km.fit_predict(X_pca)
    
    agglomerative = AgglomerativeClustering(n_clusters=40)
    labels_agg = agglomerative.fit_predict(X_pca)
    print("Cluster sizes agglomerative clustering: {}".format(np.bincount(labels_agg)))
    print("ARI: {:.2f}".format(adjusted_rand_score(labels_agg, labels_km)))    
    
    linkage_array = ward(X_pca)
    
    plt.figure(figsize=(20, 5))
    dendrogram(linkage_array, p=7, truncate_mode='level', no_labels=True)
    plt.xlabel("Sample index")
    plt.ylabel("Cluster distance")
    plt.show()
    
    n_clusters = 40
    for cluster in [10, 13, 19, 22, 36]:
        mask = labels_agg == cluster
        fig, axes = plt.subplots(1, 15, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(15, 8))
        cluster_size = np.sum(mask)
        axes[0].set_ylabel("#{}: {}".format(cluster, cluster_size))
        for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1],
                        fontdict={'fontsize': 9})
        for i in range(cluster_size, 15):  
            axes[i].set_visible(False)
    plt.show()    
   
    
    
    
    
    
    
    