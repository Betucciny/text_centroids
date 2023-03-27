import numpy as np
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')


def covariance_matrix(x):
    return np.cov(x, rowvar=False)


def eigenmatrix(datos, k):
    cov = covariance_matrix(datos)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    print(eigenvectors.shape)
    eigenvectors = eigenvectors.T
    eigen = [(i, j) for i, j in zip(eigenvalues, eigenvectors)]
    eigenmat = np.array([i[1] for j, i in enumerate(eigen) if j < k])
    print(eigenmat.shape)
    return eigenmat


def deconstruct(eigenmat, text):
    dec = eigenmat @ text.T
    return dec


def centroid(X, Y):
    clusters = {}
    for x, y in zip(X, Y):
        if y in clusters:
            clusters[y].append(x)
        else:
            clusters[y] = [x]
    centroids = []
    centroid_label = []
    for label, cluster in clusters.items():
        centroid = np.mean(cluster, axis=0)
        centroids.append(centroid)
        centroid_label.append(label)
    return centroids, centroid_label


def best_centroid(X, centroids, centroid_label):
    distances = []
    for c, label in zip(centroids, centroid_label):
        dist = np.linalg.norm(c - X)
        distances.append((label, dist))
    return min(distances, key=lambda x: x[1])[0]


def PCA(datos, k):
    eigenmat = eigenmatrix(datos, k)
    matriz = []
    mean = np.mean(datos, axis=0, keepdims=True)
    for i in range(datos.shape[0]):
        text = datos[i] - mean
        dec = deconstruct(eigenmat, text)
        if i == 0:
            matriz = dec
        else:
            matriz = np.hstack((matriz, dec))
    return matriz


def calculate_tfidf(documents: list):
    # Create a list of all the words in all documents
    all_words = list(set([word for doc in documents for word in doc]))

    # Create a list of dictionaries, one for each document, mapping each word in the document to its frequency
    freq_dicts = []
    for doc in documents:
        freq_dict = {}
        words = doc
        for word in words:
            if word in freq_dict:
                freq_dict[word] += 1
            else:
                freq_dict[word] = 1
        freq_dicts.append(freq_dict)

    # Create a list of dictionaries, one for each document, mapping each word in the document to its inverse document frequency
    idf_dicts = {}
    for word in all_words:
        idf_count = sum([1 for freq_dict in freq_dicts if word in freq_dict])
        idf_dicts[word] = np.log(len(documents) / idf_count)

    tfidf_dicts = []
    for freq_dict in freq_dicts:
        tfidf_dict = {}
        max_freq = sum([freq for freq in freq_dict.values()])
        # max_freq = max(freq_dict.values())
        for word, freq in freq_dict.items():
            tfidf_dict[word] = (freq / max_freq) * idf_dicts[word]
        tfidf_dicts.append(tfidf_dict)

    # Convert the list of dictionaries to a 2D array
    X = np.zeros((len(documents), len(all_words)))
    for i, tfidf_dict in enumerate(tfidf_dicts):
        for j, word in enumerate(all_words):
            if word in tfidf_dict:
                X[i, j] = tfidf_dict[word]

    X_norms = np.linalg.norm(X, axis=1)
    X_normalized = X / X_norms[:, np.newaxis]

    return X_normalized


def text_preprocessing(
        text: str,
        punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_“~''',
        stop_words=None
) -> list:
    if stop_words is None:
        stop_words = stopwords.words('english')
    for x in text.lower():
        if x in punctuations:
            text = text.replace(x, "")
    # Removing words that have numbers in them
    text = re.sub(r'\w*\d\w*', '', text)
    # Removing digits
    text = re.sub(r'[0-9]+', '', text)
    # Cleaning the whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Setting every word to lower
    text = text.lower()
    # Converting all our text to a list
    text = text.split(' ')
    # Droping empty strings
    text = [x for x in text if x != '']
    # Droping stop words
    text = [x for x in text if x not in stop_words]

    return text