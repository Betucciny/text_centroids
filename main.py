from funciones import *
import random
from nltk.corpus import reuters
import nltk


def main():
    nltk.download('reuters')
    file_names = reuters.fileids()
    texts = []
    labels = []
    print(len(file_names))
    for file_name in file_names:
        text = reuters.raw(file_name)
        texts.append(text)
        labels.append(reuters.categories(file_name))

    texts = [text_preprocessing(text) for text in texts]
    tfidf = calculate_tfidf(texts)

    # Randomize
    datos = [(x, y) for x, y in zip(tfidf, labels)]
    random.shuffle(datos)
    X, Y = zip(*datos)
    # Split
    X_train = X[:8000]
    Y_train = Y[:8000]
    X_test = X[8000:]
    Y_test = Y[8000:]

    # X_train, eigen, mean = PCA(X_train, 500)
    # new_X_test = []
    # for muestra in X_test:
    #     new_X_test.append(deconstruct(eigen, muestra - mean))
    # X_test = new_X_test

    X_train_expanded = []
    Y_train_expanded = []
    for x, labels in zip(X_train, Y_train):
        for label in labels:
            X_train_expanded.append(x)
            Y_train_expanded.append(label)

    X_train = X_train_expanded
    Y_train = Y_train_expanded

    # Centroids
    centroids, centroid_label = centroid(X_train, Y_train)
    centroids = np.asarray(centroids)
    buenas = 0
    # Predict
    for muestra, etiquetas in zip(X_test, Y_test):
        best_y = best_centroid(muestra, centroids, centroid_label)
        if best_y in etiquetas:
            print("Buena: ",)
            buenas += 1
        print("Etiquetas reales: ", etiquetas, "Etiqueta predicha: ", best_y)
    print("Porcentaje de aciertos: ", buenas / len(Y_test) * 100, "%")


if __name__ == "__main__":
    main()