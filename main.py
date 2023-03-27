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
        for label in reuters.categories(file_name):
            texts.append(text)
            labels.append(label)

    texts = [text_preprocessing(text) for text in texts]
    tfidf = calculate_tfidf(texts)

    # pca_datos = np.real(PCA(tfidf, 50)).T
    # print(pca_datos.shape)

    # Randomize
    datos = [(x, y) for x, y in zip(tfidf, labels)]
    # datos = [(x, y) for x, y in zip(pca_datos, labels)]
    random.shuffle(datos)
    X, Y = zip(*datos)
    # Split
    X_train = X[:8000]
    Y_train = Y[:8000]
    X_test = X[8000:]
    Y_test = Y[8000:]
    centroids, centroid_label = centroid(X_train, Y_train)
    buenas = 0
    # Predict
    for muestra, etiqueta in zip(X_test, Y_test):
        best_y = best_centroid(muestra, centroids, centroid_label)
        if best_y == etiqueta:
            buenas += 1
        print("Etiqueta real: ", etiqueta, "Etiqueta predicha: ", best_y)
    print("Porcentaje de aciertos: ", buenas / len(Y_test) * 100, "%")





if __name__ == "__main__":
    main()