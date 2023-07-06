from keras.datasets import mnist
from sklearn.svm import SVC
import pickle
import cv2
import numpy as np
from sklearnex import patch_sklearn 

patch_sklearn() 

def Preprocess(data):
    resized_images = []
    for image in data:
        resized_image = cv2.resize(image.reshape(28, 28), (14, 14))
        resized_images.append(resized_image.flatten())
    return  np.array(resized_images)

(X_train, y_train), (_, _) = mnist.load_data()

X_train = Preprocess(X_train)

classifier = SVC(kernel='rbf', probability=True)
classifier.fit(X_train, y_train)

pickle.dump(classifier, open("SVM.pkl", 'wb'))
