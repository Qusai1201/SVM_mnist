import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import seaborn as sns
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc , classification_report
import pickle
from sklearnex import patch_sklearn 

patch_sklearn()


def Preprocess(data):
    resized_images = []
    for image in data:
        resized_image = cv2.resize(image.reshape(28, 28), (14, 14))
        resized_images.append(resized_image.flatten())
    return  np.array(resized_images)

(_, _), (X_test, y_test) = mnist.load_data()


X_test = Preprocess(X_test)


classifier = pickle.load(open("SVM.pkl", 'rb'))


y_pred = classifier.predict(X_test)


precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
classification_report = classification_report(y_test , y_pred)
print(classification_report)
print("accuracy_score : " , accuracy_score(y_test , y_pred))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(10):
    y_test_binary = np.where(y_test == i, 1, 0)
    y_pred_prob = classifier.predict_proba(X_test)[:, i]
    fpr[i], tpr[i], _ = roc_curve(y_test_binary, y_pred_prob)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(10):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {} (AUC = {:.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

Conf_mat = confusion_matrix(y_pred , y_test)


plt.figure(figsize=(10,10))
sns.heatmap(Conf_mat, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


