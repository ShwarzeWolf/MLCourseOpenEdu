from IPython.display import Image
from imutils import paths
import numpy as np
import cv2
import os

def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

imagePaths = sorted(list(paths.list_images('./train')))
trainData = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    hist = extract_histogram(image)
    trainData.append(hist)
    labels.append(label)

Y = [1 if x == 'cat' else 0 for x in labels]

from sklearn.svm import LinearSVC
svm = LinearSVC(random_state = 462, C = 1.09)
svm.fit(trainData, Y)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', #критерий разделения
                              min_samples_leaf=10, #минимальное число объектов в листе
                              max_leaf_nodes=20, #максимальное число листьев
                              random_state=462)
bagging = BaggingClassifier(tree, #базовый алгоритм
                            n_estimators=19, #количество деревьев
                            random_state=462)
bagging.fit(trainData, Y)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=19, #количество деревьев
                             criterion='entropy', #критерий разделения
                              min_samples_leaf=10, #минимальное число объектов в листе
                              max_leaf_nodes=20, #максимальное число листьев
                              random_state=462)
forest.fit(trainData, Y)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs', random_state=462)

from sklearn.ensemble import StackingClassifier

base_estimators = [('SVM', svm), ('Bagging DT', bagging), ('DecisionForest', forest)]
sclf = StackingClassifier(estimators=base_estimators, final_estimator=lr, cv=2)
sclf.fit(trainData, Y)


print(sclf.score(trainData, Y))

imageList = ['./test/dog.1023.jpg', './test/dog.1029.jpg', './test/dog.1006.jpg', './test/cat.1004.jpg']

for singleImagePath in imageList:
    singleImage = cv2.imread(singleImagePath)
    histt = extract_histogram(singleImage)
    histt2 = histt.reshape(1, -1)
    prediction = sclf.predict(histt2)
    print(sclf.predict_proba(histt2))