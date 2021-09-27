from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imutils import paths
import numpy as np
import cv2
import os

def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

imagePaths = sorted(list(paths.list_images('train')))
data = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    hist = extract_histogram(image)
    data.append(hist)
    labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)
print(labels[0])

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.25, random_state=9)

model = LinearSVC(random_state = 9, C = 0.51)
model.fit(trainData, trainLabels)

predictions = model.predict(testData)
print(classification_report(testLabels, predictions, target_names=le.classes_))

predictions = model.predict(testData)
f1_score(testLabels, predictions, average='macro')

print(model.coef_[0][256])
print(model.coef_[0][370])
print(model.coef_[0][124])

predictionList = ["prediction/cat.1046.jpg", "prediction/dog.1025.jpg", "prediction/cat.1042.jpg", "prediction/cat.1003.jpg"]

for animalPhoto in predictionList:
    singleImage = cv2.imread(animalPhoto)

    histt = extract_histogram(singleImage)
    histt2 = histt.reshape(1, -1)
    prediction = model.predict(histt2)

    print(prediction)
