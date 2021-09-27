from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

(X_train, y_train), (X_pred, y_pred) = mnist.load_data()

dim = 784 # 28*28
X_train = X_train.reshape(len(X_train), dim)

pca = PCA(n_components=56, svd_solver='full')
modelPCA = pca.fit(X_train)
X_train = modelPCA.transform(X_train)

print(pca.explained_variance_ratio_)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=95)

logisticRegression = LogisticRegression(solver='lbfgs', random_state=95)
logisticClassification = OneVsRestClassifier(logisticRegression).fit(X_train, y_train)

y_pred = logisticClassification.predict(X_test)
logisticConfusionMatrix = confusion_matrix(y_test, y_pred)

print(logisticConfusionMatrix[6][6])

test_data = pd.read_csv('./pred_for_task.csv')

test_data_x = test_data.iloc[:, 2:]
test_data_y = test_data.iloc[:, 1:2]
test_data_x = modelPCA.transform(test_data_x)

prediction = logisticClassification.predict_proba(test_data_x)
print(max(prediction[0]))
