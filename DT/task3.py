import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

raw_data = pd.read_csv("./diabetes.csv")

data = raw_data.head(660)

print(len(data[data['Outcome'] == 0]))

train = data.head(int(len(data)*0.8))
test = data.tail(int(len(data)*0.2))

features = list(train.columns[:8])
x = train[features]
y = train['Outcome']

tree = DecisionTreeClassifier(criterion='entropy', #критерий разделения
                              min_samples_leaf=5, #минимальное число объектов в листе
                              max_leaf_nodes=5, #максимальное число листьев
                              random_state=2020)
clf=tree.fit(x, y)

columns = list(x.columns)
export_graphviz(clf, out_file='tree.dot',
                feature_names=columns,
                class_names=['0', '1'],
                rounded = True, proportion = False,
                precision = 2, filled = True, label='all')

with open('tree.dot') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)

print(clf.tree_.max_depth)

x = test[features]
y_true = test['Outcome']
y_pred = clf.predict(x)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_pred))

from sklearn.metrics import f1_score
print(f1_score(y_true, y_pred, average='macro'))

numbers = [751, 748, 754, 746]

for i in numbers:
    print(clf.predict([raw_data.loc[i, features].tolist()])[0])

