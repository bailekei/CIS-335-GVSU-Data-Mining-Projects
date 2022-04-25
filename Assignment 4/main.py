from numpy import where
from numpy import unique
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from tabulate import tabulate
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt


def score_values(if_scaled, solver_is, func, x_val, y_val):
    value = [if_scaled, solver_is]
    # Accuracy
    score = cross_val_score(nn, X, Y, cv=5, scoring='accuracy')
    value.append(score.mean().round(3))
    # Recall
    score = cross_val_score(nn, X, Y, cv=3, scoring='recall')
    value.append(score.mean().round(3))
    # Precision
    score = cross_val_score(nn, X, Y, cv=3, scoring='precision')
    value.append(score.mean().round(3))
    # F1
    score = cross_val_score(nn, X, Y, cv=5, scoring='f1_macro')
    value.append(score.mean().round(3))
    return value


names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
         'DiabetesPredigreeFuntion', 'Age', 'Outcome']
df = pd.read_csv('pima-indians-diabetes.csv', names=names)

# Z-score scaling
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled.head()

# Assign X and Y values
X = df.drop(['Outcome'], axis=1)
Y = df['Outcome']

# X and Y for z scaled
Xz = df_scaled.drop(['Outcome'], axis=1)

# Setting up table values
table_title = ['Scaling', 'Hidden_layer_combo', 'Accuracy', 'Recall', 'Precision', 'F1']
table = []

nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, ), random_state=1, max_iter=1000)
mn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(6, 1), random_state=1, max_iter=1000)
mlp = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(4, 3), random_state=1, max_iter=1000)
tw = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(3, 4), random_state=1, max_iter=2000)

# Unscaled
nn.fit(X, Y)
table.append(score_values('Unscaled', '5, 2',  nn, X, Y))
clf.fit(X, Y)
table.append(score_values('Unscaled', '7, ',  clf, X, Y))
mn.fit(X, Y)
table.append(score_values('Unscaled', '6, 1',  nn, X, Y))
mlp.fit(X, Y)
table.append(score_values('Unscaled', '4, 3',  mlp, X, Y))
tw.fit(X, Y)
table.append(score_values('Unscaled', '3, 4',  tw, X, Y))
#
# # Scaled
nn.fit(Xz, Y)
table.append(score_values('Scaled', '5, 2',  mn, Xz, Y))
clf.fit(Xz, Y)
table.append(score_values('Scaled', '7, ', clf, Xz, Y))
mn.fit(Xz, Y)
table.append(score_values('Scaled', '6, 1',  mn, Xz, Y))
mlp.fit(Xz, Y)
table.append(score_values('Scaled', '4, 3', mlp, Xz, Y))
tw.fit(Xz, Y)
table.append(score_values('Unscaled', '3, 4', tw, Xz, Y))

print(tabulate(table, headers=table_title, tablefmt='fancy_grid'))

X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
model = AgglomerativeClustering(n_clusters=2)
that = model.fit_predict(X)
clusters = unique(that)

for cluster in clusters:
    row_x = where(that == cluster)
    plt.scatter(X[row_x, 0], X[row_x, 1])

plt.show()
