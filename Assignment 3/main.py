import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate


def features_selected(df, X_new):
    v = 0
    currfeat = []
    mask = X_new.get_support()
    for col in df.columns:
        if v < 8 and mask[v] == True:
            currfeat.append(col)
        v += 1
    return currfeat


def ForwardFeatureSelection(X, Y, df, dec):
    if dec is True:
        knn = KNeighborsClassifier(n_neighbors=3)
        sfs = SequentialFeatureSelector(knn, n_features_to_select=3, direction='forward')
    else:
        clf = ExtraTreesRegressor(n_estimators=50)
        sfs = SequentialFeatureSelector(clf, n_features_to_select=3, direction='forward')
    sfs1 = sfs.fit(X, Y)
    return features_selected(df, sfs1)


def ClassifierValues(name, scaling, seletor, param, clf, x, y):
    values = [name, scaling, seletor, param]
    # values = []
    values.append(cross_val_score(clf, x, y, cv=5, scoring='accuracy').mean().round(3))
    values.append(cross_val_score(clf, x, y, cv=5, scoring='recall').mean().round(3))
    values.append(cross_val_score(clf, x, y, cv=5, scoring='precision').mean().round(3))
    values.append(cross_val_score(clf, x, y, cv=5, scoring='f1_macro').mean().round(3))
    return values


def AdaboostValues(scale, feature_selector, X, Y):
    adaboost_clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    return ClassifierValues(name="Adaboost", scaling=scale, seletor=feature_selector, param="n_estimators= 10",
                            clf=adaboost_clf, x=X,
                            y=Y)


def RandomForestValues(scale, feature_selector, X, Y):
    rf_clf = RandomForestClassifier(max_depth=4, random_state=0)
    return ClassifierValues(name="Random Forest", scaling=scale, seletor=feature_selector, param="n_estimators= 10",
                            clf=rf_clf, x=X, y=Y)


def BaggingValues(scale, feature_selector, X, Y):
    bagging_clf = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0)
    return ClassifierValues(name="Bagging", scaling=scale, seletor=feature_selector, param="n_estimators= 10",
                            clf=bagging_clf, x=X, y=Y)


def NaiveBayesValues(scale, feature_selector, X, Y):
    nb_clf = GaussianNB()
    return ClassifierValues(name="Naive Bayes", scaling=scale, seletor=feature_selector, param="n_estimators= 100",
                            clf=nb_clf, x=X, y=Y)


def DecisionTreeValues(scale, feature_selector, X, Y):
    dt_clf = DecisionTreeClassifier()
    return ClassifierValues(name="Decision Tree", scaling=scale, seletor=feature_selector, param="n_estimators= 100",
                            clf=dt_clf, x=X, y=Y)


def RFESelection(X, Y, df):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=3, step=1)
    X_new = selector.fit(X, Y)
    return features_selected(df, X_new)


names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
         'DiabetesPredigreeFuntion', 'Age', 'Outcome']
df = pd.read_csv('pima-indians-diabetes.csv', names=names)
output = pd.read_csv('output.csv')
for col in output.columns:
    print(output[col].max())

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

# FFS
print("Forward Feature Selection:")
fsf_unscaled = ForwardFeatureSelection(X, Y, df, dec=True)
fsf_scaled = ForwardFeatureSelection(Xz, Y, df_scaled, dec=False)

# RFE feature selection
print("\n\nRFE:")
rfe_unscaled = RFESelection(X, Y, df)
rfe_scaled = RFESelection(Xz, Y, df_scaled)

print("\nBuilding table...\n")
# Table information
data = []
table_names = ["Name", "Scaling Feature", "selector", "parameter", "Recall", "Precision", "Accuracy", "F1score"]

# AdaBoost
data.append(AdaboostValues(scale="none", feature_selector="none", X=X, Y=Y))
data.append(AdaboostValues(scale="z-score", feature_selector="none", X=Xz, Y=Y))
data.append(AdaboostValues(scale="z-score", feature_selector="ffs", X=df[fsf_scaled], Y=Y))
data.append(AdaboostValues(scale="none", feature_selector="ffs", X=df[fsf_unscaled], Y=Y))
data.append(AdaboostValues(scale="z-score", feature_selector="rfe", X=df[rfe_scaled], Y=Y))
data.append(AdaboostValues(scale="none", feature_selector="rfe", X=df[rfe_unscaled], Y=Y))

# RandomForest
data.append(RandomForestValues(scale="none", feature_selector="none", X=X, Y=Y))
data.append(RandomForestValues(scale="z-score", feature_selector="none", X=Xz, Y=Y))
data.append(RandomForestValues(scale="z-score", feature_selector="ffs", X=df[fsf_scaled], Y=Y))
data.append(RandomForestValues(scale="none", feature_selector="ffs", X=df[fsf_unscaled], Y=Y))
data.append(RandomForestValues(scale="z-score", feature_selector="rfe", X=df[rfe_scaled], Y=Y))
data.append(RandomForestValues(scale="none", feature_selector="rfe", X=df[rfe_unscaled], Y=Y))

# NaiveBayes
data.append(NaiveBayesValues(scale="none", feature_selector="none", X=X, Y=Y))
data.append(NaiveBayesValues(scale="z-score", feature_selector="none", X=Xz, Y=Y))
data.append(NaiveBayesValues(scale="z-score", feature_selector="ffs", X=df[fsf_scaled], Y=Y))
data.append(NaiveBayesValues(scale="none", feature_selector="ffs", X=df[fsf_unscaled], Y=Y))
data.append(NaiveBayesValues(scale="z-score", feature_selector="rfe", X=df[rfe_scaled], Y=Y))
data.append(NaiveBayesValues(scale="none", feature_selector="rfe", X=df[rfe_unscaled], Y=Y))

# BaggingClassifier
data.append(BaggingValues(scale="none", feature_selector="none", X=X, Y=Y))
data.append(BaggingValues(scale="z-score", feature_selector="none", X=Xz, Y=Y))
data.append(BaggingValues(scale="z-score", feature_selector="ffs", X=df[fsf_scaled], Y=Y))
data.append(BaggingValues(scale="none", feature_selector="ffs", X=df[fsf_unscaled], Y=Y))
data.append(BaggingValues(scale="z-score", feature_selector="rfe", X=df[rfe_scaled], Y=Y))
data.append(BaggingValues(scale="none", feature_selector="rfe", X=df[rfe_unscaled], Y=Y))

# Decision Tree Classifier
data.append(DecisionTreeValues(scale="none", feature_selector="none", X=X, Y=Y))
data.append(DecisionTreeValues(scale="z-score", feature_selector="none", X=Xz, Y=Y))
data.append(DecisionTreeValues(scale="z-score", feature_selector="ffs", X=df[fsf_scaled], Y=Y))
data.append(DecisionTreeValues(scale="none", feature_selector="ffs", X=df[fsf_unscaled], Y=Y))
data.append(DecisionTreeValues(scale="z-score", feature_selector="rfe", X=df[rfe_scaled], Y=Y))
data.append(DecisionTreeValues(scale="none", feature_selector="rfe", X=df[rfe_unscaled], Y=Y))

# Prints table
print(tabulate(data, headers=table_names))
