import pandas as pd
from scipy.stats import stats, zscore
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.svm import SVR


def features_selected(df, X_new):
    v = 0
    currfeat = []
    mask = X_new.get_support()
    for col in df.columns:
        if v < 8 and mask[v] == True:
            currfeat.append(col)
        v += 1
    return currfeat


def BackwardFeatureSelection(X, Y, df, s, dec):
    print(s)
    if dec is True:
        knn = KNeighborsClassifier(n_neighbors=3)
        sfs = SequentialFeatureSelector(knn, n_features_to_select=5, direction='backward')
    else:
        clf = ExtraTreesRegressor(n_estimators=50)
        sfs = SequentialFeatureSelector(clf, n_features_to_select=5, direction='backward')
    sfs1 = sfs.fit(X, Y)
    print(features_selected(df, sfs1))


def ForwardFeatureSelection(X, Y, df, s, dec):
    print(s)
    if dec is True:
        knn = KNeighborsClassifier(n_neighbors=3)
        sfs = SequentialFeatureSelector(knn, n_features_to_select=5, direction='forward')
    else:
        clf = ExtraTreesRegressor(n_estimators=50)
        sfs = SequentialFeatureSelector(clf, n_features_to_select=5, direction='forward')
    sfs1 = sfs.fit(X, Y)
    print(features_selected(df, sfs1))


def RFESelection(X, Y, df, s):
    print(s)
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=5, step=1)
    X_new = selector.fit(X, Y)
    print(features_selected(df, X_new))


names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
         'DiabetesPredigreeFuntion', 'Age', 'Outcome']
df = pd.read_csv('pima-indians-diabetes.csv', names=names)

# Min-Max Scaling
min_max_scaler = preprocessing.MinMaxScaler()
df_minmax = min_max_scaler.fit_transform(df)
df_minmax = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
df_minmax.head()
print("Min-Max Scaling:")
print(df_minmax.head(5))

# Z-score scaling
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled.head()
print("Z-score scaling:")
print(df_scaled.head(5))

# z = np.abs(stats.zscore(df_scaled))

# MaxAbsScaler
transformer = preprocessing.MaxAbsScaler().fit(df)
df_MaxAbsScaler = transformer.transform(df)
df_MaxAbsScaler = pd.DataFrame(transformer.fit_transform(df), columns=df.columns)
print("MaxAbsScaler:")
print(df_MaxAbsScaler.head(5))

# Defining pca
pca = PCA(n_components=5)

# unscaled pca
pc = pca.fit(df)
print("Unscaled: ", pc.explained_variance_ratio_)

# min-max pca
pca_mn = pca.fit(df_minmax)
print("Min-Max: ", pca_mn.explained_variance_ratio_)

# Z-score PCA
pca_z = pca.fit_transform(df_scaled)

# MaxAbs pca
pca_maxabs = pca.fit_transform(df_MaxAbsScaler)

# Assign X and Y values
X = df.drop('Outcome', axis=1)
Y = df['Outcome']

# X and Y for min-max scaled
Xmn = df_minmax.drop('Outcome', axis=1)
Ynm = df_minmax['Outcome']

# X and Y for z scaled
Xz = df_scaled.drop('Outcome', axis=1)
Yz = df_scaled['Outcome']

# X and Y for MaxAbs scaled
Xabs = df_MaxAbsScaler.drop('Outcome', axis=1)
Yabs = df_MaxAbsScaler['Outcome']

# FFS
print("Forward Feature Selection:")
ForwardFeatureSelection(X, Y, df, s="Unscaled:", dec=True)
ForwardFeatureSelection(Xmn, Ynm, df_minmax, s="Min Max:", dec=True)
ForwardFeatureSelection(Xz, Yz, df_scaled, s="Z-Score", dec=False)
ForwardFeatureSelection(Xabs, Yabs, df_MaxAbsScaler, s="MaxAbs:", dec=True)

# BFS
print("\n\nBackward Feature Selection:")
BackwardFeatureSelection(X, Y, df, s="Unscaled:", dec=True)
BackwardFeatureSelection(Xmn, Ynm, df_minmax, s="Min Max:", dec=True)
BackwardFeatureSelection(Xz, Yz, df_scaled, s= "Z-Score", dec=False)
BackwardFeatureSelection(Xabs, Yabs, df_MaxAbsScaler, s="MaxAbs:", dec=True)

# RFE feature selection
print("\n\nRFE:")
RFESelection(X, Y, df, s="Unscaled:")
RFESelection(Xmn, Ynm, df_minmax, s="Min Max:")
RFESelection(Xz, Yz, df_scaled, s="Z-Score")
RFESelection(Xabs, Yabs, df_MaxAbsScaler, s="MaxAbs:")
