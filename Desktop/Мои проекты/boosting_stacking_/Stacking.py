from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

titanic = pd.read_csv('titanic.csv')
targets = titanic.Survived
data = titanic.drop(columns='Survived')

x_train, x_test, y_train, y_test = train_test_split(data,
                                                    targets,
                                                    train_size=0.8,
                                                    random_state=0)

train, valid, train_true, valid_true = train_test_split(x_train,
                                                        y_train,
                                                        train_size=0.5,
                                                        random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn_model = knn.fit(train, train_true)

lr = LogisticRegression(random_state=17)
lr_model = lr.fit(train, train_true)

dtc = DecisionTreeClassifier(max_leaf_nodes=4, random_state=17)
dtc_model = dtc.fit(train, train_true)

svc = SVC(random_state=17)
svc_model = svc.fit(train, train_true)

models = [knn_model, lr_model, dtc_model, svc_model]
meta_mtrx = np.empty((valid.shape[0], len(models)))  # (кол-во объектов, 4 алгоритма)

for n, model in enumerate(models):
    meta_mtrx[:, n] = model.predict(valid)
    predicted = model.predict(x_test)
    print(f'{n} auc: {roc_auc_score(y_test, predicted)}')

meta = XGBClassifier(n_estimators=40)
meta_model = meta.fit(meta_mtrx, valid_true)

meta_mtrx_test = np.empty((x_test.shape[0], len(models)))

for n, model in enumerate(models):
    meta_mtrx_test[:, n] = model.predict(x_test)

meta_predict = meta.predict(meta_mtrx_test)
print(f'Stacking AUC: {roc_auc_score(y_test, meta_predict)}')

models = [knn_model, svc_model]
meta_mtrx = np.empty((valid.shape[0], len(models)))  # (кол-во объектов, 4 алгоритма)

for n, model in enumerate(models):
    meta_mtrx[:, n] = model.predict(valid)
    predicted = model.predict(x_test)
    print(f'{n} auc: {roc_auc_score(y_test, predicted)}')

meta_model = meta.fit(meta_mtrx, valid_true)
meta_mtrx_test = np.empty((x_test.shape[0], len(models)))
for n, model in enumerate(models):
    meta_mtrx_test[:, n] = model.predict(x_test)
meta_predict = meta.predict(meta_mtrx_test)
print(f'Stacking AUC: {roc_auc_score(y_test, meta_predict)}')