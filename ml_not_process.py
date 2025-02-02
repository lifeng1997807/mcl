# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 15:33:08 2024

@author: benker
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

'''輸入資料
'''
data = pd.read_csv('C:\\Users\\benker\\Downloads\\archive\\study_performance.csv')

data['math_encoded'] = data['math_score'].apply(lambda x: 1 if x >= 60 else 0)

#label encoding
data["parental_level_of_education"]=data["parental_level_of_education"].map({"bachelor's degree" : 4, "some college":2, "master's degree":5, 
        "associate's degree":3, "high school":1, "some high school":0})

data["gender"]=data["gender"].map({"male" : 1,"female":0})

data["lunch"]=data["lunch"].map({"standard" : 1,"free/reduced":0})

data["test_preparation_course"]=data["test_preparation_course"].map({"completed" : 1,"none":0})


#onehot encoding
onehot_encoder=OneHotEncoder()
onehot_encoder.fit(data[["race_ethnicity"]])
city_encoded=onehot_encoder.transform(data[["race_ethnicity"]]).toarray()
data[["group A","group B","group C","group D","group E"]]=city_encoded
data=data.drop(["race_ethnicity"],axis=1)

#檢查點
#print(data.head(10))
#print(data[["group A","group B","group C","group D","group E"]].head(10))
#print(data[["test_preparation_course"]].head(10))
#print(data[["parental_level_of_education"]].head(10))

x=data[["gender","parental_level_of_education","lunch","test_preparation_course",
        "group A","group B","group C","group D","group E","reading_score",
        "writing_score"]]
scaler = StandardScaler()
x = scaler.fit_transform(x)
#y=data[["math_score","reading_score","writing_score"]]
y=data[["math_encoded"]]
# 将 DataFrame 轉换为 numpy 数组
array_2d = y.to_numpy()
'''
#檢查缺失值
missing_data = np.isnan(x)
print(missing_data)
missing_data = np.isnan(y)
print(missing_data)
missing_count = np.isnan(x).sum()
print(missing_count)
missing_count = np.isnan(y).sum()
print(missing_count)
'''

y = array_2d.flatten()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=64)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 1, 3, 5],
    'min_samples_split': [2, 4, 6],

}
rf = RandomForestClassifier(random_state=64)
# 使用 GridSearchCV 
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# 输出最佳参數
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# 使用最佳參數
best_rf = grid_search.best_estimator_
rf_pred = best_rf.predict(x_test)

# 分類报告


print("最佳随機森林準確率:", accuracy_score(y_test, rf_pred))
print("最佳随機森林分類報告:\n", classification_report(y_test, rf_pred))

from sklearn.model_selection import learning_curve


# 計算学學習曲線
train_sizes, train_scores, test_scores = learning_curve(
    best_rf, x_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 20)
)

# 計算訓練和測試分数的均值
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

# 繪製學習曲線
plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation score')
plt.title('rf Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()   
    
#svm
# 創建Pipeline
poly_svm_clf = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(max_iter=100000, C=1, loss='hinge', random_state=64, dual=True))
])

# 訓練模型
poly_svm_clf.fit(x_train, y_train)
y_pred = poly_svm_clf.predict(x_test)

# 計算并打印準確率
accuracy = accuracy_score(y_test, y_pred)
print("多項式核SVM準確率:", accuracy)

# 打印詳细的分類報告
report = classification_report(y_test, y_pred)
print("多項式核SVM分類報告:")
print(report)

# 计算交叉驗證得分
cv_scores = cross_val_score(poly_svm_clf, x_train, y_train, cv=5)

# 输出交叉驗證分数
print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean()}')

# 繪製學習曲線
train_sizes, train_scores, test_scores = learning_curve(
    poly_svm_clf, x_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.01, 1.0, 20)
)

# 計算訓練和測試分数的均值和標準差
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

# 繪製學習曲線
plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation score')
plt.title('svm Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()  
#KNN
error_rate = []

for i in range(1,6):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(x_train,y_train)
  pred_i = knn.predict(x_test)
  error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1,6),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

min_index = error_rate.index(min(error_rate))

#從k=1開始測試
knn = KNeighborsClassifier(n_neighbors=min_index+1)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

# 计算並打印準確率
accuracy = accuracy_score(y_test, y_pred)
print("KNN率:", accuracy)

# 打印詳细的分類報告
report = classification_report(y_test, y_pred)
print("KNN分告:")
print(report)

# 計算交叉驗證得分
cv_scores = cross_val_score(knn, x_train, y_train, cv=5)

# 输出交叉驗證分数
print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean()}')


train_sizes, train_scores, test_scores = learning_curve(
    knn, x_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.01, 1.0, 20)
)


train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)


plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation score')
plt.title('knn Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()  
#DT
# 創建決策樹
dt = DecisionTreeClassifier(random_state=64)
# 设置網格
param_grid = {
    'max_depth': [3, 6, 9],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
}

# 使用 GridSearchCV 
grid_search = GridSearchCV(dt, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)



# 使用最佳模型
best_dt = grid_search.best_estimator_
dt_pred = best_dt.predict(x_test)

print("最佳决策樹準確率:", accuracy_score(y_test, dt_pred))
print("最佳决策分類报告:\n", classification_report(y_test, dt_pred))


cv_scores = cross_val_score(best_dt, x_train, y_train, cv=5)


print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean()}')


train_sizes, train_scores, test_scores = learning_curve(
    best_dt, x_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.01, 1.0, 20)
)


train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation score')
plt.title('dt Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()  

