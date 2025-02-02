import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import time

start_time = time.time()

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



#y=data[["math_score","reading_score","writing_score"]]
y=data[["math_encoded"]]


poly_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.001,
                    fit_inverse_transform=True)
z_poly=poly_pca.fit_transform(x)


plt.figure(figsize=(3,3))
plt.scatter(z_poly[:,0], z_poly[:,1], c=y["math_encoded"])
plt.xlabel("z1")
plt.ylabel("z2")
plt.show()

xp_poly=poly_pca.inverse_transform(z_poly)
fig=plt.figure(figsize=(4,4))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(xp_poly[:,0],xp_poly[:,1],xp_poly[:,2],c=y["math_encoded"])

ax.set_xlabel('$x_1$',fontsize=18)
ax.set_ylabel('$x_2$',fontsize=18)
ax.set_zlabel('$x_3$',fontsize=18)
plt.show()




















