# Kaggle Dataset : Pokemon with Stats

Dataset hyprlink :  https://www.kaggle.com/abcsds/pokemon  

透過不同能力值的分析，希望能藉此預測屬性及神獸屬性。  

## Introduction

欄位：

|column|type|content|
|:--:|:--:|:--:|
|Name|string|寶可夢名字|
|Type 1|string|第一屬性|
|Type 2|string|第二屬性|
|Total|int|加總能力數值|
|Defense|int|防禦|
|Sp. Atk|int|特殊攻擊|
|Sp. Def|int|特殊攻擊|
|Speed|int|敏捷|
|Generation|int|第幾代|
|Legendary|bool|神獸|

程式內容：

```python
data.head(5)
```
![data](https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/data.png)

```python
data.hist(figsize = (20,20))
```

![Subplot](https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/subplot.png)

## 資料前處理  
```python
dummy_type_1 = pd.get_dummies(data['Type 1'])
```
![dummy](https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/type.PNG)


把欄位標準化
```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
columns_list = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
numerical =  pd.DataFrame(sc_X.fit_transform(poke_data_new[columns_list]),
                          columns = columns_list,
                          index = poke_data_new.index)

# numerical
poke_clean_standard = poke_data_new.copy(deep=True)
poke_clean_standard[columns_list] = numerical[columns_list]
poke_clean_standard.head()
```



## 預測 legendary 屬性  

1. 最近鄰居法 : K 選擇從 1 到 15  
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y,random_state = 2,test_size=0.4,stratify=y)

from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []

for i in range(1,15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
```

![](https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/KNN.png)
![](https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/KNN_accuracy.PNG)



2. Random Forest 隨機森林  
```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 5)
clf = clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
clf.score(X_test, y_predict)
```
![](https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/random_forest.PNG)


3. Logistic Regression 邏輯迴歸  

```python
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression

imp = Imputer()
imp.fit(X_train)
imp_train_data = imp.transform(X_train)
imp_test_data = imp.transform(X_test)

lr = LogisticRegression().fit(imp_train_data, y_train)
lr.score(X_test, y_test)
```
![](https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/logistic.PNG)

<hr/>  

## 預測 type 屬性  
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD


model = Sequential()
# Define DNN structure
model.add(Dense(32, input_dim=input_shape, activation='relu'))
model.add(Dense(units=logits, activation='softmax'))
     

model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
model.summary()
```
![](https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/DNN.png)
![](https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/epoch.PNG")
<img src = "https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/DNN01.PNG" width = "420px">
<img src = "https://github.com/ChengYiHuang/Machine-Learning-Course-2019-07/blob/master/fig/DNN02.PNG" width = "420px">

