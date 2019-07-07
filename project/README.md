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

## 課堂回饋  

- `著重於特定主題` : 涵蓋範圍太廣，需集中於特定主題，尚能學得更精實。  

由於模組化課程時間不長，因此要在短時間內將所有機器學習內容講述過一次，對老師與學生們都是極大的負擔 (預設這門課聽講的學生對此領域尚未了解透徹)，
縱然分類與分群之間的關係的確不宜分成兩門獨立課程，但在學習上，感覺也不太適合在短時間內全數傾囊相授
(因為它們彼此之間還是屬於可能會在學習產生逆向干擾的情況)，但還是希望模組化課程能將主題們再細分些，或者是規畫較為初階的課程
(ex: 資料視覺化課程、資料前處理)。

- `公布所需先備知識` : 天數太少，作業難度太高，沒有程式基礎者感到相當痛苦。  

課程當中感受到最主要的問題還是作業實作日期過短，一般程式基礎的人對於此課程一開始感受到的作業壓力蠻大的，且在課程後期所教授的內容又屬更進階，
因此對於沒有先備知識的人對於學習過程當中，會感受到極大的挫折。

- `課程時數加長` : 希望課程時數可以增加，才能涵蓋更多的範圍，學習到更多機器學習的精神。 

四天的課程能深刻體會機器學習的深 (數學模型) 與廣 (應用在多方面)，但因為課程還要顧及上機時間，因此對於內部細節講解過於匆促，
假若未來能有類似的課程，希望課程時數也許能被拉長至兩個禮拜，學習的步調能被放緩，讓學習的內容更加充實。  
