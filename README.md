# EX-8 : IMPLEMENTATION OF DECISIONO TREE CLASSIFIER MODEL FOR PREDICTING EMPLOYEE CHURN

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## EQUIPMENTS REQUIRED:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM:
1.Load and explore the employee dataset using pandas.

2.Encode categorical data (like salary) using LabelEncoder.

3.Define features (X) and target (y), then split the data.

4.Train a Decision Tree Classifier using the training set.

5.Predict and evaluate accuracy, then use the model for new predictions.
## PROGRAM:
```
import pandas as pd
data = pd.read_csv("Employee.csv")
print("HEAD DATA:")
print(data.head())
print("DATA INFO")
print(data.info())
print("NULL DATA SET")
print(data.isnull().sum())
print('VALUE COUNT ON LEFT:')
print(data['left'].value_counts())
​
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
​
data['salary'] = le.fit_transform(data['salary'])
print("DATASET TRANSFORM HEAD:")
print(data.head())
​
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
print("X HEAD:")
print(x.head())
​
y=data['left']
​
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
​
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
​
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
print("ACCURACY:")
print(accuracy)
print("DATA PRECICTION:")
print(dt.predict([[0.5,0.8,9,260,6,0,1,2]]))
​

```


## OUTPUT:

### Dataset :
![image](https://github.com/user-attachments/assets/5002c999-7519-4ead-b545-e7ffab6e613f)

### Null Dataset :
![image](https://github.com/user-attachments/assets/560d1a8f-93b8-4842-8fb6-7a68967c5715)

### Value Count in Left Coloumn :
![image](https://github.com/user-attachments/assets/d8723bdf-86df-44e6-bc95-a384bc0f19a8)

### Dataset Transfered Head :
![image](https://github.com/user-attachments/assets/4602f03a-0d71-4de3-b180-f4f7d8002073)

### X_Head :
![image](https://github.com/user-attachments/assets/03bffc9b-8162-4456-a7c2-43b5f48f851e)

### Accuracy :
![image](https://github.com/user-attachments/assets/36f4c375-82d8-4458-be25-d794da5a933b)

### Data Prediction :
![image](https://github.com/user-attachments/assets/66137079-d1c9-4327-b041-89ce6c54a7fc)








## RESULT:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
