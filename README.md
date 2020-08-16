# Indian Liver Patient Analysis using Machine Learning

Liver Disease is one of the most serious health issue everywhere and every individual must have a little understanding about it, symptoms, age factor and so on.
In this project I've used Machine Learning model for the analysis and prediction purpose. 

I have downloaded the dataset from the UCI ML Repository and the link is mentioned below.
(https://www.kaggle.com/uciml/indian-liver-patient-records)


Steps involved are:
1. Data Analysis
2. Data Visualization
3. Feature Selection
4. Prediction

### Importing the required libraries:
```markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
```
## 1. Data analysis:
```markdown
df=pd.read_csv('C:\\Users\\raksh\\Desktop\\dlithe-project\\indian_liver_patient.csv')
```
Data Cleaning
Lets look for any missing value in our dataset.
```
df.isnull().sum()


Age                           0
Gender                        0
Total_Bilirubin               0
Direct_Bilirubin              0
Alkaline_Phosphotase          0
Alamine_Aminotransferase      0
Aspartate_Aminotransferase    0
Total_Protiens                0
Albumin                       0
Albumin_and_Globulin_Ratio    4
Dataset                       0
dtype: int64
```

We see that we have 4 missing values for Albumin_and_Globulin_Ratio
to check for rows having missing values
```
print(df[df.Albumin_and_Globulin_Ratio.isnull()])

     Age  Gender  Total_Bilirubin  Direct_Bilirubin  Alkaline_Phosphotase  \
209   45  Female              0.9               0.3                   189   
241   51    Male              0.8               0.2                   230   
253   35  Female              0.6               0.2                   180   
312   27    Male              1.3               0.6                   106   

     Alamine_Aminotransferase  Aspartate_Aminotransferase  Total_Protiens  \
209                        23                          33             6.6   
241                        24                          46             6.5   
253                        12                          15             5.2   
312                        25                          54             8.5   

     Albumin  Albumin_and_Globulin_Ratio  Dataset  
209      3.9                         NaN        1  
241      3.1                         NaN        1  
253      2.7                         NaN        2  
312      4.8                         NaN        2  

```
Since only 4 rows have missing values, we can delete these rows
```
print(df.dropna(how='any', inplace=True))
```

## 2. Data Visualization:
```
import seaborn as sns
sns.countplot(data=df, x = 'Dataset', label='Count')
LD, NLD = df['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)

Number of patients diagnosed with liver disease:  416
Number of patients not diagnosed with liver disease:  167
```
![](images/download%20(0).png)

```
sns.countplot(data=df, x = 'Gender', label='Count')

M, F = df['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)

Number of patients that are male:  441
Number of patients that are female:  142
```
![](images/download%20(1).png)

```
sns.factorplot(x="Age", y="Gender", hue="Dataset", data=df);
```
![](images/download%20(2).png)

Above implies that AGE is an important factor.
```
df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).count().sort_values(by='Dataset', ascending=False)
```
![](images/Screenshot%201.png)
```
df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).mean().sort_values(by='Dataset', ascending=False)
```
![](images/Screenshot%202.png)
```
g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');
```
![](images/download%20(3).png)

```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")
plt.subplots_adjust(top=0.9)
```
![](images/download%20(5).png)

There seems to be direct relationship between Total_Bilirubin and Direct_Bilirubin. We have the possibility of removing one of this feature.
```
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=df, kind="reg")
```
![](images/download%20(6).png)

```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Aspartate_Aminotransferase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)
```
![](images/download%20(7).png)

#There is linear relationship between Aspartate_Aminotransferase and Alamine_Aminotransferase and the gender. We have the possibility of removing one of this feature.
```
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=df, kind="reg")
```
![](images/download%20(8).png)

```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Alkaline_Phosphotase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)
```
![](images/download%20(9).png)

```
sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data=df, kind="reg")
```
![](images/download%20(10).png)

No linear correlation between Alkaline_Phosphotase and Alamine_Aminotransferase.

```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")
plt.subplots_adjust(top=0.9)
```
![](images/download%20(11).png)

#There is linear relationship between Total_Protiens and Albumin and the gender. We have the possibility of removing one of this feature.
```
sns.jointplot("Total_Protiens", "Albumin", data=df, kind="reg")
```
![](images/download%20(12).png)

```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")
plt.subplots_adjust(top=0.9)
```
![](images/download%20(13).png)

```
sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=df, kind="reg")
```
![](images/download%20(14).png)
```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin_and_Globulin_Ratio", "Total_Protiens",  edgecolor="w")
plt.subplots_adjust(top=0.9)
```
![](images/download%20(15).png)

Convert the categorical values of gender.
```
pd.get_dummies(df['Gender'], prefix = 'Gender').head()
```
![](images/Screenshot%20(3).png)
```
df = pd.concat([df,pd.get_dummies(df['Gender'], prefix = 'Gender')], axis=1)
```
The input variables/features are all the inputs except Dataset. The prediction or label is 'Dataset' that determines whether the patient has liver disease or not. 
```
X = df.drop(['Gender','Dataset'], axis=1)
X.head(3)
```

```
 Y= df['Dataset'] # 1 for liver disease; 2 for no liver disease
 ```
 Correlation
 ```
 corr = X.corr()
plt.figure(figsize=(30, 30))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'coolwarm')
plt.title('Correlation between features');
```
![](images/download%20(16).png)

The above correlation also indicates the following correlation
Total_Protiens & Albumin
Alamine_Aminotransferase & Aspartate_Aminotransferase
Direct_Bilirubin & Total_Bilirubin
There is some correlation between Albumin_and_Globulin_Ratio and Albumin. But its not as high as Total_Protiens & Albumin

## 3.Feature Selection
```
from sklearn.model_selection import train_test_split as tts
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.17, random_state=299)
print (X_train.shape)
print (Y_train.shape)
print (X_test.shape)
print (Y_test.shape)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
#Logistic Regression
#Create an object
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, Y_train)
#Predict Output
log_predicted= logreg.predict(X_test)

logreg_score = logreg.score(X_train, Y_train)*100
logreg_score_test = logreg.score(X_test, Y_test)*100
#Equation coefficient and Intercept
print('Logistic Regression Training Score: \n', logreg_score)
print('Logistic Regression Test Score: \n', logreg_score_test)
print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)
print('Accuracy: \n', accuracy_score(Y_test,log_predicted))
print('Confusion Matrix: \n', confusion_matrix(Y_test,log_predicted))
print('Classification Report: \n', classification_report(Y_test,log_predicted))

sns.heatmap(confusion_matrix(Y_test,log_predicted),annot=True,fmt="d")

Output:
Logistic Regression Training Score: 
 70.0
Logistic Regression Test Score: 
 81.81818181818183
Coefficient: 
 [[-0.01112551 -0.05348263 -0.41995266 -0.00064794 -0.00956595 -0.00199349
  -0.17603323  0.33754815  0.33522491  0.22843094  0.13535704]]
Intercept: 
 [0.38090827]
Accuracy: 
 0.8181818181818182
Confusion Matrix: 
 [[75  7]
 [11  6]]
Classification Report: 
               precision    recall  f1-score   support

           1       0.87      0.91      0.89        82
           2       0.46      0.35      0.40        17

    accuracy                           0.82        99
   macro avg       0.67      0.63      0.65        99
weighted avg       0.80      0.82      0.81        99


 ```
![](images/new(1).png)

```
coeff_df = pd.DataFrame(X.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
```
![](images/Screenshot%20(4).png)

Linear Regression
```
from sklearn import linear_model
linear = linear_model.LinearRegression()
# Train the model using the training sets and check score
linear.fit(X_train, Y_train)
#Predict Output
lin_predicted = linear.predict(X_test)
```
```
linear_score = round(linear.score(X_train, Y_train) * 100, 2)
linear_score_test = round(linear.score(X_test, Y_test) * 100, 2)
#Equation coefficient and Intercept
print('Linear Regression Score: \n', linear_score)
print('Linear Regression Test Score: \n', linear_score_test)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

from sklearn.feature_selection import RFE
rfe =RFE(linear, n_features_to_select=3)
rfe.fit(X,Y)

Output:
Linear Regression Score: 
 11.39
Linear Regression Test Score: 
 3.62
Coefficient: 
 [-2.80721776e-03 -6.32097600e-04 -2.32713782e-02 -1.76474075e-04
 -4.03931935e-04  7.33243868e-05 -7.33226344e-02  1.28007261e-01
 -4.34929567e-02  2.24880063e-02 -2.24880063e-02]
Intercept: 
 1.6712228628389874

```
Considering seven important features based on recursive feature elimination
```
finX = df[['Total_Protiens','Albumin', 'Gender_Male']]
finX.head(4)
```
![](images/Screenshot%20(5).png)

Logistic Regression
```
#Logistic Regression
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, Y_train)
#Predict Output
log_predicted= logreg.predict(X_test)

logreg_score = logreg.score(X_train, Y_train) * 100
logreg_score_test = logreg.score(X_test, Y_test) * 100
#Equation coefficient and Intercept
print('Logistic Regression Train Accuracy:', logreg_score)
print('Logistic Regression Test Accuracy:', logreg_score_test)
print('Coefficient:', logreg.coef_)
print('Intercept:', logreg.intercept_)
print('Accuracy:', accuracy_score(Y_test,log_predicted))

#compare right vs wrong predictions
#here its comparing to knw how many matches and not
conmat=confusion_matrix(Y_test,log_predicted)
print('Confusion Matrix:',conmat)

print('Classification Report:\n', classification_report(Y_test,log_predicted))

sns.heatmap(confusion_matrix(Y_test,log_predicted),annot=True,fmt="d")

Output:
Logistic Regression Train Accuracy: 70.0
Logistic Regression Test Accuracy: 81.81818181818183
Coefficient: [[-0.01112551 -0.05348263 -0.41995266 -0.00064794 -0.00956595 -0.00199349
  -0.17603323  0.33754815  0.33522491  0.22843094  0.13535704]]
Intercept: [0.38090827]
Accuracy: 0.8181818181818182
Confusion Matrix: [[75  7]
 [11  6]]
Classification Report:
               precision    recall  f1-score   support

           1       0.87      0.91      0.89        82
           2       0.46      0.35      0.40        17

    accuracy                           0.82        99
   macro avg       0.67      0.63      0.65        99
weighted avg       0.80      0.82      0.81        99

```
![](images/new%20(2).png)

```
loregaccuracy=logreg.score(X_test,Y_test)
loregaccuracy*100

Output:
81.81818181818183
```
K-Nearest Neighbors
```
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_test,Y_test)
knnpred=knn.predict(X_test)
knnacc=knn.score(X_test,Y_test)
knnacc*100

Output:
83.83838383838383
```
Decision Tree classifier
```
from sklearn.tree import DecisionTreeClassifier
DTree= DecisionTreeClassifier()
DTree.fit(X_train,Y_train)
DTreePred=DTree.predict(X_test)
DTreeAccu=DTree.score(X_test,Y_test)
DTreeAccu*100

Output:
72.72727272727273
```
Random forest Classifier
```
from sklearn.ensemble import RandomForestClassifier
RDF=RandomForestClassifier(random_state=142)
RDF.fit(X_train,Y_train)
RDFPred=RDF.predict(X_test)
RDFAccu=RDF.score(X_test,Y_test)
RDFAccu*100

Output:
82.82828282828282
```

## Prediction
From the above Model Prediction we conclude that Logistic Regression(78.45) and KNN(79.31) have highest Accuracy.



```
#here its comparing to knw how many matches and not 
from sklearn.metrics import confusion_matrix
conmat=confusion_matrix(Y_test,log_predicted)
conmat

Output:
array([[75,  7],
       [11,  6]], dtype=int64)

from sklearn.metrics import confusion_matrix
knnmat=confusion_matrix(Y_test,knnpred)
knnmat

Output:
array([[77,  5],
       [11,  6]], dtype=int64)
       
```       
