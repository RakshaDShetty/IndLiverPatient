# Indian Liver Patient Analysis using Machine Learning

Liver Disease is one of the most serious health issue everywhere and every individual must have a little understanding about it, symptoms, age factor and so on.
In this project I've used Machine Learning model for the analysis and prediction purpose. 

I have downloaded the dataset from the UCI ML Repository and the link is mentioned below.
(https://www.kaggle.com/uciml/indian-liver-patient-records)


Steps involved are:
1. Data Analysis
2. Data Visualization
3. Feature Selection
4. Predict whether a patient has any liver disease or not.

## Importing the required libraries:
```markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
```
### 1. Data analysis:
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

### 2. Data Visualization:
```
import seaborn as sns
sns.countplot(data=df, x = 'Dataset', label='Count')
LD, NLD = df['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)

Number of patients diagnosed with liver disease:  416
Number of patients not diagnosed with liver disease:  167
```
Download

```
sns.countplot(data=df, x = 'Gender', label='Count')

M, F = df['Gender'].value_counts()
print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)

Number of patients that are male:  441
Number of patients that are female:  142
```
Download 2

```
sns.factorplot(x="Age", y="Gender", hue="Dataset", data=df);
```
Download
Above implies that AGE is an important factor.
```
df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).count().sort_values(by='Dataset', ascending=False)
```
Dataset pic
```
df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).mean().sort_values(by='Dataset', ascending=False)
```
Dataset pic
```
g = sns.FacetGrid(df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');
```
Download

```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")
plt.subplots_adjust(top=0.9)
```
Download 

There seems to be direct relationship between Total_Bilirubin and Direct_Bilirubin. We have the possibility of removing one of this feature.
```
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=df, kind="reg")
```
Download

```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Aspartate_Aminotransferase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)
```
Download

#There is linear relationship between Aspartate_Aminotransferase and Alamine_Aminotransferase and the gender. We have the possibility of removing one of this feature.
```
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=df, kind="reg")
```
download

```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Alkaline_Phosphotase", "Alamine_Aminotransferase",  edgecolor="w")
plt.subplots_adjust(top=0.9)
```
Download

```
sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data=df, kind="reg")
```
Downlaod

No linear correlation between Alkaline_Phosphotase and Alamine_Aminotransferase.

```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")
plt.subplots_adjust(top=0.9)
```
Download

#There is linear relationship between Total_Protiens and Albumin and the gender. We have the possibility of removing one of this feature.
```
sns.jointplot("Total_Protiens", "Albumin", data=df, kind="reg")
```
Dowmload

```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")
plt.subplots_adjust(top=0.9)
```
download

```
sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=df, kind="reg")
```
Download
```
g = sns.FacetGrid(df, col="Gender", row="Dataset", margin_titles=True)
g.map(plt.scatter,"Albumin_and_Globulin_Ratio", "Total_Protiens",  edgecolor="w")
plt.subplots_adjust(top=0.9)
```
Download

Convert the categorical values of gender.
```
pd.get_dummies(df['Gender'], prefix = 'Gender').head()
```
Dataset pic!
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
Download

The above correlation also indicates the following correlation
Total_Protiens & Albumin
Alamine_Aminotransferase & Aspartate_Aminotransferase
Direct_Bilirubin & Total_Bilirubin
There is some correlation between Albumin_and_Globulin_Ratio and Albumin. But its not as high as Total_Protiens & Albumin

### 3.Feature Selection
```
from sklearn.model_selection import train_test_split as tts
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.30, random_state=101)
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

Logistic Regression Training Score: 
 72.5925925925926
Logistic Regression Test Score: 
 70.6896551724138
Coefficient: 
 [[-0.01389644  0.0052153  -0.47259648 -0.00072215 -0.01065968 -0.00503362
  -0.18770568  0.46130938  0.24226435  0.18570231  0.12571321]]
Intercept: 
 [0.33329304]
Accuracy: 
 0.7068965517241379
Confusion Matrix: 
 [[113  11]
 [ 40  10]]
Classification Report: 
               precision    recall  f1-score   support

           1       0.74      0.91      0.82       124
           2       0.48      0.20      0.28        50

    accuracy                           0.71       174
   macro avg       0.61      0.56      0.55       174
weighted avg       0.66      0.71      0.66       174

 ```
download

```
coeff_df = pd.DataFrame(X.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
```
Dataset

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
Linear Regression Score: 12.97
Linear Regression Test Score: 8.71
Coefficient: 
 [-3.19568356e-03 -3.72174104e-04 -1.77536841e-02 -1.75872415e-04
 -3.89111234e-04  5.21731268e-05 -8.12987750e-02  1.73328624e-01
 -7.66122406e-02  2.53152260e-02 -2.53152260e-02]
Intercept: 1.6053571110507452
```
Considering seven important features based on recursive feature elimination
```
finX = df[['Total_Protiens','Albumin', 'Gender_Male']]
finX.head(4)
```
Dataset
```
X_train, X_test, Y_train, Y_test = train_test_split(finX, Y, test_size=0.30, random_state=101)
```
Logistic Regreession
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
Logistic Regression Train Accuracy: 71.35802469135803
Logistic Regression Test Accuracy: 71.26436781609196
Coefficient: [[-0.53473804  1.14349258 -0.32966062]]
Intercept: [-0.87751055]
Accuracy: 0.7126436781609196
Confusion Matrix: [[123   1]
                  [ 49   1]]
Classification Report:
               precision    recall  f1-score   support

           1       0.72      0.99      0.83       124
           2       0.50      0.02      0.04        50

    accuracy                           0.71       174
   macro avg       0.61      0.51      0.43       174
weighted avg       0.65      0.71      0.60       174
```
Download

```
loregaccuracy=logreg.score(X_test,Y_test)
loregaccuracy*100

Output:
71.26436781609196
```
```
#here its comparing to knw how many matches and not
from sklearn.metrics import confusion_matrix
conmat=confusion_matrix(Y_test,log_predicted)
conmat

Output:
array([[123,   1],
       [ 49,   1]], dtype=int64)
```       
