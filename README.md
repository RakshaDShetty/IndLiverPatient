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



### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/RakshaDShetty/IndLiverPatient/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
