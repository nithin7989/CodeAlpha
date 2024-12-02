import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

/kaggle/input/loan-data-set/loan_train.csv
/kaggle/input/loan-data-set/loan_test.csv

df1=pd.read_csv('/kaggle/input/loan-data-set/loan_train.csv')
df2=pd.read_csv('/kaggle/input/loan-data-set/loan_train.csv')

df1.head()

	Gender 	Married 	Dependents 	Education 	Self_Employed 	Applicant_Income 	Coapplicant_Income 	Loan_Amount 	Term 	Credit_History 	Area 	Status
0 	Male 	No 	0 	Graduate 	No 	584900 	0.0 	15000000 	360.0 	1.0 	Urban 	Y
1 	Male 	Yes 	1 	Graduate 	No 	458300 	150800.0 	12800000 	360.0 	1.0 	Rural 	N
2 	Male 	Yes 	0 	Graduate 	Yes 	300000 	0.0 	6600000 	360.0 	1.0 	Urban 	Y
3 	Male 	Yes 	0 	Not Graduate 	No 	258300 	235800.0 	12000000 	360.0 	1.0 	Urban 	Y
4 	Male 	No 	0 	Graduate 	No 	600000 	0.0 	14100000 	360.0 	1.0 	Urban 	Y

df1.shape

(614, 12)

df1.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 614 entries, 0 to 613
Data columns (total 12 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Gender              601 non-null    object 
 1   Married             611 non-null    object 
 2   Dependents          599 non-null    object 
 3   Education           614 non-null    object 
 4   Self_Employed       582 non-null    object 
 5   Applicant_Income    614 non-null    int64  
 6   Coapplicant_Income  614 non-null    float64
 7   Loan_Amount         614 non-null    int64  
 8   Term                600 non-null    float64
 9   Credit_History      564 non-null    float64
 10  Area                614 non-null    object 
 11  Status              614 non-null    object 
dtypes: float64(3), int64(2), object(7)
memory usage: 57.7+ KB

df1.describe(include='all')

	Gender 	Married 	Dependents 	Education 	Self_Employed 	Applicant_Income 	Coapplicant_Income 	Loan_Amount 	Term 	Credit_History 	Area 	Status
count 	601 	611 	599 	614 	582 	6.140000e+02 	6.140000e+02 	6.140000e+02 	600.00000 	564.000000 	614 	614
unique 	2 	2 	4 	2 	2 	NaN 	NaN 	NaN 	NaN 	NaN 	3 	2
top 	Male 	Yes 	0 	Graduate 	No 	NaN 	NaN 	NaN 	NaN 	NaN 	Semiurban 	Y
freq 	489 	398 	345 	480 	500 	NaN 	NaN 	NaN 	NaN 	NaN 	233 	422
mean 	NaN 	NaN 	NaN 	NaN 	NaN 	5.403459e+05 	1.621246e+05 	1.414104e+07 	342.00000 	0.842199 	NaN 	NaN
std 	NaN 	NaN 	NaN 	NaN 	NaN 	6.109042e+05 	2.926248e+05 	8.815682e+06 	65.12041 	0.364878 	NaN 	NaN
min 	NaN 	NaN 	NaN 	NaN 	NaN 	1.500000e+04 	0.000000e+00 	0.000000e+00 	12.00000 	0.000000 	NaN 	NaN
25% 	NaN 	NaN 	NaN 	NaN 	NaN 	2.877500e+05 	0.000000e+00 	9.800000e+06 	360.00000 	1.000000 	NaN 	NaN
50% 	NaN 	NaN 	NaN 	NaN 	NaN 	3.812500e+05 	1.188500e+05 	1.250000e+07 	360.00000 	1.000000 	NaN 	NaN
75% 	NaN 	NaN 	NaN 	NaN 	NaN 	5.795000e+05 	2.297250e+05 	1.647500e+07 	360.00000 	1.000000 	NaN 	NaN
max 	NaN 	NaN 	NaN 	NaN 	NaN 	8.100000e+06 	4.166700e+06 	7.000000e+07 	480.00000 	1.000000 	NaN 	NaN

df1.isnull().sum()

Gender                13
Married                3
Dependents            15
Education              0
Self_Employed         32
Applicant_Income       0
Coapplicant_Income     0
Loan_Amount            0
Term                  14
Credit_History        50
Area                   0
Status                 0
dtype: int64

#I chose the mode (the most frequent value) to replace the missing values
mode_valeur = df1['Gender'].mode()[0]
df1['Gender'].fillna(mode_valeur, inplace=True)

mode_valeur = df1['Married'].mode()[0]
df1['Married'].fillna(mode_valeur, inplace=True)

mode_valeur = df1['Dependents'].mode()[0]
df1['Dependents'].fillna(mode_valeur, inplace=True)

mode_valeur = df1['Education'].mode()[0]
df1['Education'].fillna(mode_valeur, inplace=True)

mode_valeur = df1['Self_Employed'].mode()[0]
df1['Self_Employed'].fillna(mode_valeur, inplace=True)

mode_valeur = df1['Area'].mode()[0]
df1['Area'].fillna(mode_valeur, inplace=True)

mode_valeur = df1['Status'].mode()[0]
df1['Status'].fillna(mode_valeur, inplace=True)

/tmp/ipykernel_33/2039257962.py:3: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df1['Gender'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2039257962.py:6: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df1['Married'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2039257962.py:9: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df1['Dependents'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2039257962.py:12: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df1['Education'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2039257962.py:15: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df1['Self_Employed'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2039257962.py:18: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df1['Area'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2039257962.py:21: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df1['Status'].fillna(mode_valeur, inplace=True)

# Same for DF2
mode_valeur = df2['Gender'].mode()[0]
df2['Gender'].fillna(mode_valeur, inplace=True)

mode_valeur = df2['Married'].mode()[0]
df2['Married'].fillna(mode_valeur, inplace=True)

mode_valeur = df2['Dependents'].mode()[0]
df2['Dependents'].fillna(mode_valeur, inplace=True)

mode_valeur = df2['Education'].mode()[0]
df2['Education'].fillna(mode_valeur, inplace=True)

mode_valeur = df2['Self_Employed'].mode()[0]
df2['Self_Employed'].fillna(mode_valeur, inplace=True)

mode_valeur = df2['Area'].mode()[0]
df2['Area'].fillna(mode_valeur, inplace=True)

mode_valeur = df2['Status'].mode()[0]
df2['Status'].fillna(mode_valeur, inplace=True)

/tmp/ipykernel_33/2950636577.py:3: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df2['Gender'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2950636577.py:6: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df2['Married'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2950636577.py:9: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df2['Dependents'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2950636577.py:12: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df2['Education'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2950636577.py:15: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df2['Self_Employed'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2950636577.py:18: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df2['Area'].fillna(mode_valeur, inplace=True)
/tmp/ipykernel_33/2950636577.py:21: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df2['Status'].fillna(mode_valeur, inplace=True)

df2.isnull().sum()

Gender                 0
Married                0
Dependents             0
Education              0
Self_Employed          0
Applicant_Income       0
Coapplicant_Income     0
Loan_Amount            0
Term                  14
Credit_History        50
Area                   0
Status                 0
dtype: int64

# I chose the median to replace the missing numerical values.
mediane = df1['Term'].median()
df1['Term'].fillna(mediane, inplace=True)

mediane = df1['Credit_History'].median()
df1['Credit_History'].fillna(mediane, inplace=True)
##########################""
mediane = df2['Term'].median()
df2['Term'].fillna(mediane, inplace=True)

mediane = df2['Credit_History'].median()
df2['Credit_History'].fillna(mediane, inplace=True)

/tmp/ipykernel_33/2092338627.py:3: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df1['Term'].fillna(mediane, inplace=True)
/tmp/ipykernel_33/2092338627.py:6: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df1['Credit_History'].fillna(mediane, inplace=True)
/tmp/ipykernel_33/2092338627.py:9: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df2['Term'].fillna(mediane, inplace=True)
/tmp/ipykernel_33/2092338627.py:12: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df2['Credit_History'].fillna(mediane, inplace=True)

df2.isnull().sum()

Gender                0
Married               0
Dependents            0
Education             0
Self_Employed         0
Applicant_Income      0
Coapplicant_Income    0
Loan_Amount           0
Term                  0
Credit_History        0
Area                  0
Status                0
dtype: int64

# Etude sur la correlation 
noms_colonnes = df1.columns.tolist()
noms_colonnes

['Gender',
 'Married',
 'Dependents',
 'Education',
 'Self_Employed',
 'Applicant_Income',
 'Coapplicant_Income',
 'Loan_Amount',
 'Term',
 'Credit_History',
 'Area',
 'Status']

df1.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 614 entries, 0 to 613
Data columns (total 12 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Gender              614 non-null    object 
 1   Married             614 non-null    object 
 2   Dependents          614 non-null    object 
 3   Education           614 non-null    object 
 4   Self_Employed       614 non-null    object 
 5   Applicant_Income    614 non-null    int64  
 6   Coapplicant_Income  614 non-null    float64
 7   Loan_Amount         614 non-null    int64  
 8   Term                614 non-null    float64
 9   Credit_History      614 non-null    float64
 10  Area                614 non-null    object 
 11  Status              614 non-null    object 
dtypes: float64(3), int64(2), object(7)
memory usage: 57.7+ KB

df1[['Coapplicant_Income','Applicant_Income','Loan_Amount','Term','Credit_History']].corr()

	Coapplicant_Income 	Applicant_Income 	Loan_Amount 	Term 	Credit_History
Coapplicant_Income 	1.000000 	-0.116605 	0.189237 	-0.059383 	0.011134
Applicant_Income 	-0.116605 	1.000000 	0.539615 	-0.046531 	-0.018615
Loan_Amount 	0.189237 	0.539615 	1.000000 	0.039440 	0.006015
Term 	-0.059383 	-0.046531 	0.039440 	1.000000 	-0.004705
Credit_History 	0.011134 	-0.018615 	0.006015 	-0.004705 	1.000000

    Pour Coapplicant_Income et Applicant_Income, la corrélation est de -0.116605, ce qui indique une faible corrélation négative entre les deux.

    Entre Coapplicant_Income et Loan_Amount, la corrélation est de 0.189237, ce qui suggère une corrélation positive faible à modérée.

    Pour Applicant_Income et Loan_Amount, la corrélation est de 0.539615, ce qui indique une corrélation positive modérée.

    Entre Loan_Amount et Term, la corrélation est de 0.039440, ce qui indique une corrélation positive très faible ........

TARGET = "Status"

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

label_encoder = LabelEncoder()
ordinal_encoder = OrdinalEncoder()

label_columns_train = ["Gender", "Married", "Education", "Self_Employed", "Area", "Status"]
label_columns_test = [column for column in label_columns_train if column != TARGET]
ordinal_columns = ["Dependents"]

df1[label_columns_train] = df1[label_columns_train].apply(label_encoder.fit_transform)
df2[label_columns_test] = df2[label_columns_test].apply(label_encoder.fit_transform)

df1[ordinal_columns] = ordinal_encoder.fit_transform(df1[ordinal_columns])
df2[ordinal_columns] = ordinal_encoder.fit_transform(df2[ordinal_columns])

X = df1.drop(columns=[TARGET], axis=1)
y = df1[TARGET]

modéle 1 : Random Foarest Classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Créer un modèle de forêt aléatoire avec un générateur de nombres aléatoires fixe
model = RandomForestClassifier(random_state=1)

# Entraîner le modèle
model.fit(X_train, y_train)

# Obtenir les prédictions du modèle
y_pred = model.predict(X_test)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

              precision    recall  f1-score   support

           0       0.75      0.42      0.54        43
           1       0.75      0.93      0.83        80

    accuracy                           0.75       123
   macro avg       0.75      0.67      0.68       123
weighted avg       0.75      0.75      0.73       123

Accuracy: 0.7479674796747967

Modéle 2: GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle de Gradient Boosting Classifier
model = GradientBoostingClassifier(random_state=1)

# Entraîner le modèle
model.fit(X_train, y_train)

# Obtenir les prédictions du modèle
y_pred = model.predict(X_test)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

              precision    recall  f1-score   support

           0       0.75      0.42      0.54        43
           1       0.75      0.93      0.83        80

    accuracy                           0.75       123
   macro avg       0.75      0.67      0.68       123
weighted avg       0.75      0.75      0.73       123

Accuracy: 0.7479674796747967

Modéle 3 : svc

from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle de Support Vector Machine
model = SVC(random_state=1)

# Entraîner le modèle
model.fit(X_train, y_train)

# Obtenir les prédictions du modèle
y_pred = model.predict(X_test)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

              precision    recall  f1-score   support

           0       0.00      0.00      0.00        43
           1       0.65      1.00      0.79        80

    accuracy                           0.65       123
   macro avg       0.33      0.50      0.39       123
weighted avg       0.42      0.65      0.51       123

Accuracy: 0.6504065040650406

/opt/conda/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/opt/conda/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/opt/conda/lib/python3.10/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))

Modéle 4 LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle
model.fit(X_train, y_train)

# Obtenir les prédictions du modèle
y_pred = model.predict(X_test)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

              precision    recall  f1-score   support

           0       0.50      0.07      0.12        43
           1       0.66      0.96      0.78        80

    accuracy                           0.65       123
   macro avg       0.58      0.52      0.45       123
weighted avg       0.60      0.65      0.55       123

Accuracy: 0.6504065040650406

Modéle 5 : KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle KNN
model = KNeighborsClassifier()

# Entraîner le modèle
model.fit(X_train, y_train)

# Obtenir les prédictions du modèle
y_pred = model.predict(X_test)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

              precision    recall  f1-score   support

           0       0.38      0.23      0.29        43
           1       0.66      0.80      0.72        80

    accuracy                           0.60       123
   macro avg       0.52      0.52      0.51       123
weighted avg       0.56      0.60      0.57       123

Accuracy: 0.6016260162601627

Modéle 6: GaussianNB

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle Naive Bayes
model = GaussianNB()

# Entraîner le modèle
model.fit(X_train, y_train)

# Obtenir les prédictions du modèle
y_pred = model.predict(X_test)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

              precision    recall  f1-score   support

           0       0.50      0.05      0.09        43
           1       0.66      0.97      0.78        80

    accuracy                           0.65       123
   macro avg       0.58      0.51      0.43       123
weighted avg       0.60      0.65      0.54       123

Accuracy: 0.6504065040650406

Modéle 7 :DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle d'arbre de décision
model = DecisionTreeClassifier()

# Entraîner le modèle
model.fit(X_train, y_train)

# Obtenir les prédictions du modèle
y_pred = model.predict(X_test)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

              precision    recall  f1-score   support

           0       0.55      0.53      0.54        43
           1       0.75      0.76      0.76        80

    accuracy                           0.68       123
   macro avg       0.65      0.65      0.65       123
weighted avg       0.68      0.68      0.68       123

Accuracy: 0.6829268292682927

model_1 = RandomForestClassifier()
model_2 = GradientBoostingClassifier()
model_3 = SVC()
model_4 = LogisticRegression()
model_5 = KNeighborsClassifier()
model_6 = GaussianNB()
model_7 = DecisionTreeClassifier()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
print ("Train size : ",X_train.shape)
print ("Test size : ",X_test.shape)

Train size :  (491, 11)
Test size :  (123, 11)

model_1.fit(X_train,y_train)
model_2.fit(X_train,y_train)
model_3.fit(X_train,y_train)
model_4.fit(X_train,y_train)
model_5.fit(X_train,y_train)
model_6.fit(X_train,y_train)
model_7.fit(X_train,y_train)

DecisionTreeClassifier()

In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# Predict data
y_pred1=model_1.predict(X_test)
y_pred2=model_2.predict(X_test)
y_pred3=model_3.predict(X_test)
y_pred4=model_4.predict(X_test)
y_pred5=model_5.predict(X_test)
y_pred6=model_6.predict(X_test)
y_pred7=model_7.predict(X_test)
#Show F1 Score
from sklearn.metrics import f1_score
f1_model1=f1_score(y_test,y_pred1,average='weighted',labels=np.unique(y_pred1))
f1_model2=f1_score(y_test,y_pred2,average='weighted',labels=np.unique(y_pred2))
f1_model3=f1_score(y_test,y_pred3,average='weighted',labels=np.unique(y_pred3))
f1_model4=f1_score(y_test,y_pred4,average='weighted',labels=np.unique(y_pred4))
f1_model5=f1_score(y_test,y_pred5,average='weighted',labels=np.unique(y_pred5))
f1_model6=f1_score(y_test,y_pred6,average='weighted',labels=np.unique(y_pred6))
f1_model7=f1_score(y_test,y_pred7,average='weighted',labels=np.unique(y_pred7))

print("F1 score Model 1 : ",f1_model1)
print("F1 score Model 2 : ",f1_model2)
print("F1 score Model 3 : ",f1_model3)

print("F1 score Model 4 : ",f1_model4)
print("F1 score Model 5 : ",f1_model5)
print("F1 score Model 6 : ",f1_model6)
print("F1 score Model 7 : ",f1_model7)

F1 score Model 1 :  0.7534336031843106
F1 score Model 2 :  0.7461443705346144
F1 score Model 3 :  0.7761194029850745
F1 score Model 4 :  0.7761194029850745
F1 score Model 5 :  0.5641933702709393
F1 score Model 6 :  0.5356893977103035
F1 score Model 7 :  0.6683560342096927

Réseau de neurones

import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle de réseau de neurones avec TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Obtenir les prédictions du modèle
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))

# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

2024-05-31 11:16:18.165638: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-05-31 11:16:18.165796: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-05-31 11:16:18.334962: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

Epoch 1/10

/opt/conda/lib/python3.10/site-packages/keras/src/layers/core/dense.py:86: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)

13/13 ━━━━━━━━━━━━━━━━━━━━ 2s 20ms/step - accuracy: 0.6728 - loss: 225477.7031 - val_accuracy: 0.6566 - val_loss: 41797.2617
Epoch 2/10
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.5009 - loss: 90555.6016 - val_accuracy: 0.5051 - val_loss: 33629.6758
Epoch 3/10
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.5804 - loss: 71336.1250 - val_accuracy: 0.6162 - val_loss: 31908.7852
Epoch 4/10
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.5628 - loss: 78209.7969 - val_accuracy: 0.5859 - val_loss: 27886.0918
Epoch 5/10
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.6019 - loss: 52839.1445 - val_accuracy: 0.6061 - val_loss: 25899.5254
Epoch 6/10
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.5540 - loss: 65038.2500 - val_accuracy: 0.6364 - val_loss: 25838.2949
Epoch 7/10
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.6032 - loss: 59161.8086 - val_accuracy: 0.6364 - val_loss: 22995.7852
Epoch 8/10
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.5968 - loss: 48313.8164 - val_accuracy: 0.5859 - val_loss: 21244.4414
Epoch 9/10
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - accuracy: 0.5800 - loss: 47081.0586 - val_accuracy: 0.6364 - val_loss: 19773.6914
Epoch 10/10
13/13 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.5776 - loss: 48281.5703 - val_accuracy: 0.6162 - val_loss: 17975.5918
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 13ms/step
              precision    recall  f1-score   support

           0       0.43      0.35      0.38        43
           1       0.68      0.75      0.71        80

    accuracy                           0.61       123
   macro avg       0.56      0.55      0.55       123
weighted avg       0.59      0.61      0.60       123

Accuracy: 0.6097560975609756
