import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import datasets, linear_model, metrics


data = pd.read_csv("/kaggle/input/best-universities-in-the-united- kingdom/uk_universities.csv")
data.head()

print("Records : ",data.shape[0],"Features : ",data.shape[1]) data.info()
data.isna().any()
data.isna().sum() missing_values =
pd.DataFrame({'Missing_Values_Count':data.isna().sum(),"Percentage":data.isna().s um()/len(data) * 100})
missing_values.style.background_gradient(cmap='hot') data['Motto'].replace(np.NaN,"NULL",inplace=True)
grp_data = data.groupby(['Region'])

data['Academic_Calender'] = grp_data['Academic_Calender'].transform(lambda x :
x.fillna(method='ffill')) data['Academic_Calender'].value_counts()

data['Campus_setting'] = data['Campus_setting'].fillna(method='ffill') data['Campus_setting'].value_counts() plt.hist(data['CWUR_score'],bins=20)
plt.axvline(data['CWUR_score'].mean(),color='g')

plt.axvline(data['CWUR_score'].median(),color='r')

data['CWUR_score'] = data['CWUR_score'].interpolate() data['CWUR_score']
sns.heatmap(data.isna())

col = data.select_dtypes(exclude='object') for i in col:
print("Feature : ",i) plt.boxplot(data[i])
plt.show()

col = data[['Longitude','Latitude','Estimated_cost_of_living_per_year_(in_pounds)','PG_ average_fees_(in_pounds)','UG_average_fees_(in_pounds)','CWUR_score','World_rank'
,'UK_rank','Founded_year']]

def IQR(data,col):

q1 = data[col].quantile(0.25) q3 = data[col].quantile(0.75) iqr = q3 - q1
return iqr,q1,q3 for i in col:
iqr,q1,q3 = IQR(data,i) lower = q1 - 1.5*iqr upper = q3 + 1.5*iqr
data = data[(data[i] > lower) & (data[i] < upper)] for i in col:
print("Feature : ",i) plt.boxplot(data[i])

plt.show()




data.info()

p_table = pd.pivot_table(data,values=['UG_average_fees_(in_pounds)','PG_average_fees_(in_po unds)','Minimum_IELTS_score'],columns='Campus_setting',aggfunc=(np.median))
p_table

data['UG_average_fees_(in_pounds)'].plot(kind='box')

data['UG_Fess'] = (data['UG_average_fees_(in_pounds)'] - min(data['UG_average_fees_(in_pounds)'])) / (max(data['UG_average_fees_(in_pounds)']) - min(data['UG_average_fees_(in_pounds)']))
data['UG_Fess'] data['UG_Fess'].plot(kind='box')
data['PG_Fess'] = (data['PG_average_fees_(in_pounds)'] - min(data['PG_average_fees_(in_pounds)'])) / (max(data['PG_average_fees_(in_pounds)']) - min(data['PG_average_fees_(in_pounds)']))
sns.kdeplot(data['PG_Fess'],fill=True)

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(25,7))

sns.kdeplot(data['UG_Fess'],fill=True,ax=ax1)

sns.kdeplot(data['PG_Fess'],fill=True,ax=ax2)

data['Std_UG'] = (data['UG_average_fees_(in_pounds)']- np.mean(data['UG_average_fees_(in_pounds)'])) / np.std(data['UG_average_fees_(in_pounds)'])
data['Std_UG']

data['Std_PG'] = (data['PG_average_fees_(in_pounds)']- np.mean(data['PG_average_fees_(in_pounds)'])) / np.std(data['PG_average_fees_(in_pounds)'])
data['Std_PG']

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(25,7))

sns.kdeplot(data['Std_PG'],fill=True,ax=ax1)

sns.kdeplot(data['Std_UG'],fill=True,ax=ax2)

X=dataset.iloc['Longitude','Latitude','Estimated_cost_of_living_per_year_(in_poun ds)','PG_average_fees_(in_pounds)','UG_average_fees_(in_pounds)','CWUR_score','Wo rld_rank','UK_rank','Founded_year'].values
y=dataset.iloc['Longitude','Latitude','Estimated_cost_of_living_per_year_(in_poun ds)','PG_average_fees_(in_pounds)','UG_average_fees_(in_pounds)','CWUR_score','Wo rld_rank','UK_rank','Founded_year'].values
from sklearn.preprocessing import Imputer imputer=Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer=imputer.fit(X['Longitude','Latitude','Estimated_cost_of_living_per_year_( in_pounds)','PG_average_fees_(in_pounds)','UG_average_fees_(in_pounds)','CWUR_sco re','World_rank','UK_rank','Founded_year'])
X[<range of rows and columns>]=imputer.transform(X['Longitude','Latitude','Estimated_cost_of_living_pe r_year_(in_pounds)','PG_average_fees_(in_pounds)','UG_average_fees_(in_pounds)',' CWUR_score','World_rank','UK_rank','Founded_year'])
from sklearn.preprocessing import LabelEncoder le_X=LabelEncoder()
X[<range of rows and columns>]=le_X.fit_transform(X['Longitude','Latitude','Estimated_cost_of_living_p er_year_(in_pounds)','PG_average_fees_(in_pounds)','UG_average_fees_(in_pounds)', 'CWUR_score','World_rank','UK_rank','Founded_year'])
labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder

oneHE=OneHotEncoder(categorical_features=['Longitude','Latitude','Estimated_cost_ of_living_per_year_(in_pounds)','PG_average_fees_(in_pounds)','UG_average_fees_(i n_pounds)','CWUR_score','World_rank','UK_rank','Founded_year'])
X=oneHE.fit_transform(X).toarray()

from sklearn.preprocessing import StandardScaler sc_X=StandardScaler()
X=sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25) from sklearn.metrics import confusion_matrix
classifier = confusion_matrix(y_test, y_pred) classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix cm = confusion_matrix(y_test, y_pred) print(cm)


from sklearn.metrics import classification_report target_names = [<list of class names>]
print(classification_report(y_test, y_pred, target_names=target_names))



boston = datasets.load_boston(return_X_y=False)


X = boston.data

y = boston.target


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,

random_state=1)


reg = linear_model.LinearRegression()

reg.fit(X_train, y_train)


print('Coefficients: ', reg.coef_)


print('Variance score: {}'.format(reg.score(X_test, y_test)))


plt.style.use('fivethirtyeight')


plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,

color = "green", s = 10, label = 'Train data')


plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,

color = "blue", s = 10, label = 'Test data')




plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)


plt.legend(loc = 'upper right')


plt.title("Residual errors")


plt.show()


from sklearn.datasets import make_classification from sklearn.model_selection import train_test_split from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression nb_samples = 1000

x, y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression() model.fit(xtrain, ytrain)
print(accuracy_score(ytest, model.predict(xtest)))
