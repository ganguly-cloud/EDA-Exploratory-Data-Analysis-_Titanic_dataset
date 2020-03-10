import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print train_data.shape  # (891, 12)
print test_data.shape   # (418, 11)

print train_data.columns
'''
Index([u'PassengerId', u'Survived', u'Pclass', u'Name', u'Sex', u'Age',
       u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked'],
      dtype='object')'''
print test_data.columns
'''
Index([u'PassengerId', u'Pclass', u'Name', u'Sex', u'Age', u'SibSp', u'Parch',
       u'Ticket', u'Fare', u'Cabin', u'Embarked'],
      dtype='object')'''

print train_data.iloc[:,:3].head()   # just to explore all the columns
'''
   PassengerId  Survived  Pclass
0            1         0       3
1            2         1       1
2            3         1       3
3            4         1       1
4            5         0       3 '''

print train_data.iloc[:,3:5].head()
'''
0                            Braund, Mr. Owen Harris    male
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female
2                             Heikkinen, Miss. Laina  female
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female
4                           Allen, Mr. William Henry    male'''

print train_data.iloc[:,5:].head()
'''
    Age  SibSp  Parch            Ticket     Fare Cabin Embarked
0  22.0      1      0         A/5 21171   7.2500   NaN        S
1  38.0      1      0          PC 17599  71.2833   C85        C
2  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
3  35.0      1      0            113803  53.1000  C123        S
4  35.0      0      0            373450   8.0500   NaN        S'''

print train_data.isnull().sum()
'''
PassengerId      0   # 1 survive  & 0 for not servive
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64'''
sns.heatmap(train_data.isnull(),yticklabels=False,cmap='viridis')

# plot to see how many null values are there

sns.heatmap(train_data.isnull(),yticklabels=False,cmap='viridis')
plt.savefig('Before_remove_null_values')
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train_data)
plt.savefig('Howmany_servived_0_notservive_1_servive')
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train_data)
plt.savefig('Howmany_servived_male&female')
plt.show()

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train_data)
plt.savefig('Howmany_servived_based_on_Pclass')
plt.show()

# To check the age distributions

sns.distplot(train_data['Age'].dropna(),kde=False,color='darkred',bins=40)
plt.savefig('Age_distribution_plot')
plt.show()

# Plot to check how many siblings has

sns.set_style('whitegrid')
sns.countplot(x='SibSp',data=train_data)
plt.savefig('plot_for_how_many_siblings has')
plt.show()

# plot to check Fare
sns.set_style('whitegrid')
train_data['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.savefig('Plot_for_fare_check')
plt.show()

# DATA CLEANING

sns.set_style('whitegrid')
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train_data,palette='winter')
plt.savefig('Box_plot_to_checkthe_MeanofEacH')
plt.show()

#1st replace all age null values

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

### Now apply that function

train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis =1)
train_data.drop('Cabin',axis=1,inplace=True)
train_data.dropna(inplace=True)
sns.heatmap(train_data.isnull(),yticklabels=False,cmap='viridis')
plt.show()
print train_data.isnull().sum()
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

test_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis =1)
test_data.drop('Cabin',axis=1,inplace=True)
test_data.dropna(inplace=True)
print test_data.isnull().sum()
##

sns.heatmap(test_data.isnull(),yticklabels=False,cmap='viridis')
plt.show()

sex = pd.get_dummies(train_data['Sex'],drop_first=True)
emb = pd.get_dummies(train_data['Embarked'],drop_first=True)
train_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

final_train_data = pd.concat([train_data,sex,emb],axis=1)
main_df = train_data.copy()
print final_train_data.shape  # (889, 10)

sex = pd.get_dummies(test_data['Sex'],drop_first=True)
emb = pd.get_dummies(test_data['Embarked'],drop_first=True)
test_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

final_test_data = pd.concat([test_data,sex,emb],axis=1)

print final_test_data.shape   # (416, 9)

final_df = pd.concat([final_train_data,final_test_data],axis=0)
print final_df.head()
'''
    Age     Fare  Parch  PassengerId  Pclass  Q  S  SibSp  Survived  male
0  22.0   7.2500      0            1       3  0  1      1       0.0     1
1  38.0  71.2833      0            2       1  0  0      1       1.0     0
2  26.0   7.9250      0            3       3  0  1      0       1.0     0
3  35.0  53.1000      0            4       1  0  1      1       1.0     0
4  35.0   8.0500      0            5       3  0  1      0       0.0     1'''
print final_df.shape   # (1305, 10)


print final_df.isnull().sum()

df_train = final_df.iloc[:889,:]
df_test  = final_df.iloc[889:,:]

df_test.drop(['Survived'],axis=1,inplace=True)

print df_test.shape  # (416, 9)

x = df_train.drop('Survived',axis=1)
y = df_train['Survived']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.30,random_state=101)
logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)

# We can our model into Pickle file since to train will consume much time
  # so no need to train our model again and again

import pickle
filename = 'Finalized_model.pkl'
pickle.dump(logmodel,open(filename,'wb'))

y_pred = logmodel.predict(x_test)

accu = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
print 'Accuracy of our model : ',accu
''' Accuracy of our model :  0.8202247191011236 '''

print cm
'''
[[151  12]
 [ 36  68]]'''

y_pred = logmodel.predict(df_test)
print y_pred
'''
[0. 1. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 1. 1. 0. 0. 1. 1. 0. 0. 1. 1.
 1. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0.
 1. 1. 1. 0. 1. 1. 0. 0. 0. 0. 0. 1. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1.
 1. 1. 1. 0. 1. 0. 1. 0. 1. 0. 0. 0. 0. 1. 1. 1. 0. 1. 0. 1. 0. 0. 0. 1.
 0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 0. 0. 1. 1. 1. 1.
 0. 1. 0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 1. 0. 1.
 0. 1. 0. 0. 0. 1. 0. 1. 0. 0. 1. 1. 0. 1. 1. 0. 1. 0. 0. 1. 0. 0. 1. 1.
 0. 0. 0. 0. 0. 1. 1. 0. 1. 1. 0. 0. 1. 0. 1. 0. 1. 0. 0. 0. 0. 1. 0. 0.
 0. 0. 1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0.
 1. 0. 1. 0. 1. 0. 1. 1. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1. 1. 1. 1.
 1. 0. 0. 0. 1. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0.
 1. 0. 0. 0. 1. 0. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 1. 0. 0.
 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 1. 0. 0. 0. 1.
 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0. 1. 0. 0.
 0. 0. 0. 0. 0. 1. 0. 1. 0. 1. 0. 1. 1. 0. 0. 0. 1. 0. 1. 0. 0. 1. 0. 1.
 1. 0. 1. 0. 0. 1. 1. 0. 0. 1. 1. 0. 1. 1. 1. 0. 1. 0. 0. 0. 1. 1. 0. 1.
 0. 0. 0. 0. 1. 1. 0. 0. 0. 1. 0. 1. 0. 0. 1. 0. 1. 1. 1. 0. 0. 1. 1. 1.
 1. 1. 1. 0. 1. 0. 0. 0.]'''

print len(y_pred)   # 416

# have to save all predected results into a file for that convert all the
 # array into DataFrame

pred = pd.DataFrame(y_pred)
sub_df =pd.read_csv('gender_submission.csv')
datasets = pd.concat([sub_df['PassengerId'],pred],axis=1)
datasets.columns=['PassengerId','Survived']
datasets.to_csv('sample_submission.csv',index= False)
