import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble

#read train file
train=pd.read_csv("train.csv")

#fill missing categorical values
cat_miss=['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
for col in cat_miss:
    train[col]=train[col].fillna(train[col].mode()[0])

#convert categorical values to numerical values
catcol=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
dummy=pd.get_dummies(train[catcol],drop_first=True)

#feature normalization
num_col=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
num=(train[num_col]-train[num_col].mean())/train[num_col].std()

#set loan status to 1 if approved else 0
status=train.Loan_Status.apply(lambda x:0 if x=='N' else 1)

train1=pd.concat([num,dummy,status],axis=1)

#create dataframe with numerical missing values dropped and missing values with median inputed
dropped=train1.dropna()
fillmed=train1.fillna(train1.median())

#feture creation
fillmed['LoanAmount_per_term'] = fillmed.LoanAmount/fillmed.Loan_Amount_Term
fillmed['ratio_income_per_term'] = fillmed.ApplicantIncome/fillmed.LoanAmount_per_term

#train test split
x=fillmed.drop('Loan_Status',axis=1).columns
Xtrain, Xtest, Ytrain, Ytest = train_test_split(fillmed[x], fillmed.Loan_Status, test_size = .2)

#training the model
clf=ensemble.RandomForestClassifier(n_estimators=300,max_features=3,min_samples_split=5,oob_score=True,n_jobs=-1,criterion='entropy')
clf.fit(Xtrain,Ytrain)
