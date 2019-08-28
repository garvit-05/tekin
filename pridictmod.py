import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble

#read train file
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

d=[['LP001015','Male','Yes','0','Graduate','No',5720,0,110,360,1,'Urban','Y']]#random test data
test=pd.DataFrame(d,columns=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'])

#fill missing categorical values
cat_miss=['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
for col in cat_miss:
    train[col]=train[col].fillna(train[col].mode()[0])
    test[col]=test[col].fillna(test[col].mode()[0])

#convert categorical values to numerical values
catcol=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
dummy=pd.get_dummies(train[catcol],drop_first=True)
td=[0,0,0,0,0,0,0,0,0,0]
if d[0][1]=='Male':
    td[1]=1
if d[0][2]=='Yes':
    td[2]=1
if d[0][4]=='Graduate':
    td[6]=1
if d[0][5]=='No':
    td[7]=1
if d[0][10]==1:
    td[0]=1
if d[0][11]=='Urban':
    td[9]=1
if d[0][11]=='Semiurban':
    td[8]=1
if d[0][3]=='1':
    td[3]=1
if d[0][3]=='2':
    td[4]=1
if d[0][3]!='0' and d[0][3]!='1' and d[0][3]!='2':
    td[5]=1
testdummy=pd.DataFrame([td],columns=['Credit_History', 'Gender_Male', 'Married_Yes', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 'Education_Not Graduate','Self_Employed_Yes', 'Property_Area_Semiurban', 'Property_Area_Urban'])

#feature normalization
num_col=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
num=(train[num_col]-train[num_col].mean())/train[num_col].std()
testnum=(test[num_col]-train[num_col].mean())/train[num_col].std()

#set loan status to 1 if approved else 0
status=train.Loan_Status.apply(lambda x:0 if x=='N' else 1)

train1=pd.concat([num,dummy,status],axis=1)
test1=pd.concat([testnum,testdummy],axis=1)

#create dataframe with numerical missing values dropped and missing values with median inputed
dropped=train1.dropna()
fillmed=train1.fillna(train1.median())

#feture creation
fillmed['LoanAmount_per_term'] = fillmed.LoanAmount/fillmed.Loan_Amount_Term
fillmed['ratio_income_per_term'] = fillmed.ApplicantIncome/fillmed.LoanAmount_per_term

test1['LoanAmount_per_term'] = test1.LoanAmount/test1.Loan_Amount_Term
test1['ratio_income_per_term'] = test1.ApplicantIncome/test1.LoanAmount_per_term

#train test split
x=fillmed.drop('Loan_Status',axis=1).columns
Xtrain, Xtest, Ytrain, Ytest = train_test_split(fillmed[x], fillmed.Loan_Status, test_size = .2)

#training the model
clf=ensemble.RandomForestClassifier(n_estimators=300,max_features=3,min_samples_split=5,oob_score=True,n_jobs=-1,criterion='entropy')
clf.fit(Xtrain,Ytrain)

#prediction
if clf.predict(test1)[0]==1:
    print("Yes")
else:
    print("No")
