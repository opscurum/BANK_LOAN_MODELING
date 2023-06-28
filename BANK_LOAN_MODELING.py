#!/usr/bin/env python
# coding: utf-8

# # BANK LOAN MODELING

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:


df=loans_2007=pd.read_csv("loans_2007.csv")
df=loans_2007.copy()


# In[ ]:


df


# 
# # DATA PREPROCESSING
Data Cleaning/Cleasing
# In[5]:


#Let's catch the outliers first.


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


#We delete columns of data that we do not need directly.


# In[9]:


removed_columns=['id','member_id','funded_amnt','funded_amnt_inv','grade','sub_grade','emp_title','issue_d','zip_code',
                 'out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int',
                 'total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt']
df=df.drop(removed_columns, axis=1)
df.head(50)


# In[10]:


df.dropna(how="all",axis=0)
df.dropna(how="all", axis=1)


# In[11]:


df.info()


# In[12]:


df.dtypes.value_counts()


# In[13]:


df.describe().T


# In[14]:


df.loan_status.value_counts()


# In[15]:


#Since the important thing for us is whether the loan is paid on time or not, I flew other data.


# In[16]:


drop_columns=[]
for col in df.columns:
    is_unique=len(df[col].dropna().unique())
    if is_unique==1:
        drop_columns.append(col)


# In[17]:


drop_columns


# In[18]:


df=df.drop(drop_columns, axis=1)


# In[19]:


df = df[(df['loan_status'] == 'Fully Paid') | (df['loan_status'] == 'Charged Off')]


# In[20]:


mapping_dict={
    "loan_status":{
        "Fully Paid"  :1,
        "Charged Off" :0
    }
}
df=df.replace(mapping_dict)


# In[21]:


df.loan_status.value_counts()


# # Visualizing the missing data structure

# In[22]:


get_ipython().system('pip install missingno')


# In[23]:


import missingno as msno



# In[24]:


msno.bar(df)


# In[25]:


msno.matrix(df)


Missing values in this data set did not occur randomly. Therefore, it would be logical to operate on multivariate outlier observation analysis while basing on outliers.
# In[26]:


df.isnull().sum()


# In[27]:


msno.heatmap(df)

We understand from the heat map. There is no random correlation between them. Also, since there is no correlation between them, it will not do any harm to delete the data or fill it with the average.
# In[28]:


df.tail(50)


# In[29]:


#Which data has null values, now we will get to them.


# In[30]:


null_counts=df.isnull().sum()
null_counts[null_counts>0]


# In[31]:


print(df.value_counts(normalize=True, dropna=False))

91% of the pub_rec_bankruptcies column generates a value of '0.0'. But it is important because it is a term related to bankruptcy. Therefore, we will not blow it.That's why we'll fly people with null values.
# In[32]:


df=df.drop('pub_rec_bankruptcies', axis=1)


# In[33]:


df=df.dropna(axis=0)


# In[34]:


df.isnull().sum()


# In[35]:


#Empty values are cleared.


# In[36]:


#Below is the code to fix accidentally duplicated data types.


# In[37]:


duplicated_data_type = 'object'
df = df.loc[:, ~df.columns.duplicated()]
duplicated_data_type = ["float64","int64"]
df = df.loc[:, ~df.columns.duplicated()]


# In[38]:


df.dtypes.value_counts()


# In[39]:


df_full=df.copy()


# In[40]:


df_full


# # Multivariate Outlier Observation Analysis

# In[41]:


import seaborn as sns
df=df.select_dtypes(include=["float64","int64"])


# In[42]:


#I prefer the LOF method (Local Outlier Factor) for multivariate outlier observations.


# In[43]:


from sklearn.neighbors import LocalOutlierFactor


# In[44]:


clf=LocalOutlierFactor(n_neighbors=20, contamination=0.1)


# In[45]:


clf.fit_predict(df)


# In[46]:


df_scores=clf.negative_outlier_factor_


# In[47]:


df


# In[48]:


np.sort(df_scores)[0:1000]


# In[49]:


threshold_value=np.sort(df_scores)[19]


# In[50]:


threshold_value


# In[51]:


against_tf=df_scores<threshold_value


# In[52]:


against_tf


# In[53]:


df_scores[against_tf][0:100]

We have reached contradictory values. It does not make much sense to assign or delete the average value here. Therefore, the suppression method with a threshold value will be more appropriate. I will assign threshold values instead of contradictory values.
# In[54]:


df[df_scores==threshold_value]


# In[55]:


pressure_value=df[df_scores==threshold_value]


# In[56]:


outliers=df[df_scores<threshold_value]


# In[57]:


df_scores[against_tf][0:19]


# In[58]:


otliers=df[against_tf]


# In[59]:


outliers


# In[60]:


outliers.to_records(index=False)


# In[61]:


#The action we have done above; DataFrame; numpy is to convert to array. Thus, we got rid of indexes.


# In[62]:


res=outliers.to_records(index=False)


# In[63]:


res[:]=pressure_value.to_records(index=False)


# In[64]:


res


# In[65]:


#The pressure threshold has replaced all the contradictions.


# In[66]:


#Numpy arrays now need to be converted to DataFrame again.


# In[67]:


df[against_tf]=pd.DataFrame(res,index=df[against_tf].index)



# In[68]:


df[against_tf]


# In[69]:


df


# # convert object data to numeric data
# 
# 

# In[70]:


mapping_dict={
    "loan_status":{"Fully Paid"  :1,
                   "Charged Off" :0
    }
}
df=df.replace(mapping_dict)


# In[71]:


df_full


# In[72]:


df_object=df_full.select_dtypes(include=["object"])


# In[73]:


df_object.head()


# In[74]:


df_object.dtypes.value_counts()


# In[75]:


cols=["home_ownership","verification_status","emp_length","term","addr_state"]
for col in cols:
    print(df_object[col].value_counts())


# In[76]:


df_object.purpose.value_counts()


# In[77]:


df_object.info()


# In[78]:


df_object.title.value_counts()


# In[79]:


df_object = df_object.drop(["last_credit_pull_d", "addr_state", "title", "earliest_cr_line"], axis=1)


# In[80]:


#We want to update the "int_rate" column by removing the percent sign and converting it to float data type.


# In[81]:


df_object["int_rate"] = df_object["int_rate"].replace('%', '', regex=True)
df_object["int_rate"] = df_object["int_rate"].astype('float64')


# In[82]:


#We want to update the "revol_util" column by removing the percent sign and converting it to float data type.


# In[83]:


df_object["revol_util"] = df_object["revol_util"].replace('%', '', regex=True)
df_object["revol_util"] = df_object["revol_util"].astype('float64')


# In[84]:


mapping_dict={
    "emp_length":{
        "10+ years":10,
         "9 years"  :9,
         "8 years"  :8,
         "7 years"  :7,
         "6 years"  :6,
         "5 years"  :5,
         "4 years"  :4,
         "3 years"  :3,
         "2 years"  :2,
         "1 years"  :1,
         "1 year"  :1,
         "< 1 year" :0,
         "n/a"      :0
        
    }   
}
df_object=df_object.replace(mapping_dict)


# In[85]:


df_object


# In[86]:


#We will apply multiple dummy variables to the remaining 4 variables. It repeats at certain intervals.


# In[87]:


dummy_df=pd.get_dummies(df_object[["term","verification_status","home_ownership","purpose"]])


# In[88]:


dummy_df


# In[89]:


df_object=df_object.drop(["term","verification_status","home_ownership","purpose"],axis=1)



# In[90]:


df_object


# In[91]:


df_new=pd.concat([df_object,dummy_df], axis=1)


# In[92]:


df_new


# In[93]:


df_new.isnull().sum()


# In[94]:


df


# In[95]:


df=pd.concat([df,df_new], axis=1)


# In[96]:


df.head(50)


# In[97]:


df.isnull().sum()


# In[98]:


#Now our data is ready for the model.


# In[ ]:





# # Important note:
For this data set, you will see two different versions of Logistic Regression here at the beginning. Since I was not very satisfied with them, I also installed the Xgboost model.
# In[ ]:





# # Loan modeling with estimation algorithm

# In[99]:


df.info()


# In[100]:


print(df.columns)
df = df.loc[:, ~df.columns.duplicated()]


# In[101]:


print(df.columns)

We will use this to create a dataset where the "loan_status" column will not be used as an argument (property). So it is now our dependent variable.
# In[102]:


df["loan_status"].value_counts().plot.barh();


# In[103]:


df["loan_status"].value_counts()


# In[104]:


df.describe().T


# In[105]:


df.loan_status.value_counts()


# In[106]:


#We choose the 'LogisticRegression' model because we are doing the classification process.


# In[107]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



# In[108]:


#Modeling with Scikit-Learn


# In[109]:


y=df["loan_status"]
x=df.drop(["loan_status"], axis=1)


# In[110]:


loj = LogisticRegression(solver='liblinear')
loj_model=loj.fit(x,y)
loj_model


# In[111]:


#constant coefficient


# In[112]:


loj_model.intercept_


# In[113]:


#Coefficients of independent variables


# In[114]:


loj_model.coef_


# # Predict&Model Tuning
We need to check and make sure that the dependent variable we are dealing with is the correct class of the class we are dealing with. This is a common problem in classification problems.
# In[115]:


y_pred=loj_model.predict(x)


# In[116]:


#Our correct classification rate.


# In[117]:


confusion_matrix(y,y_pred)


# In[118]:


accuracy_score(y,y_pred)


# In[119]:


print(classification_report(y,y_pred))


# In[120]:


#Following are the estimated values


# In[121]:


loj_model.predict(x)[0:10]


# In[122]:


#The prediction has drawn the probability values. Here we will select the threshold value.


# In[123]:


loj_model.predict_proba(x)[0:10][:,0:2]


# In[124]:


#Actual values


# In[125]:


y[0:10]


# In[126]:


#Now we will choose a threshold value. This way we will do the validation ourselves


# In[127]:


y_probs=loj_model.predict_proba(x)
y_probs=y_probs[:,1]


# In[128]:


y_probs[0:10]


# In[ ]:


y_pred = [1 if i > 0.5 else 0 for i in y_probs] 


# In[ ]:


y_pred[0:10]


# In[ ]:


#Result: We made a good guess. We verified that we chose a correct dependent variable.


# In[ ]:


loj_model.predict_proba(x)[:,1][0:5]


# In[ ]:


logit_roc_auc=roc_auc_score(y,loj_model.predict(x))
fpr,tpr,thresholds=roc_curve(y,loj_model.predict_proba(x)[:,1])
plt.figure()
plt.plot(fpr,tpr,label='AUC (area=%0.2f)'% logit_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()


# In[ ]:


x_train, x_test,y_train,y_test=train_test_split(x,y, test_size=0.25, random_state=42)


# In[135]:


loj = LogisticRegression(solver='liblinear')
loj_model=loj.fit(x_train,y_train)
loj_model



# In[136]:


accuracy_score(y_test,loj_model.predict(x_test))


# In[137]:


cross_val_score(loj_model,x_test,y_test, cv=10)


# In[138]:


cross_val_score(loj_model,x_test,y_test, cv=10).mean()


# In[139]:


#When we get the average, we have reached the most accurate result.


# In[ ]:





# # Estimating with the Other Method
I wanted to reach a solution from a different method for those who want to reach the result in an easier way without doing the Test-Train training.
# In[140]:


df.info()


# In[141]:


df.loan_status.value_counts()


# In[142]:


features=df.drop("loan_status",axis=1) 
target=df['loan_status']


# In[143]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
lr=LogisticRegression(class_weight='balanced')


# In[144]:


df.head()


# In[145]:


features.head()


# In[146]:


predictions=cross_val_predict(lr,features,target, cv=3)


# In[147]:


predictions_2=pd.Series(predictions)


# In[148]:


predictions_2.value_counts()


# In[149]:


#False positives:
fp_filter=(predictions==1)&(target==0)
fp=len(predictions[fp_filter])

#True positives:
tp_filter=(predictions==1)&(target==1)
tp=len(predictions[fp_filter])


#False negatives:
fn_filter=(predictions==0)&(target==1)
fn=len(predictions[fp_filter])


#True negatives:
tn_filter=(predictions==0)&(target==0)
tn=len(predictions[fp_filter])



# In[150]:


#Rates:
tpr=tp/(tp+fn)
fpr=fp/(fp+tn)
print(tpr)
print(fpr)


# In[151]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import pandas as pd

# Penalty dictionary
penalty = {0: 10, 1: 1}

# Logistic Regression model with class weights
lr = LogisticRegression(class_weight=penalty)


# Make predictions using 3-fold cross-validation
predictions = cross_val_predict(lr, features, target, cv=3)
predictions = pd.Series(predictions)

# False positives:
fp_filter = (predictions == 1) & (target == 0)
fp = len(predictions[fp_filter])

# True positives:
tp_filter = (predictions == 1) & (target == 1)
tp = len(predictions[tp_filter])

# False negatives:
fn_filter = (predictions == 0) & (target == 1)
fn = len(predictions[fn_filter])

# True negatives:
tn_filter = (predictions == 0) & (target == 0)
tn = len(predictions[tn_filter])

# Rates:
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print("True Positive Rate (TPR):", tpr)
print("False Positive Rate (FPR):", fpr)


# In[152]:


#Number of credits given without penalty system


# In[153]:


predictions.head(40)


# In[154]:


#Number of credits given when we apply the penalty system


# In[155]:


predictions_2.head(40)


# In[156]:


df


# In[160]:


df.dtypes.value_counts()


# # XGBoost Modeli   (eXtreme Gradient Boosting)

# # Model&Predict

# In[161]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


# In[162]:


y=df["loan_status"]
x=df.drop(["loan_status"], axis=1)
x=pd.DataFrame(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)



# In[163]:


pip install xgboost


# In[164]:


print(x_train.columns)
x_train = x_train.loc[:, ~x_train.columns.duplicated()]



# In[165]:


print(x_train.columns)


# In[166]:


xgb_model=XGBClassifier().fit(x_train,y_train)


# In[167]:


xgb_model


# In[168]:


y_pred=xgb_model.predict(x_test)
accuracy_score(y_test,y_pred)


# # Model Tuning

# In[169]:


xgb_params={
    'n_estimators':[100,500,1000,2000],
    'subsample'   :[0.6,0.8,1.0],
    'max_depth'   :[3,4,5,6],
    'learning_rate':[0.1,0.01,0.02,0.05],
    'min_samples_split':[2,5,10]
    
    
    
}


# In[170]:


xgb=XGBClassifier()
xgb_cv_model=GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1, verbose=2)


# In[ ]:


xgb_cv_model.fit(x_train,y_train)


# In[ ]:


xgb_cv_model.best_params_


# In[ ]:


xgb=XGBClassifier(learning_rate=0.01,
                 max_depth=6,
                  min_samples_split=2,
                  n_estimators=100,
                  subsample=0.8)
                  


# In[ ]:


xgb_tuned=xgb.fit(x_train,y_train)


# In[ ]:


y_pred=xgb_tuned.predict(x_test)
accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




