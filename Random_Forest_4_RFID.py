#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import _tree
import seaborn as sns
from logger import logging_file
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score,recall_score,f1_score,precision_recall_curve,roc_curve,roc_auc_score

lg=logging_file.define_logger("logger_4_RFID_Random_forest")
lg.basic_config()
#reading the dataset
try:
    df=pd.read_csv("E:\\ml project\\classification project\\Activity Recognition\\cleaned_data\\4_RFID.csv")
    lg.info("Reading the 3_RFID File!!!!")
except Exception as e:
    lg.info("There is a issue while reading the 4_RFID csv file issus is {}...".format(e))

print(df.shape)

print(df["Activity"].value_counts())


#dropping first column
try:
    df.drop(["Unnamed: 0","ID Antenna Sensor"],axis=1,inplace=True)
    lg.info("Unnamed: 0 and ID Antenna Sensor column has been dropped!!")
except Exception as e:
    lg.error("ISsue has happened while dropping the column Unnamed:0 and ID Antena Sensor")


print(df.head(10))

#checking for nulls values
lg.info("Checking for nulls")
print(df.isnull().sum())

print(df.shape)


print(df["Activity"].value_counts())

#handiling imbalanaced dataset
lg.info("USing oversampling to handle imbalanced dataset!!")
try:    
    x=df.drop(["Activity"],axis=1)
    y=df["Activity"]
except Exception as e:
    lg.error("There is a issue while dividing the data x and y code line 56 and the error is",e)
try:
    os=SMOTETomek(random_state=42)
    lg.info("SMOTETomek object has been created with random state 42")
except Exception as e:
    lg.error("Issue while object creation time of SMOTETomek")
x_imb,y_imb=os.fit_resample(x,y)
print(y_imb.value_counts())
df=x_imb
df["Activity"]=y_imb
print(df.shape)
#checking distribution of all the data
# lg.info("Checking distribution of the data")
# try:
#     for cols in range(len(df.columns)):
#         print(df.columns[cols])
#         lg.info(f"Plotting the distribution of {df.columns[cols]} column")
#         try:
#             sns.displot(df[df.columns[cols]],kind="kde")
#             plt.title(f"Distribution of {df.columns[cols]}")
#             plt.savefig(f'{df.columns[cols]}.pdf')
#             plt.show()
#         except Exception as e:
#             lg.error(f"Issue while creating the {df.columns[cols]} distribution")
# except Exception as e:
#     lg.error("issue has happended while lopping through data points!!!!")




#checking the data types

# lg.info("checking boxplots")
# try:
#     for cols in range(len(df.columns)):
#         print(df.columns[cols])
#         lg.info(f"Plotting the distribution of {df.columns[cols]} column")
#         try:
#             plt.boxplot(df[df.columns[cols]])
#             plt.title(f"Distribution of {df.columns[cols]}")
#             plt.show()
#         except Exception as e:
#             lg.error(f"Issue while creating the {df.columns[cols]} boxplot issue  is {e}")
# except Exception as e:
#     lg.error("issue has happended while lopping through data points!!!!")


lg.info("time_seconds,Accelaration in G Frontal,Accelaration in G Vertical have high outliers")
lg.info("Handiling them outliers..")
#handiling outliers

df.info()

try:
    Q1=df["time_seconds"].quantile(0.25)
    lg.info("Q1 from time seconds data")
except Exception as e:
    lg.error("Issue has happended for Q1 of time_seconds")

try:
    Q3=df["time_seconds"].quantile(0.75)
    lg.info("Q1 from time seconds data")
except Exception as e:
    lg.error("Issue has happended for Q3 of time_seconds")
lg.info("Creating IQR")
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
lg.info("lower bound created!!")
upper_bound=Q3+1.5*IQR
lg.info("upper bound created!!")
df=df[(df["time_seconds"]>=lower_bound) & (df["time_seconds"]<=upper_bound)]
lg.info("time seconds column outlier handled by percentile method")


try:
    Q1=df["Accelaration in G Frontal"].quantile(0.25)
    lg.info("Q1 from Accelaration in G Frontal data")
except Exception as e:
    lg.error("Issue has happended for Q1 of Accelaration in G Frontal")


try:
    Q3=df["Accelaration in G Frontal"].quantile(0.75)
    lg.info("Q3 from Accelaration in G Frontal data")
except Exception as e:
    lg.error("Issue has happended for Q3 of Accelaration in G Frontal")
lg.info("Creating IQR")
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
df=df[(df["Accelaration in G Frontal"]>=lower_bound) & (df["Accelaration in G Frontal"]<=upper_bound)]
lg.info("Accelaration in G Frontal column outlier handled by percentile method")



try:
    Q1=df["Accelaration in G Vertical"].quantile(0.25)
    lg.info("Q1 from Accelaration in G Vertical")
except Exception as e:
    lg.error("Issue has happended for Q1 of Accelaration in G Vertical")

try:
    Q3=df["Accelaration in G Vertical"].quantile(0.75)
    lg.info("Q1 from Accelaration in G Vertical")
except Exception as e:
    lg.error("Issue has happended for Q3 of Accelaration in G Vertical")
lg.info("Creating IQR")
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
df=df[(df["Accelaration in G Vertical"]>=lower_bound) & (df["Accelaration in G Vertical"]<=upper_bound)]
lg.info("Accelaration in G Vertical column outlier handled by percentile method")


# lg.info("After handiling outliers checking boxplots")
# try:
#     for cols in range(len(df.columns)):
#         print(df.columns[cols])
#         lg.info(f"Plotting the distribution of {df.columns[cols]} column")
#         try:
#             plt.boxplot(df[df.columns[cols]])
#             plt.title(f"Distribution of {df.columns[cols]}")
#             plt.show()
#         except Exception as e:
#             lg.error(f"Issue while creating the {df.columns[cols]} boxplot issue  is {e}")
# except Exception as e:
#     lg.error("issue has happended while lopping through data points!!!!")


print(df.describe(percentiles=[0.25,0.50,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999]))

lg.info("Accelaration in G Lateral lesser than 0.92 percentile")
df=df[df["Accelaration in G Lateral"]<=df["Accelaration in G Lateral"].quantile(0.92)]

# lg.info("After handiling outliers checking boxplots")
# try:
#     for cols in range(len(df.columns)):
#         print(df.columns[cols])
#         lg.info(f"Plotting the distribution of {df.columns[cols]} column")
#         try:
#             plt.boxplot(df[df.columns[cols]])
#             plt.title(f"Distribution of {df.columns[cols]}")
#             plt.show()
#         except Exception as e:
#             lg.error(f"Issue while creating the {df.columns[cols]} boxplot issue  is {e}")
# except Exception as e:
#     lg.error("issue has happended while lopping through data points!!!!")

print(df.describe(percentiles=[0.25,0.50,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999]))

print(df.shape)


#train test split
lg.info("Splitting the data")
x=df.drop("Activity",axis=1)
y=df["Activity"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train_rep=x_train.copy()
x_test_rep=x_test.copy()
#scaling the data
lg.info("Creating object for standard scaling the dataset....")
st=StandardScaler()
#fitting and transforming the data
lg.info("Fit and transform the data")
x_train=st.fit_transform(x_train)
#transformed data converting to dataframe
lg.info("Transformed data converted to dataframe..")
x_train=pd.DataFrame(x_train,columns=x_train_rep.columns)
print(x_train)
# #checking multicolinearity using heatmap
# lg.info("Plotting heatmap")
# sns.heatmap(x_train.corr(),cmap="Greens",annot=True)
# plt.show()

# x_train["ratio"]=x_train["Accelaration in G Frontal"]/x_train["Accelaration in G Lateral"]

# sns.heatmap(x_train.corr(),cmap="Greens",annot=True)
# plt.show()
lg.info("Creating a dataframe for vif")
vif_df=pd.DataFrame()
col=x_train.columns
vif_df["Features"]=col
v=[vif(x_train[col].values, i) for i in range(x_train[col].shape[1])]
vif_df["vif"]=v
vif_df["vif"]=round(vif_df["vif"],2)
print(vif_df)

lg.info("Accelaration in G Frotal has high vif value so we should drop")
x_train.drop("Accelaration in G Frontal",axis=1,inplace=True)
print(x_train.head())

lg.info("Creating a dataframe for vif")
vif_df=pd.DataFrame()
col=x_train.columns
vif_df["Features"]=col
v=[vif(x_train[col].values, i) for i in range(x_train[col].shape[1])]
vif_df["vif"]=v
vif_df["vif"]=round(vif_df["vif"],2)
print(vif_df)


sns.heatmap(x_train.corr(),annot=True,cmap="Greens")
plt.show()


print(x_test)
#transforming x_test
x_test=st.transform(x_test)
x_test=pd.DataFrame(x_test,columns=x_test_rep.columns)
#dropping from test data
lg.info("Dropping Accelaration in G Frontal column from test data")
x_test.drop("Accelaration in G Frontal",axis=1,inplace=True)
#Logistics Regression
lg.info("Creating Logistics Regression Model")
try:
    lr=LogisticRegression(penalty='l2',tol=0.0001,solver="newton-cg")
except Exception as e:
    lg.error("Error has happened on the line 272 while creating Logistic model!!")
try:
    lg.info("fitting the model")
    lr.fit(x_train,y_train)
except Exception as e:
    lg.error("Error while fitting the Logistics Regression Model")
lg.info("Predicting the output through Logistic Regression!!")

y_test_pred=lr.predict(x_test)
y_test_pred_prob=lr.predict_proba(x_test)

lg.info("Plotting classificatio report")
print(classification_report(y_test,y_test_pred))

# #applying grid seach cv
# param={
#     "penalty":['l1', 'l2', 'elasticnet'],
#     "tol":[0.1,0.001,0.0001,0.00001,0.000001,0.00000001],
#     "C":[0,0.1,0.001,0.2,0.002,0.3,0.003,0.004,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
#     "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#     "max_iter":range(0,200),
#     "multi_class":['ovr', 'multinomial']
# }
# try:
#     g=GridSearchCV(estimator=lr,param_grid=param,cv=50,verbose=True)
#     lg.info("Grid Search CV Has been applied")
# except Exception as e:
#     lg.error("Error has occured while grid search cv")
# try:
#     g.fit(x_train,y_train)
#     lg.info("Fitting has been done on Grid Search CV")
# except Exception as e:
#     lg.error("Error has happened while fitting in grid search cv")
# try:
#     pred=g.predict(x_test)
#     lg.info("Prediction has done on Grid Search CV")
# except Exception as e:
#     lg.error("Error while prediction in Grid Search CV")
# print(pred)

# lg.info("Plotting classificatio report on Grid Search CV (Logistics Regression)")
# print(classification_report(y_test,y_test_pred))


#applying Decison Tree classifier
lg.info("Applying Decison Tree classifier")
try:
    dt=DecisionTreeClassifier()
    lg.info("DecisionTreeClassifier object created!!")
except Exception as e:
    lg.error("Error has happened while creating object!!")
#fit the data
try:
    dt.fit(x_train,y_train)
    lg.info("Fitting the training data through decision tree model")
except Exception as e:
    lg.error("Error while dt fitting!!")
#predict the data
y_test_pred=dt.predict(x_test)
print(classification_report(y_test,y_test_pred))

#trying to get ccp_alpha
try:
    path=dt.cost_complexity_pruning_path(x_train,y_train)
    lg.info("cost complexity pruning has happened")
    print(path['ccp_alphas'])
except Exception as e:
    lg.error("ISsue while selecting ccp_alphas error is ",{e})

#applying Random forest classifier
lg.info("Applying Decison Tree classifier")
try:
    r=RandomForestClassifier(criterion="gini",max_depth=9,max_features="log2",min_samples_leaf=1,min_samples_split=3)
    lg.info("RandomForestClassifier object created!!")
except Exception as e:
    lg.error(f"Error has happened while creating object!! and the error is {e}")
#fit the data
try:
    r.fit(x_train,y_train)
    lg.info("Fitting the training data through random forest model")
except Exception as e:
    lg.error("Error while dt fitting!!")
#predict the data
y_train_pred=r.predict(x_train)
print(classification_report(y_train,y_train_pred))

y_test_pred=r.predict(x_test)
print(classification_report(y_test,y_test_pred))
# param={
#     "criterion":['gini', 'entropy', 'log_loss'],
#     "max_depth":range(0,40),
#     "min_samples_split":range(0,40),
#     "min_samples_leaf":range(0,40)
#     # "max_features":['auto', 'sqrt', 'log2'],
#     # "max_leaf_nodes":range(0,40),
#     # "min_impurity_decrease":range(0,40)
# }
# g_dt=GridSearchCV(estimator=r,param_grid=param,verbose=True)
# g_dt.fit(x_train,y_train)

# print(g_dt.best_params_)



