#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
from scipy import rand
import seaborn as sns
from logger import logging_file
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
lg=logging_file.define_logger("logger_4_RFID")
lg.basic_config()
#reading the dataset
try:
    df=pd.read_csv("E:\\ml project\\classification project\\Activity Recognition\\cleaned_data\\4_RFID.csv")
    lg.info("Reading the 4_RFID File!!!!")
except Exception as e:
    lg.info("There is a issue while reading the 3_RFID csv file issus is {}...".format(e))


# print(df.head())

print(df["Activity"].value_counts())

print(df.isnull().sum())

#dropping first column
try:
    df.drop(["Unnamed: 0","ID Antenna Sensor"],axis=1,inplace=True)
    lg.info("Unnamed: 0 and ID Antenna Sensor column has been dropped!!")
except Exception as e:
    lg.error("ISsue has happened while dropping the column Unnamed:0 and ID Antena Sensor")


print(df.shape)

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

lg.info("time_seconds,Accelaration in G lateral,RSSI have high outliers")
lg.info("Handiling them outliers..")

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
    Q1=df["Accelaration in G Lateral"].quantile(0.25)
    lg.info("Q1 from Accelaration in G Lateral data")
except Exception as e:
    lg.error("Issue has happended for Q1 of Accelaration in G lateral")
try:
    Q3=df["Accelaration in G Lateral"].quantile(0.75)
    lg.info("Q1 from Accelaration in G lateral data")
except Exception as e:
    lg.error("Issue has happended for Q3 of Accelaration in G lateral")
lg.info("Creating IQR")
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
lg.info("lower bound created!!")
upper_bound=Q3+1.5*IQR
lg.info("upper bound created!!")
df=df[(df["Accelaration in G Lateral"]>=lower_bound) & (df["Accelaration in G Lateral"]<=upper_bound)]
lg.info("Accelaration in G lateral column outlier handled by percentile method")

try:
    Q1=df["RSSI"].quantile(0.25)
    lg.info("Q1 from RSSI data")
except Exception as e:
    lg.error("Issue has happended for Q1 of RSSI")
try:
    Q3=df["RSSI"].quantile(0.75)
    lg.info("Q1 from RSSI")
except Exception as e:
    lg.error("Issue has happended for Q3 of Accelaration in G lateral")
lg.info("Creating IQR")
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
lg.info("lower bound created!!")
upper_bound=Q3+1.5*IQR
lg.info("upper bound created!!")
df=df[(df["RSSI"]>=lower_bound) & (df["RSSI"]<=upper_bound)]
lg.info("RSSI column outlier handled by percentile method")


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

# print(df.describe(percentiles=[0.25,0.50,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999]))

lg.info("time_seconds lesser than 0.97 percentile")
df=df[df["time_seconds"]<=df["time_seconds"].quantile(0.97)]
print(df.head())

print(df.describe(percentiles=[0.25,0.50,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999]))



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

#checking multicolinearity using heatmap
lg.info("Plotting heatmap")
sns.heatmap(x_train.corr(),cmap="Greens",annot=True)
plt.show()


lg.info("Creating a dataframe for vif")
vif_df=pd.DataFrame()
col=x_train.columns
vif_df["Features"]=col
v=[vif(x_train[col].values, i) for i in range(x_train[col].shape[1])]
vif_df["vif"]=v
vif_df["vif"]=round(vif_df["vif"],2)
print(vif_df)

lg.info("Accelaration in G Frotal has high vif value so we should drop")
x_train.drop("Accelaration in G Vertical",axis=1,inplace=True)
print(x_train.head())

lg.info("Creating a dataframe for vif")
vif_df=pd.DataFrame()
col=x_train.columns
vif_df["Features"]=col
v=[vif(x_train[col].values, i) for i in range(x_train[col].shape[1])]
vif_df["vif"]=v
vif_df["vif"]=round(vif_df["vif"],2)
print(vif_df)


lg.info("Plotting heatmap")
sns.heatmap(x_train.corr(),cmap="Greens",annot=True)
plt.show()






