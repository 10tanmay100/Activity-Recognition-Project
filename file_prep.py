#importing libraries 
import pandas as pd
import numpy as np
import logging as lg
import matplotlib.pyplot as plt
import seaborn as sns
import os
from logger import logging_file



# let's create a log file with name "logger" by calling a class from different py file
logger=logging_file.define_logger("logger1")
logger.basic_config()
#creating a empty dataframe
try:
    main_df=pd.DataFrame(columns=["A","B","C","D","E","F","G","H","i"])
    logger.info("Empty dataframe has been created!!")
except Exception as e:
    logger.error(f"Error has happened while creating empty df --->{e}")

for folders in range(len(os.listdir('E:\ml project\classification project\Activity Recognition'))):
    if os.listdir('E:\ml project\classification project\Activity Recognition')[folders]=="S1_Dataset" or os.listdir('E:\ml project\classification project\Activity Recognition')[folders]=="S2_Dataset":

        try:
            path="E:\ml project\classification project\Activity Recognition\{}".format(os.listdir('E:\ml project\classification project\Activity Recognition')[folders])
            logger.info(f"Read {folders} folder")
        except Exception as e:
            logger.error("Path is not correct!!")
        logger.info(f"Doing work for the {folders} folder")
        if os.listdir('E:\ml project\classification project\Activity Recognition')[folders]=="S1_Dataset":
            logger.info(f"path for file {path}")
            logger.info("Creating dataframe for files in the {} folder".format(folders))
            pd_df_s1=pd.DataFrame()
            logger.info("pd_df_s1 logger has been created")
            logger.info("Looping through the {} folder directory to read all the files!!".format(folders))
            for files in os.listdir(path):
                if files[-3:]=="txt":
                    logger.warning("Ignoring .txt file on the {} folder".format(folders))
                    continue
                else:
                    logger.info("Read the path of {} file".format(files))
                    path_file=path+"\{}".format(files)
                try:
                    df=pd.read_csv(path_file)
                    logger.info(f"{files} has been read by pandas dataframe")
                except Exception as e:
                    logger.error("Some issue while reading the {} file and the issue is {}".format(files,e))
                #adding headers
                headerList = ['time_seconds', 'Accelaration in G Frontal', 'Accelaration in G Vertical',"Accelaration in G Lateral","ID Antenna Sensor","RSSI","Phase","Frequency","Activity"]
                logger.info("Adding headers for csv's")
                #converting dataframe to csv
                logger.info("Converting dataframe to csv")
                try:
                    df.to_csv("df.csv", header=headerList, index=False)
                    logger.info("Csv conversion done with the name as df.csv with paramerter header=headerlist and index=False")
                except Exception as e:
                    logger.error("Issue has happened while converting to csv and the issue is that {}".format(e))
                try:
                    df=pd.read_csv("df.csv")
                    logger.info("Reading that csv")
                except Exception as e:
                    logger.error("Error is {}".format(e))
                os.remove("df.csv")
                logger.info("remove that csv")
                try:
                    pd_df_s1=pd.concat([pd_df_s1,df])
                    logger.info("Concatenating two dfs")
                except Exception as e:
                    logger.error("Issue while concatenation has happened")

        elif os.listdir('E:\ml project\classification project\Activity Recognition')[folders]=="S2_Dataset":
            logger.info("Creating dataframe for files in the {} folder".format(folders))
            pd_df_s2=pd.DataFrame()
            logger.info("pd_df_s2 logger has been created!!")
            logger.info("Looping through the {} folder directory to read all the files!!".format(folders))
            for files in os.listdir(path):
                if files[-3:]=="txt":
                    logger.warning("Ignoring .txt file on the {} folder".format(folders))
                    continue
                else:
                    logger.info("Read the path of {} file".format(files))
                    path_file=path+"\{}".format(files)
                try:
                    df=pd.read_csv(path_file)
                    logger.info(f"{files} has been read by pandas dataframe")
                except Exception as e:
                    logger.error("Some issue while reading the {} file and the issue is {}".format(files,e))
                #adding headers
                headerList = ['time_seconds', 'Accelaration in G Frontal', 'Accelaration in G Vertical',"Accelaration in G Lateral","ID Antenna Sensor","RSSI","Phase","Frequency","Activity"]
                logger.info("Adding headers for csv's")
                #converting dataframe to csv
                logger.info("Converting dataframe to csv")
                try:
                    df.to_csv("df.csv", header=headerList, index=False)
                    logger.info("Csv conversion done with the name as df.csv with paramerter header=headerlist and index=False")
                except Exception as e:
                    logger.error("Issue has happened while converting to csv and the issue is that {}".format(e))
                try:
                    df=pd.read_csv("df.csv")
                    logger.info("Reading the csv")
                except Exception as e:
                    logger.error("Issue while csv reading has happened")
                os.remove("df.csv")
                logger.info("removing the csv")
                try:
                    pd_df_s2=pd.concat([pd_df_s2,df])
                    logger.info("Concatenating two dfs")
                except Exception as e:
                    logger.error("Issue while concatenation has happened")
            # print(pd_df_s2.shape)
        else:
            continue
    else:
        continue
print(pd_df_s1.shape)
print(pd_df_s2.shape)

logger.info("Now we are converting those df's in csv file")
os.chdir("E:\ml project\classification project\Activity Recognition\cleaned_data")
logger.info("Changing the directory")
pd_df_s1.to_csv("4_RFID.csv")
logger.info("First Convertion Done")
pd_df_s2.to_csv("3_RFID.csv")
logger.info("Second Convertion Done")






























