import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from seaborn import countplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def TitanicLogistic():
    Line = "*||*"*14
    
    print("<->"*10)
    print("Inside Logistic Function")
    print("<->"*10)
    
    print()
    #Step 1 - Load Data
    Titanic_data = pd.read_csv("MarvellousTitanicDataset.csv")
    
    print(Line)
    print("First five records of data set")
    print(Line)
    
    print()
    
    print("-{}-"*21)
    print(Titanic_data.head())
    print("-{}-"*21)
    
    print()
    
    print(Line)
    print("Total number of records are : ",len(Titanic_data))
    print(Line)
    
    #print("Total number of records are : ",Titanic_data.info)
    #print("Total number of records are : ",Titanic_data.shape)
    
    #Step 2 - Analyze The Data
    print()
    
    print(Line)
    print("Visualisation of Survived and Non-Survived passanger")
    print(Line)
    
    figure()
    countplot(data=Titanic_data,x = "Survived").set_title("Survived vs Non-Survived")
    show()
    
    print()
    
    print(Line)
    print("Visualisation according to Sex")
    print(Line)
    
    figure()
    countplot(data = Titanic_data,x = "Survived",hue = "Sex").set_title("Visualisation according to Sex")
    show()
    
    print()
    
    print(Line)
    print("Visualisation according to Passenger Class")
    print(Line)
    
    figure()
    countplot(data = Titanic_data,x = "Survived",hue = "Pclass").set_title("Visualisation according to Passenger")
    show()
    
    print()
    
    print(Line)
    print("Survived vs Non-Survived based on Age")
    print(Line)
    
    figure()
    Titanic_data["Age"].plot.hist().set_title("Visualisation according to Age")
    show()
    
    #Step 3 - Data Cleaning
    
    Titanic_data.drop("zero",axis = 1,inplace = True)
    
    print()
    
    print(Line)
    print("Data after column removal")
    print(Line)
    
    print()
    
    print("-{}-"*21)
    print(Titanic_data.head())
    print("-{}-"*21)
    
    #Creating Dummy Columns
    
    Sex = pd.get_dummies(Titanic_data["Sex"])
    
    print()
    
    print("-()-"*6)
    print(Sex.head())
    print("-()-"*6)
    
    Sex = pd.get_dummies(Titanic_data["Sex"],drop_first = True)
    
    print()
    
    print(Line)
    print("Sex Column after Updation")
    print(Line)
    
    print()
    
    print("-()-"*6)
    print(Sex.head())
    print("-()-"*6)
    
    Pclass = pd.get_dummies(Titanic_data["Pclass"])
    
    print()
    print("Data of Pclass")
    print()
    print("-()-"*6)
    print(Pclass.head())
    print("-()-"*6)
    
    Pclass = pd.get_dummies(Titanic_data["Pclass"],drop_first = True)
    
    print()
    
    print(Line)
    print("Pclass Column after Updation")
    print(Line)
    
    print("-()-"*6)
    print(Pclass.head())
    print("-()-"*6)
    
    #Concate Sex and Pclass field in our Dataset
    Titanic_data = pd.concat([Titanic_data,Sex,Pclass],axis = 1)
    
    print()
    
    print(Line)
    print("Data after concatination")
    print(Line)
    
    print()
    
    print("-{}-"*21)
    print(Titanic_data.head())
    print("-{}-"*21)
    
    #Removing uneccessary field
    
    Titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis = 1,inplace = True)
    
    print()
    
    print("-{}-"*21)
    print(Titanic_data.head())
    print("-{}-"*21)
    
    #Divide the data set into x and yield
    
    x = Titanic_data.drop("Survived",axis = 1)
    y = Titanic_data["Survived"]
    
    #Split the data for trainig and testing purpose
    
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.5)
    
    print("-{}-"*20)
    print("\nDataset split Size",len(xtrain), len(xtest), len(ytrain), len(ytest))
    print("-{}-"*20)
    
    #Get Model object
    obj = LogisticRegression(max_iter = 2000)
    
    #Step 4 - Train the Dataset
    
    obj.fit(xtrain,ytrain)
    
    #Step 5 - Testing
    
    output = obj.predict(xtest)
    
    print()
    
    print(Line)
    print("The accuracy of the given dataset is")
    print(Line)
    
    print()
    
    print("<^>"*7)
    print(accuracy_score(ytest,output))
    print("<^>"*7)
    
    print()
    
    print(Line)
    print("Confusion metrics is")
    print(confusion_matrix(ytest,output))
    print(Line)
    
def main():
    print()
    
    print("-*-"*10)
    print("Logistic Case Study")
    print("-*-"*10)
    
    print()
    
    TitanicLogistic()

if __name__ == "__main__":
    main()