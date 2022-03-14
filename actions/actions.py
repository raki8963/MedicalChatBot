# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
import csv
import warnings

class ActionStoreSymptoms(Action):
    def name(self) -> Text:
        return "action_store_symptoms"
    
    def run(self,dispatcher:CollectingDispatcher,tracker:Tracker,domain:Dict[Text,Any]) -> List[Dict[Text,Any]]:
        p=tracker.get_slot('symptom')
        print(p)
        f=open("C:/Users/KATAKAM19/Desktop/rasa/actions/storesymptoms.txt","a")
        for k in p:
            print(k)
            f.write(k + "\n")
        return []

class ActionDiagnoseySymptoms(Action):

    def name(self) -> Text:
        return "action_diagnose_symptoms"


    def run(self,dispatcher:CollectingDispatcher,tracker:Tracker,domain:Dict[Text,Any]) -> List[Dict[Text,Any]]:

        def editDistDP(str1, str2, m, n):
        # Create a table to store results of subproblems
            dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
        
            # Fill d[][] in bottom up manner
            for i in range(m + 1):
                for j in range(n + 1):
        
                    # If first string is empty, only option is to
                    # insert all characters of second string
                    if i == 0:
                        dp[i][j] = j    # Min. operations = j
        
                    # If second string is empty, only option is to
                    # remove all characters of second string
                    elif j == 0:
                        dp[i][j] = i    # Min. operations = i
        
                    # If last characters are same, ignore last char
                    # and recur for remaining string
                    elif str1[i-1] == str2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
        
                    # If last character are different, consider all
                    # possibilities and find minimum
                    else:
                        mn=dp[i][j-1]
                        if mn<dp[i-1][j]:
                            mn=dp[i-1][j]
                        if mn<dp[i-1][j-1]:
                            mn=dp[i-1][j-1]
                        dp[i][j] = 1 + mn  
        
            return dp[m][n]
        training = pd.read_csv('Training.csv')
        symprec=pd.read_csv('actions/symptom_precaution.csv')
        symdesc=pd.read_csv('actions/symptom_Description.csv')
        docdata=pd.read_csv('actions/doctors_dataset.csv')
        cols= training.columns
        cols= cols[:-1]
        x = training[cols]
        print(cols)
        y = training['prognosis']
        symptoms=[]

        with open('actions/storesymptoms.txt') as file:
            symptoms = [line.rstrip() for line in file]
        symind=dict()
        i=0
        for sym in cols:
            symind[sym]=i
            i=i+1
        print(symptoms)
        y_user=np.zeros(len(cols))

        ip=[]
        for sym in symptoms:
            mini=100
            #print(sym)
            match='no'
            for rs in cols.tolist():
                #print(rs)
                valu=editDistDP(sym,rs,len(sym),len(rs))
                if valu<mini:
                    match=rs
                    mini=valu
            #print(':',match,mini)
            ip.append(match)
        print(ip)
        for s in ip:
            ind=symind[s]
            y_user[ind]=1
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        # Cv=[]
        # ac=[]
        # c=0.003

        # while c<0.03:
        #     clf=svm.SVC(C=c)
        #     clf.fit(x_train,y_train)
        #     sc=clf.score(x_test,y_test)
        #     Cv.append(c)
        #     ac.append(sc)
        #     c=c+0.001

        clf=svm.SVC(C=0.0175)
        clf.fit(x_train,y_train)

        yte=clf.predict(y_user.reshape(1,-1))
        p1=yte

        symdesc.columns=['Disease','Description']

        record=symdesc[symdesc['Disease']==yte[0]]
        # print(record)
        diseaseDescription=record['Description'].values[0]
        print('The disease description :',diseaseDescription)

        p1+="\nThe disease description: "
        p1+=diseaseDescription

        

        symprec.columns=['Disease','one','two','three','four']

        re=symprec[symprec['Disease']=='Heart attack']
        print('dup:',re)
        print(yte[0])
        precaution=symprec[symprec['Disease']==yte[0]]
        print(precaution)
        print('The precautions are:')
        p1+='\nThe precautions are:'
        
        i=1
        pre=precaution['one'].values[0]
        print(pre)

        if pre!='nan':
            print(str(i),')',pre,sep='')
            p1+="\n"
            p1+=(pre)
            i=i+1
        
        pre=precaution['two'].values[0]
        if pre!='nan':
            print(str(i),')',pre,sep='')
            p1+="\n"
            p1+=(pre)
            i=i+1
        pre=precaution['three'].values[0]
        if pre!='nan':
            print(str(i),')',pre,sep='')
            p1+="\n"
            p1+=(pre)
            i=i+1

        pre=precaution['four'].values[0]
        if pre!='nan':
            print(str(i),')',pre,sep='')
            p1+="\n"
            p1+=(pre)
            i=i+1

        dimensionality_reduction = training.groupby(training['prognosis']).max()
        diseases = dimensionality_reduction.index
        diseases = pd.DataFrame(diseases)
        docdata.columns=['Doctor','Link']
        docdata['Disease'] = diseases['prognosis']

        print('Are you feeling high severity of your disease? Enter yes/no')
        query=input()
        if query.lower()=='yes':
            record=docdata[docdata['Disease']==yte[0]]
            print('Doctor name:',record['Doctor'].values[0])
            print('Vist link:',record['Link'].values[0])
        else:
            print('follw precautions')

        dispatcher.utter_message(p1)

        return []
        