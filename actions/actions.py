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
        f=open("storesymptoms.txt","a")
        for k in p[0].split(','):
            print(k)
            f.write(k + "\n")
        return []

class ActionDiagnoseySymptoms(Action):

    def name(self) -> Text:
        return "action_diagnose_symptoms"

    def run(self,dispatcher:CollectingDispatcher,tracker:Tracker,domain:Dict[Text,Any]) -> List[Dict[Text,Any]]:
        print("custom code goes here")
        shannuwastefellow= tracker.get_slot('symptom')
        r3=tracker.latest_message.get('text')
        
        message="Your symptoms are "
        
        f=open('storesymptoms.txt','r')
        ls=[]
        lt=f.readlines()
        for ln in lt:
            ls.append(ln)


        training = pd.read_csv('C:/Users/KATAKAM19/Desktop/rasa/actions/Training.csv')
        testing= pd.read_csv('C:/Users/KATAKAM19/Desktop/rasa/actions/Testing.csv')
        cols= training.columns
        cols= cols[:-1]
        x = training[cols]
        y = training['prognosis']
        y1= y
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        testx    = testing[cols]
        testy    = testing['prognosis'] 
        symind=dict()
        i=0
        for sym in cols:
            symind[sym]=i
            i=i+1 
        clf=svm.SVC(C=1.0)
        clf.fit(x_train,y_train)
        y_user=np.zeros(len(cols))
        # for s in shannuwastefellow[0].split(','):
        #     if s in symind.keys():
        #         ind=symind[s]
        #         y_user[ind]=1 
        #     else:
        #         print(s)  

        y_u=np.zeros(len(cols))

        for s in ls:
            if s in symind.keys():
                ind=symind[s]
                y_user[ind]=1 
            
        

        yte=clf.predict(y_u.reshape(1,-1))
        
        msg=" "
        for k in yte:
            msg+=k   
        dispatcher.utter_message(text=msg)
        
        return []
        