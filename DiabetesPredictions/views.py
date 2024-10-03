from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def home(request):
    return render(request,'home.html')


def predictionssss(request):
    return render(request,'predictionssss.html')

def result(request):
    data = pd.read_csv("diabetes.csv")

    X = data.drop("Outcome", axis=1)
    Y = data['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    if request.method == 'POST':
        val1 = float(request.POST['n1'])
        val2 = float(request.POST['n2'])
        val3 = float(request.POST['n3'])
        val4 = float(request.POST['n4'])
        val5 = float(request.POST['n5'])
        val6 = float(request.POST['n6'])
        val7 = float(request.POST['n7'])
        val8 = float(request.POST['n8'])

        pred=model.predict([[val1,val2,val3,val4,val5,val6,val7,val8]]) #pred is a variable that can store the value
        # #model is the name of upper#.predict is predefined function

        result1=""
        if pred==[1]:
            result1="Positive"
        else:
            result1="Negative"


        return render(request,'res.html',{"result2":result1})  #same page to you
    return render(request, 'predictionssss.html')

