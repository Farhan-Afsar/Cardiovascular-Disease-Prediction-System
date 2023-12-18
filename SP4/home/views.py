from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier





def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request,'predict.html')


def result(request):
    dataset = pd.read_csv(r'/Users/farhanafsar/Desktop/SP4/SP4/home/templates/cardio_train.csv',sep = ";")

    dataset.drop('id', axis=1, inplace=True)
    dataset['age'] = (dataset['age'] / 365).astype('int')
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=1)
    KN = KNeighborsClassifier()
    KN.fit(xtrain, ytrain)

    input1 = int(request.GET.get('age'))
    input2 = int(request.GET.get('gender'))
    input3 = int(request.GET.get('height'))
    input4 = float(request.GET.get('weight'))
    input5 = int(request.GET.get('ap_hi'))
    input6 = int(request.GET.get('ap_lo'))
    input7 = int(request.GET.get('cholesterol'))
    input8 = int(request.GET.get('glucose'))
    input9 = int(request.GET.get('smoke'))
    input10 = int(request.GET.get('alco'))
    input11 = int(request.GET.get('active'))

    pred = KN.predict([[input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11]])

    result1 = " "

    if pred == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request,'predict.html',{"result2": result1})
