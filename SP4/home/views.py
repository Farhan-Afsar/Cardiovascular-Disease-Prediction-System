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

    intput1 = int(request.GET.get('age'))
    intput2 = int(request.GET.get('gender'))
    intput3 = int(request.GET.get('height'))
    intput4 = float(request.GET.get('weight'))
    intput5 = int(request.GET.get('ap_hi'))
    intput6 = int(request.GET.get('ap_lo'))
    intput7 = int(request.GET.get('cholesterol'))
    intput8 = int(request.GET.get('glucose'))
    intput9 = int(request.GET.get('smoke'))
    intput10 = int(request.GET.get('alco'))
    intput11 = int(request.GET.get('active'))

    pred = KN.predict([[intput1, intput2, intput3, intput4, intput5, intput6, intput7, intput8, intput9, intput10, intput11]])

    result1 = " "

    if pred == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request,'predict.html',{"result2": result1})
