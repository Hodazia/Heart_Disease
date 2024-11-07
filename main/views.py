from django.shortcuts import render
from django.http import HttpResponse
import joblib
import numpy as np
# Create your views here.
def home(request):
    return render(request,'home.html')

def getPredictions_knn(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13):
    model = joblib.load('models/knn.pkl')
    arr = np.array([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
    prediction = model.predict(arr)
    if prediction == 0:
        return "0"
    elif prediction == 1:
        return "1"
    else:
        return "error"
    
def getPredictions_svm(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13):
    model = joblib.load('models/SVC.pkl')
    arr = np.array([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
    prediction = model.predict(arr)
    if prediction == 0:
        return "0"
    elif prediction == 1:
        return "1"
    else:
        return "error"

def getPredictions_logistic(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13):
    model = joblib.load('models/LR.pkl')
    arr = np.array([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
    prediction = model.predict(arr)
    if prediction == 0:
        return "0"
    elif prediction == 1:
        return "1"
    else:
        return "error"

def getPredictions_random(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13):
    model = joblib.load('models/RF.pkl')
    arr = np.array([[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]])
    prediction = model.predict(arr)
    if prediction == 0:
        return "0"
    elif prediction == 1:
        return "1"
    else:
        return "error"

def result(request):
    p1 = int(request.POST['age'])
    p2 = int(request.POST['sex'])
    p3 = int(request.POST['cp'])
    p4 = int(request.POST['trestbps'])
    p5 = int(request.POST['chol'])
    p6 = int(request.POST['fbs'])
    p7 = int(request.POST['restecg'])
    p8 = int(request.POST['thalach'])
    p9 = int(request.POST['exang'])
    p10 = float(request.POST['oldpeak'])
    p11 = int(request.POST['slope'])
    p12 = int(request.POST['ca'])
    p13 = int(request.POST['thal'])
    res1 = getPredictions_knn(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13)
    res2 = getPredictions_svm(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13)
    res3 = getPredictions_logistic(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13)
    res4 = getPredictions_random(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13)
    context = {
        'result1': res1,
        'result2': res2,
        'result3' : res3,
        'result4' : res4    
        }
    return render(request, 'result.html', context)