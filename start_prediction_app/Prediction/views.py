from django.shortcuts import render, HttpResponse, HttpResponseRedirect
from .forms import DataColumnForm
from sklearn.preprocessing import StandardScaler
import pickle

"""
    Before Model Serving:
    
    step1: Prepare your data (Data_Preparation),
    step2: Train the model based on different different algorithms (Model_Traning),
    step3: Evaluate the best model and get the accuracy score (Model_Evaluation),
    step4: Based on finalized_model, get the output.
    
"""


def Predict_White_Wine(request):
    try:
        if request.method == 'POST':
            fm = DataColumnForm(request.POST)
            if fm.is_valid():
                # load all the data
                Fixed_Acidity = fm.cleaned_data['col1']
                Citric_Acidity = fm.cleaned_data['col2']
                Chlorides = fm.cleaned_data['col3']
                PH = fm.cleaned_data['col4']
                Sulphates = fm.cleaned_data['col5']
                Alcohol = fm.cleaned_data['col6']

                # load the finalized_model for predict
                scaler = StandardScaler()
                # path = "../Wine Quality Predictions/Saved_Model/finalized_model.pickle"
                path1 = "E:\\Dibyendu\\Projects\\1. Machine Learning Projects\\Wine Quality Predictions\\Saved_Model\\finalized_model.pickle"
                with open(path1, 'rb') as f:
                    model = pickle.load(f)
                    print(model)
                    predct = model.predict(scaler.fit_transform([[Fixed_Acidity, Citric_Acidity, Chlorides,
                                                                  PH, Sulphates, Alcohol]]))
                print("Model Name: ", model)
                fm = DataColumnForm()
                if predct == 1:
                    return HttpResponseRedirect('/good/')
                else:
                    return HttpResponseRedirect('/bad/')

        else:
            fm = DataColumnForm()
        return render(request, 'index.html', {'form': fm})

    except Exception as e:
        print(e)


def Result_Bad(request):
    return render(request, 'precidtlow.html', {'result': 0})


def Result_Good(request):
    return render(request, 'precidtgood.html', {'result': 0})
