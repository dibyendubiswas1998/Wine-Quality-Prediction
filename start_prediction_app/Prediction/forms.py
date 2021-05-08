from django import forms
from .models import DataColumn, Path


class DataColumnForm(forms.ModelForm):
    class Meta:
        model = DataColumn
        fields = "__all__"
        widgets = {
            'col1': forms.NumberInput(attrs={'class': 'myclass', 'placeholder': 'Fixed Acidity'}),
            'col2': forms.NumberInput(attrs={'class': 'myclass', 'placeholder': 'Citric Acidity'}),
            'col3': forms.NumberInput(attrs={'class': 'myclass', 'placeholder': 'Chlorides'}),
            'col4': forms.NumberInput(attrs={'class': 'myclass', 'placeholder': 'PH'}),
            'col5': forms.NumberInput(attrs={'class': 'myclass', 'placeholder': 'Sulphates'}),
            'col6': forms.NumberInput(attrs={'class': 'myclass', 'placeholder': 'Alcohol'}),
        }
        labels = {
            'col1': 'Fixed Acidity',
            'col2': 'Citric Acidity',
            'col3': 'Chlorides',
            'col4': 'PH',
            'col5': 'Sulphates',
            'col6': 'Alcohol'
        }


class PathForm(forms.ModelForm):
    class Meta:
        model = Path
        fields = "__all__"
        widgets = {
            'path': forms.TextInput(attrs={'class': 'myclass', 'placeholder': "Eg:  E:\\folder1\\folder2\\folder3\\data.csv"}),
        }
