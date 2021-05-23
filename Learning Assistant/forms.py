from django import forms
from .models import QP
class QuestionForm(forms.ModelForm):
    # paragraph = forms.TextInput()
    # question = forms.CharField()
    class Meta:
        model = QP
        fields = ['paragraph','question']