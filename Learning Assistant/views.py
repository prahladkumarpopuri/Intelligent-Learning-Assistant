from django.shortcuts import render, redirect
from .forms import QuestionForm
from .BERTSQuADmaster.bert import QA
from .Translate.translate import englishToHindiTranslate
# Create your views here.
def basic(request):
    if request.method =='POST':
        question1 = request.POST['question1']
        question2 = request.POST['question2']
        para = request.POST['paragraph']
        print(question1)
        model = QA("/Users/rithwik/Desktop/fyp/bert_translate/BERTSQuADmaster/model")

        answer1 = model.predict(para,question1)
        answer2 = model.predict(para,question2)

        if answer1["confidence"] < .6:
            english_text1 = "No answer found"
        else:
            english_text1 = answer1["answer"]
        print(answer1["answer"])
        print(answer1["confidence"])
        if answer2["confidence"] < .6:
            english_text2 = "No answer found"
        else:
            english_text2 = answer2["answer"]
        print(answer2["answer"])
        print(answer2["confidence"])



        hindi1 = englishToHindiTranslate(english_text1)
        hindi2 = englishToHindiTranslate(english_text2)

        
        return render(request,'Feedback_Form.html',{'para':para,'question1':question1,'ans1':english_text1,'ans1t':hindi1.convert,'question2':question2,'ans2':english_text2,'ans2t':hindi2.convert})
    else:
        form = QuestionForm()
    return render(request,'Feedback_Form.html',{'form':form})

