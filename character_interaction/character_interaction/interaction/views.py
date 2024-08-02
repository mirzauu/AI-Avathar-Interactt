from django.shortcuts import render
import json
# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from transformers import pipeline
from g2p_en import G2p

g2p = G2p()
nlp_model = pipeline('sentiment-analysis')

def process_text(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text')
        
        # Generate response using your NLP model
        response_text = nlp_model(text)[0]['label']
        
        # Convert response text to phonemes
        phonemes = ' '.join(g2p(response_text))
        
        return JsonResponse({"response_text": response_text, "phonemes": phonemes})
    return render(request, 'index.html')


def index(request):
    return render(request, 'interaction/index.html')