from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.urls import reverse
from .models import Audio
from .forms import AudioForm
from .predict_nn import wav_nn_predict

def FileHandler(request):
    message = 'Upload a music file to find its key signature, chord progressions, and MIDI!'
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES)
        if form.is_valid():
            newaudio = Audio(docfile = request.FILES['docfile'])
            newaudio.save()
            output = wav_nn_predict(request.FILES['docfile'])
            return redirect(reverse('FileHandler'))
        message = 'Invalid file!'
    form = AudioForm()

    files = Audio.objects.all()
    context = {'files': files, 'form': form, 'message': message}

    return render(request, 'upload.html', context)