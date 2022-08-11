from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.urls import reverse
from .models import Audio
from .forms import AudioForm
from .predict_nn import wav_nn_predict2

def FileHandler(request):
    message = 'Upload a music file to convert it into a transcribed MIDI file!'
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES)
        if form.is_valid():
            if request.FILES['docfile'].temporary_file_path()[-3:] in ['mp3', 'ogg', 'wav']:
                newaudio = Audio(docfile = request.FILES['docfile'])
                newaudio.save()
                output = wav_nn_predict2(request.FILES['docfile'].temporary_file_path())
                return redirect(reverse('Confirmation'))
            else:
                message = 'Invalid file! File must be .wav, .ogg, or .mp3!'
        else:
            message = 'Invalid file!'
    form = AudioForm()

    files = Audio.objects.all()
    context = {'files': files, 'form': form, 'message': message}

    return render(request, 'upload.html', context)

def Confirmation(request):
    return HttpResponse('Processing...')
