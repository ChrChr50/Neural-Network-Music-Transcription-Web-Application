from django.shortcuts import redirect, render
from django.http import HttpResponse
from django.urls import reverse
from .models import Audio
from .forms import AudioForm

def FileHandler(request):
    message = 'Test'
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES)
        if form.is_valid():
            newaudio = Audio(docfile = request.FILES['docfile'])
            newaudio.save()
            return redirect(reverse('FileHandler'))
        message = 'Invalid file'
    form = AudioForm()

    files = Audio.objects.all()
    context = {'files': files, 'form': form, 'message': message}

    return render(request, 'upload.html', context)