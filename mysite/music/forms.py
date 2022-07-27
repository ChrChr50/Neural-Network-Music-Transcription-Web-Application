from django import forms

class AudioForm(forms.Form):
    docfile = forms.FileField(
        label = 'Select a file',
        help_text = 'File must be .wav, .ogg, or .mp3'
    )