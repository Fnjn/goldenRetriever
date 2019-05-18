from django import forms

class UploadFileForm(forms.Form):
    #title = forms.CharField(max_length=50)
    file = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='Select files'
    )

class InsertUploadForm(forms.Form):
    label = forms.CharField(label="Image's Label", max_length=100)

    file = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='Select files'
    )
