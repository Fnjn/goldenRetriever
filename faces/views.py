from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from django.views import generic
from .forms import  UploadFileForm, InsertUploadForm
from .models import UploadFile
from faces.compareFaces import CompareFaces
from faces.alignFaces_mtcnn import AlignFaces_mtcnn
from faces.searchFaces import SearchFaces

alignFaces = AlignFaces_mtcnn()
compareFaces = CompareFaces()
searchFaces = SearchFaces()

def train(request):
    searchFaces.train(alignFaces, compareFaces)
    searchFaces.saveTree()
    return HttpResponse("The training operation is successful")

def index(request):
    return render(request, 'faces/index.html')

def insertSuccess(request):
    return render(request, 'faces/insertSuccess.html')

def upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            instance = UploadFile(file_field=request.FILES['file'])
            instance.save()
            return HttpResponseRedirect(reverse('faces:uploadSuccess'))
    else:
        form = UploadFileForm()
    return render(request, 'faces/upload.html', {'form': form})

def uploadSuccess(request):
    return HttpResponse("Your upload operation is successful")


class DispalyImgView(generic.ListView):
    model = UploadFile
    template_name = 'faces/display_img.html'
    context_object_name = 'img_list'

    def get_queryset(self):
        return [obj.file_field.url.split("media/",1)[1] for obj in UploadFile.objects.filter(file_field__endswith='jpg')]

compareFiles = []
class compareUploadView(generic.edit.FormView):
    form_class = UploadFileForm
    template_name = 'faces/compareUpload.html'
    success_url = '/faces/compare'

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file')
        global compareFiles
        compareFiles = []
        if form.is_valid():
            for f in files:
                instance = UploadFile(file_field = f)
                instance.save()
                compareFiles.append(instance.file_field.url.split("/")[-1])
            return self.form_valid(form)
        else:
            return self.form_invalid(form)


searchFiles = []
class searchUploadView(generic.edit.FormView):
    form_class = UploadFileForm
    template_name = 'faces/searchUpload.html'
    success_url = '/faces/search'

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file')
        global searchFiles
        searchFiles = []
        if form.is_valid():
            for f in files:
                instance = UploadFile(file_field = f)
                instance.save()
                searchFiles.append(instance.file_field.url.split("/")[-1])
            return self.form_valid(form)
        else:
            return self.form_invalid(form)

class insertUploadView(generic.edit.FormView):
    form_class = InsertUploadForm
    template_name = 'faces/insertUpload.html'
    success_url = '/faces/insertsuccess'

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file')
        insertFiles = []
        insertLabels = []
        if form.is_valid():
            insertLabels.extend(form.cleaned_data['label'].split(';'))
            for f in files:
                instance = UploadFile(file_field = f)
                instance.save()
                insertFiles.append(instance.file_field.url.split("/")[-1])

            alignFaces.loadConfig();
            compareFaces.loadConfig();
            status, imgDict = alignFaces.main(insertFiles)
            if status == 2:
                return HttpResponse("No face recognized.")
            else:
                searchFaces.loadTree()
                emb_array = compareFaces.computeOne(imgDict['align_filenames'])
                searchFaces.insert(emb_array, insertLabels)
                return self.form_valid(form)
        else:
            return self.form_invalid(form)


def compare(request):
    alignFaces.loadConfig();
    compareFaces.loadConfig();
    status, imgDict = alignFaces.main(compareFiles)
    if status == 2:
        return HttpResponse("No face recognized.")
    res, predict = compareFaces.compareTwo(imgDict['align_filenames'])
    alignFaces.drawRec(predict, imgDict['src_paths'], imgDict['boxes'])

    if res == True:
        text = "Found Same Pet"
    else:
        text = "Not Found Same Pet"
    import os
    srcFiles = [os.path.join("UploadedFiles", cfile) for cfile in compareFiles]
    return render(request, 'faces/compareResult.html', {'text': text, 'img_list': srcFiles})

def search(request):
    alignFaces.loadConfig();
    compareFaces.loadConfig();
    status, imgDict = alignFaces.main(searchFiles)
    if status == 2:
        return HttpResponse("No face recognized.")

    searchFaces.loadTree()
    emb = compareFaces.computeOne(imgDict['align_filenames'])
    result = searchFaces.search(searchFaces.root, emb[0])
    return HttpResponse(sorted(result, key=lambda s: s[1], reverse=True))
