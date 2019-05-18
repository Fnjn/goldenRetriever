# Golden Retriever - A Lost Pet Recognition Project


## Installation
```bash
git clone git@github.com:Fnjn/goldenRetriever.git
pip2 install -r requirements.txt

git clone git@github.com:davidsandberg/facenet.git
# Add namescope to facenet.create_input_pipeline
# See https://github.com/davidsandberg/facenet/issues/852#issuecomment-431420493
```

## Run
```bash
cd goldenRetriever
python2 manage.py runserver
```

## Configure
Configure file: faces/faces.conf
Default image search directory: media/SearchDir
