# Golden Retriever - A Lost Pet Recognition Project

For Phase 1:
This project is created using two Neural networks. The first uses IBM’s cloud visual recognition technology to categorize dog breeds in our database, the second looks to facially recognize the pet based on other lost pets of the same breed


For Using the IBM cloud visual recognition use API key : 
YSIvQQnYt6d3ay9N4eAUiLu2Bx-PAAKCm2YLeT601ggw
At URL:
https://gateway.watsonplatform.net/visual-recognition/api

To classify the image run:
curl -u "apikey:{apikey}" "https://gateway.watsonplatform.net/visual-recognition/api/v3/classify?url=https://watson-developer-cloud.github.io/doc-tutorial-downloads/visual-recognition/fruitbowl.jpg&version=2018-03-19&classifier_ids=DefaultCustomModel_1285354297"

To do facial matching:

## Installation
```bash
git clone git@github.com:Fnjn/goldenRetriever.git
pip2 install -r requirements.txt

git clone git@github.com:davidsandberg/facenet.git
export PYTHONPATH=[path-to-facenet]/src:$PYTHONPATH
# Add namescope to facenet.create_input_pipeline
# See https://github.com/davidsandberg/facenet/issues/852#issuecomment-431420493
```

## Run
```bash
cd goldenRetriever
python2 manage.py runserver
```

## Configure
Configure file: faces/faces.conf <br>
Default image search directory: media/SearchDir

In phase 2 we plan to implement IBM analytics to web crawl though twitter and find tweets about lost pets and add that information to our database
