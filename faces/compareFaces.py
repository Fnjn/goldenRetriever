import tensorflow as tf
import numpy as np
import facenet
from faces import alignedDataset
import os
import sys
import math
from configparser import ConfigParser
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from django.conf import settings

class CompareFaces(object):

    def __init__(self, section='facenet'):
        parser = ConfigParser();
        parser.read('faces/faces.conf')

        self.alignedDir = getattr(settings, "MEDIA_ROOT") + parser.get(section, "alignedDir")
        self.batch_size = parser.getint(section, "batch_size")
        self.model = getattr(settings, "STATIC_ROOT") + parser.get(section, "model")
        self.image_size = parser.getint(section, "image_size")
        self.file_ext = parser.get(section, "file_ext")
        self.threshold = parser.getfloat(section, "threshold")

        self.sess = tf.Session()
        facenet.load_model(self.sess, self.model)
        self.graph = tf.get_default_graph()

    def loadConfig(self, section='facenet'):
        parser = ConfigParser();
        parser.read('faces/faces.conf')

        self.alignedDir = getattr(settings, "MEDIA_ROOT") + parser.get(section, "alignedDir")
        self.batch_size = parser.getint(section, "batch_size")
        self.image_size = parser.getint(section, "image_size")
        self.file_ext = parser.get(section, "file_ext")
        self.threshold = parser.getfloat(section, "threshold")

    def computeEmbeddings(self, paths):
        # Get input and output tensors
        self.images_placeholder = self.graph.get_tensor_by_name("input:0")
        self.embeddings = self.graph.get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        print('Runnning forward pass on images')
        nrof_images = len(paths)
        nrof_batches = int(math.ceil(1.0*nrof_images / self.batch_size))
        emb_array = np.zeros((nrof_images, self.embedding_size))

        for i in range(nrof_batches):
            start_index = i * self.batch_size
            end_index = min((i+1) * self.batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, self.image_size)
            feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
            emb_array[start_index:end_index,:] = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return emb_array

    def computeOne(self, filenames):
        paths = alignedDataset.get_paths(os.path.expanduser(self.alignedDir), filenames, self.file_ext)
        emb_array = self.computeEmbeddings(paths)
        return emb_array


    def compareTwo(self, filenames):
        if len(filenames) < 2:
            print ("Must input at least 2 images")
            return

        paths, imgNum = alignedDataset.get_compare_paths(os.path.expanduser(self.alignedDir), filenames, self.file_ext)
        emb_array = self.computeEmbeddings(paths)

        predict_issame, dist = alignedDataset.evaluate(emb_array, self.threshold)
        print ('Predictions: ', predict_issame)
        print ('Distances: ', dist)
        predict = [[False] * len (filenames[0]), [False] * len (filenames[1])]
        for i in range(len(predict_issame)):
            if predict_issame[i] == True:
                print (imgNum[i][0])
                predict[0][imgNum[i][0]] = True;
                predict[1][imgNum[i][1]] = True;
        return any(predict_issame) == True, predict

    def __del__(self):
        self.sess.close()
