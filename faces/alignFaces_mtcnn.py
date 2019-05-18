from scipy import misc
import sys
import os
from configparser import ConfigParser
from django.conf import settings
from PIL import Image, ImageDraw, ImageColor
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random

class AlignFaces_mtcnn:

    def __init__(self, section='align-mtcnn'):
        parser = ConfigParser();
        parser.read('faces/faces.conf')

        self.unalignedDir = getattr(settings, "MEDIA_ROOT") + parser.get(section, "unalignedDir")
        self.alignedDir = getattr(settings, "MEDIA_ROOT") + parser.get(section, "alignedDir")
        self.image_size = parser.getint(section, "image_size")
        self.margin = parser.getint(section, "margin")
        self.random_order = parser.getboolean(section, "random_order")
        self.gpu_memory_fraction = parser.getfloat(section, "gpu_memory_fraction")
        self.has_class_dir = parser.getboolean(section, "has_class_dir")

        print('Creating networks and loading parameters')

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction, allow_growth=True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)

        self.minsize = 20 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor

    def loadConfig(self, section='align-mtcnn'):
        parser = ConfigParser();
        parser.read('faces/faces.conf')

        self.unalignedDir = getattr(settings, "MEDIA_ROOT") + parser.get(section, "unalignedDir")
        self.alignedDir = getattr(settings, "MEDIA_ROOT") + parser.get(section, "alignedDir")
        self.image_size = parser.getint(section, "image_size")
        self.margin = parser.getint(section, "margin")
        self.random_order = parser.getboolean(section, "random_order")
        self.has_class_dir = parser.getboolean(section, "has_class_dir")

    def main(self, argv=''):
        filenames = argv[:]
        alignedDir = os.path.expanduser(self.alignedDir)
        if not os.path.exists(alignedDir):
            os.makedirs(alignedDir)

        src_path,_ = os.path.split(os.path.realpath(__file__))
        dataset = facenet.get_dataset(self.unalignedDir, self.has_class_dir, filenames)

        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)

        status = 0

        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if self.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(alignedDir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if self.random_order:
                    random.shuffle(cls.image_paths)
            boxes = []
            align_filenames = []
            src_paths = []
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename)
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:,0:3]

                        bounding_boxes, _ = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces>0:
                            dets = bounding_boxes[:,0:4]
                            img_size = np.asarray(img.shape)[0:2]

                            box = []
                            align_filename = []
                            for index in range(nrof_faces):
                                det = dets[index,:]
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(det[0]-self.margin/2, 0)
                                bb[1] = np.maximum(det[1]-self.margin/2, 0)
                                bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
                                bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
                                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                                nrof_successfully_aligned += 1
                                align_file = output_filename + str(index) + '.png'
                                misc.imsave(align_file, scaled)

                                box.append([bb[0], bb[1], bb[2], bb[3]])
                                align_filename.append(align_file.split("/")[-1].split(".")[0])

                            boxes.append(box)
                            align_filenames.append(align_filename)
                            src_paths.append(image_path)
                        else:
                            print('Unable to align "%s"' % image_path)
                            status = 2



        imgDict = {'boxes':boxes, 'align_filenames':align_filenames, 'src_paths':src_paths}
        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
        return status, imgDict

    def drawRec(self, predict, src_paths, boxes):
        for i in range(len(predict)):
            with Image.open(src_paths[i]) as im:
                draw = ImageDraw.Draw(im)
                for j in range(len(predict[i])):
                    if predict[i][j]:
                        draw.rectangle(boxes[i][j], outline=ImageColor.getrgb('green'))
                    else:
                        draw.rectangle(boxes[i][j], outline=ImageColor.getrgb('yellow'))
                del draw
                im.save(src_paths[i], "PNG")
