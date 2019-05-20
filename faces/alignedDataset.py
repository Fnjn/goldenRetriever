import os
import numpy as np

def evaluate(embeddings, threshold=1.1):
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    #dist = np.sum(diff,1)
    predict_issame = np.less(dist, threshold)
    print(dist)
    return predict_issame, dist

def get_paths(facedir, filenames, file_ext):
    path_list = []
    for filename in filenames[0]:
        path = os.path.join(facedir, filename +'.'+file_ext)
        if os.path.exists(path):
            path_list.append(path)
    return path_list

def get_compare_paths(facedir, filenames, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    image_list = []
    for i, filename1 in enumerate(filenames[0]):
        path0 = os.path.join(facedir, filename1 +'.'+file_ext)
        if not os.path.exists(path0):
            return None
        for j, filename2 in enumerate(filenames[1]):
            path1 = os.path.join(facedir, filename2 +'.'+file_ext)
            if os.path.exists(path1):
                path_list += (path0, path1)
                image_list.append([i, j])
            else:
                nrof_skipped_pairs += 1

    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    return path_list, image_list

def get_search_paths(facedir):
    image_paths = []
    labels = []
    if os.path.isdir(facedir):
        classes = os.listdir(facedir)
        for clss in classes:
            clssPath = os.path.join(facedir, clss)
            if os.path.isdir(clssPath):
                for img in os.listdir(clssPath):
                    if os.path.isfile(os.path.join(clssPath, img)):
                        image_paths.append(os.path.join(clssPath, img))
                        labels.append(clss)
    return image_paths, labels
