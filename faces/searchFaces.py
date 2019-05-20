import tensorflow as tf
import numpy as np
import os
import math
from faces.kMeans import KMeans
from django.conf import settings
from configparser import ConfigParser
from faces.alignFaces_mtcnn import AlignFaces_mtcnn
from faces.compareFaces import CompareFaces
from faces import alignedDataset

class SearchFaces:

    def __init__(self, section='searchFaces'):
        parser = ConfigParser();
        parser.read('faces/faces.conf')

        self.alignedDir = getattr(settings, "MEDIA_ROOT") + parser.get(section, "alignedDir")
        self.threshold = parser.getfloat(section, "threshold")
        self.search_tree = parser.get(section, "search_tree")


        self.root = None
        self.kMeans = KMeans()
        self.iter = 40

    def loadTree(self):
        import pickle
        with open(self.search_tree, 'rb') as f:
            self.root = pickle.load(f)

    def saveTree(self):
        import pickle
        with open(self.search_tree, 'wb') as f:
            pickle.dump(self.root, f, protocol=2)

    def train(self, alignFaces=None, compareFaces=None):
        if not alignFaces:
            alignFaces = AlignFaces_mtcnn('align-search')
        else:
            alignFaces.loadConfig('align-search')
        if not compareFaces:
            compareFaces = CompareFaces('searchFaces')
        else:
            compareFaces.loadConfig('searchFaces')

        alignFaces.main()
        paths, labels = alignedDataset.get_search_paths(os.path.expanduser(self.alignedDir))
        embeddings = compareFaces.computeEmbeddings(paths)

        self.root = self.computeNode(embeddings, labels)

    def insert(self, embeddings, labels):
        for i, emb in enumerate(embeddings):
            self.searchInsert(self.root, emb, labels[i])
        self.saveTree()

    def searchInsert(self, node, emb, label):
        if node.isLeaf:
            emb = np.expand_dims(emb,axis=0)
            node.children = np.vstack((node.children, emb))
            node.labels = np.append(node.labels, label)
        else:
            centNode = self.findNearestN(node, emb, 1)[0]
            self.searchInsert(centNode, emb, label)


    def computeNode(self, embeddings, labels, value=None):
        import math
        if embeddings.shape[0] < 50:
            return Node(True, value, embeddings, labels)

        else:
            node = Node(False, value, [])
            n_clusters = math.ceil(math.sqrt(embeddings.shape[0]//2))
            n_features = embeddings.shape[1]
            centroids, partitions, part_labels = self.kMeans.main(embeddings, labels, n_clusters, n_features, self.iter)

            for i, centroid in enumerate(centroids):
                node.children.append(self.computeNode(partitions[i], part_labels[i], centroid))
            return node

    def computeScaledDist(self, emb1, emb2):
        dist = self.computeDist(emb1, emb2)
        if dist < 0.6:
            res = 1
        elif dist < self.threshold:
            k = -0.25 / (self.threshold - 0.5)
            res = k * (dist-0.5) + 1
        elif dist < 3.0:
            k = - 0.75 / (3.0 - self.threshold)
            res = k * (dist-self.threshold) + 0.75
        else:
            res = 0
        return res


    def computeDist(self, emb1, emb2):
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff))
        return dist

    def search(self, node, emb, n=3):
        if node.isLeaf:
            return [(node.labels[i], self.computeScaledDist(emb, child)) for i, child in enumerate(node.children) if self.computeDist(emb, child) < self.threshold]
        else:
            centNodes = self.findNearestN(node, emb, n)
            ret = []
            for centNode in centNodes:
                ret.extend(self.search(centNode, emb, n))
            return list(filter(None, ret))

    def findNearestN(self, node, emb, n):
        import heapq
        small = heapq.nsmallest(n, node.children, key=lambda s: self.computeDist(emb,s.value))
        return [s for s in small if self.computeDist(emb,s.value) < 1.5]


class Node:

    def __init__(self, isLeaf, value, children, labels=None):
        self.isLeaf = isLeaf
        self.value = value
        self.children = children
        self.labels = labels
