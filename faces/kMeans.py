import tensorflow as tf
import numpy as np

class KMeans:

    def choose_random_centroids(self, samples, n_clusters):
        n_sample = tf.shape(samples)[0]
        random_indices = tf.random_shuffle(tf.range(0, n_sample))
        begin = [0,]
        size = [n_clusters,]
        size[0] = n_clusters
        centroid_indices = tf.slice(random_indices, begin, size)
        initial_centroids = tf.gather(samples, centroid_indices)
        return initial_centroids

    def assign_to_nearest(self, samples, centroids):
        expanded_vectors = tf.expand_dims(samples, 0)
        expanded_centroids = tf.expand_dims(centroids, 1)
        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors,expanded_centroids)), 2)
        nearest_indices = tf.argmin(distances, 0)
        return nearest_indices

    def update_centroids(self, samples, labels, nearest_indices, n_clusters):
        nearest_indices = tf.to_int32(nearest_indices)
        partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
        part_labels = tf.dynamic_partition(labels, nearest_indices, n_clusters)
        new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
        return new_centroids, partitions, part_labels

    def main(self, samples_feed, labels_feed, n_clusters, n_features, iter = 10):
        samples = tf.placeholder(tf.float32, shape=[None, n_features], name="samples")
        labels = tf.placeholder(tf.string, name="labels")
        centroids = tf.placeholder(tf.float32, shape=[None, n_features], name="centroids")
        init_centroids = self.choose_random_centroids(samples, n_clusters)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)) as sess:
            sess.run(tf.global_variables_initializer())
            centroids_feed = sess.run(init_centroids, feed_dict={samples: samples_feed})
            for i in range(iter):
                nearest_indices = self.assign_to_nearest(samples, centroids)
                updated_centroids = self.update_centroids(samples, labels, nearest_indices, n_clusters)
                centroids_feed, partitions, part_labels = sess.run(updated_centroids, feed_dict={samples: samples_feed, labels: labels_feed, centroids: centroids_feed})
        return centroids_feed, partitions, part_labels
