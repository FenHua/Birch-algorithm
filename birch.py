#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import warnings
import numpy as np
from math import sqrt
from sklearn.utils import check_array
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms,safe_sparse_dot
from sklearn.cluster.hierarchical import AgglomerativeClustering

def _split_node(node, threshold, branching_factor):
    new_subcluster1 = _CFSubcluster()
    new_subcluster2 = _CFSubcluster()
    new_node1 = _CFNode(
        threshold, branching_factor, is_leaf=node.is_leaf,
        n_features=node.n_features)
    new_node2 = _CFNode(
        threshold, branching_factor, is_leaf=node.is_leaf,
        n_features=node.n_features)
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2

    dist = euclidean_distances(
        node.centroids_, Y_norm_squared=node.squared_norm_, squared=True)
    n_clusters = dist.shape[0]

    farthest_idx = np.unravel_index(
        dist.argmax(), (n_clusters, n_clusters))
    node1_dist, node2_dist = dist[[farthest_idx]]

    node1_closer = node1_dist < node2_dist
    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return new_subcluster1, new_subcluster2

class _CFSubcluster(object):
    def __init__(self, linear_sum=None):
        if linear_sum is None:
            self.n_samples_ = 0
            self.squared_sum_ = 0.0
            self.linear_sum_ = 0
        else:
            self.n_samples_ = 1
            self.centroid_ = self.linear_sum_ = linear_sum
            self.squared_sum_ = self.sq_norm_ = np.dot(
                self.linear_sum_, self.linear_sum_)
        self.child_ = None

    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.squared_sum_ += subcluster.squared_sum_
        self.centroid_ = self.linear_sum_ / self.n_samples_
        self.sq_norm_ = np.dot(self.centroid_, self.centroid_)

    def merge_subcluster(self, nominee_cluster, threshold):
        """
        检查是否可以合并，条件符合就合并.
        """
        new_ss = self.squared_sum_ + nominee_cluster.squared_sum_
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        new_centroid = (1 / new_n) * new_ls
        new_norm = np.dot(new_centroid, new_centroid)
        dot_product = (-2 * new_n) * new_norm
        sq_radius = (new_ss + dot_product) / new_n + new_norm
        if sq_radius <= threshold ** 2:
            (self.n_samples_, self.linear_sum_, self.squared_sum_,self.centroid_, self.sq_norm_) = new_n, new_ls, new_ss, new_centroid, new_norm
            return True
        return False
    
class _CFNode(object):
    #初始化函数
    def __init__(self, threshold, branching_factor, is_leaf, n_features):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features
        # 列表subclusters, centroids 和 squared norms一直贯穿始终
        self.subclusters_ = []
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features))
        self.init_sq_norm_ = np.zeros((branching_factor + 1))#一维列表
        self.squared_norm_ = []
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_
        self.init_sq_norm_[n_samples] = subcluster.sq_norm_
        # 扩容
        self.centroids_ = self.init_centroids_[:n_samples + 1, :]
        self.squared_norm_ = self.init_sq_norm_[:n_samples + 1]
    
    def update_split_subclusters(self, subcluster,new_subcluster1, new_subcluster2):
        #从一个节点去掉一个subcluster，再添加两个subcluster.
        ind = self.subclusters_.index(subcluster)#找到索引位置
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.init_sq_norm_[ind] = new_subcluster1.sq_norm_
        self.append_subcluster(new_subcluster2)

    def insert_cf_subcluster(self, subcluster):
        #插入一个新的subcluster.
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        # 首先，在树中遍历寻找与当前subcluster最近的subclusters，再将subcluster插入到此处.
        dist_matrix = np.dot(self.centroids_, subcluster.centroid_)# dot矩阵相乘
        print len(self.centroids_)
        dist_matrix *= -2.
        dist_matrix += self.squared_norm_
        closest_index = np.argmin(dist_matrix)
        closest_subcluster = self.subclusters_[closest_index]#距当前点最近的subclusters集
        # 如果closest_subcluster有孩子节点，递归遍历
        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_cf_subcluster(subcluster)
            if not split_child:
                # 如果孩子节点没有分裂，仅需要更新closest_subcluster
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                self.init_sq_norm_[closest_index] = self.subclusters_[closest_index].sq_norm_
                return False

            # 如果发生了分割，需要重新分配孩子节点中的subclusters，并且在其父节点中添加一个subcluster.
            else:
                new_subcluster1, new_subcluster2 = _split_node(closest_subcluster.child_, threshold, branching_factor)
                self.update_split_subclusters(closest_subcluster, new_subcluster1, new_subcluster2)
                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        #没有孩子节点
        else:
            merged = closest_subcluster.merge_subcluster(subcluster, self.threshold)
            if merged:
                #更新操作
                self.init_centroids_[closest_index] =closest_subcluster.centroid_
                self.init_sq_norm_[closest_index] = closest_subcluster.sq_norm_
                return False
            # 待插入点和任何节点相距较远
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)
                return False
            # 如果没有足够的空间或者待插入点与其它点相近，则分裂操作.
            else:
                self.append_subcluster(subcluster)
                return True
            
class Birch():
    #初始化函数
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3,
                 compute_labels=True):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels

    def fit(self, X, y=None):
        threshold = self.threshold
        X = check_array(X, accept_sparse='csr', copy=True)
        branching_factor = self.branching_factor
        if branching_factor <= 1:
            raise ValueError("Branching_factor should be greater than one.")
        n_samples, n_features = X.shape
        #初次建立树，并且root节点是叶子.
        self.root_ = _CFNode(threshold, branching_factor, is_leaf=True,n_features=n_features)
        # 便于恢复subclusters.
        self.dummy_leaf_ = _CFNode(threshold, branching_factor,is_leaf=True, n_features=n_features)
        self.dummy_leaf_.next_leaf_ = self.root_
        self.root_.prev_leaf_ = self.dummy_leaf_
        # 未能向量化. 
        for sample in iter(X):
            subcluster = _CFSubcluster(linear_sum=sample)
            split = self.root_.insert_cf_subcluster(subcluster)
            if split:
                new_subcluster1, new_subcluster2 = _split_node(self.root_, threshold, branching_factor)
                del self.root_
                self.root_ = _CFNode(threshold, branching_factor,is_leaf=False,n_features=n_features)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids

        self._global_clustering(X)
        return self

    def _get_leaves(self):

        #返回CFNode的叶子节点
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves

    def predict(self, X):
        reduced_distance = safe_sparse_dot(X, self.subcluster_centers_.T)
        print reduced_distance
        reduced_distance *= -2
        reduced_distance += self._subcluster_norms
        return self.subcluster_labels_[np.argmin(reduced_distance, axis=1)]
    
    def _global_clustering(self, X=None):

        #对fitting之后获得的subclusters进行global_clustering
        clusterer = self.n_clusters
        centroids = self.subcluster_centers_
        compute_labels = (X is not None) and self.compute_labels

        # 预处理
        not_enough_centroids = False
        if isinstance(clusterer, int):
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
            if len(centroids) < self.n_clusters:
                not_enough_centroids = True
        elif (clusterer is not None):
            raise ValueError("n_clusters should be an instance of " "ClusterMixin or an int")

        # 避免predict环节，重复运算
        self._subcluster_norms = row_norms(
            self.subcluster_centers_, squared=True)

        if clusterer is None or not_enough_centroids:
            self.subcluster_labels_ = np.arange(len(centroids))
            if not_enough_centroids:
                warnings.warn(
                    "Number of subclusters found (%d) by Birch is less than (%d). Decrease the threshold."% (len(centroids), self.n_clusters))
        else:
            # 对所有叶子节点的subcluster进行聚类，它将subcluster的centroids作为样本，并且找到最终的centroids.
            self.subcluster_labels_ = clusterer.fit_predict(
                self.subcluster_centers_)

        if compute_labels:
            self.labels_ = self.predict(X)
