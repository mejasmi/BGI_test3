import numpy as np
from numpy.random import uniform
import random

class TCluster:
    def __init__(self, n_clusters=10, max_iter=10000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    ## FIT
    # @brief Cluster the available data
    # @param dataset - list of numbers range(0, number of cells)
    # @param distance_matrix - matrix of distance metric for all cell combinations
    # @param thereshold - maximum distance for two cells to be in the same cluster
    def fit(self, dataset, distance_matrix, threshold):
        # Pick random point for first cluster
        clusters = [[] for _ in range(self.n_clusters)]
        curr_cluster = 0
        
        available_cells = [x for x in dataset]
        # all cells need to be assigned to a cluster
        while (self.n_clusters > curr_cluster):
            # flag for cluster merger
            cluster_merged = False
            # start with random cell from available cells and assign it to a new cluster
            curr_cell = random.choice(available_cells)
            # all cells with distance smaller than threshold from the active cell are added to the cluster
            close_cells = np.array(dataset)[distance_matrix[curr_cell, :] < threshold]
            # check if the first neighbourhood of the cell contains some points of other cluster
            for cluster in clusters:
                if (len(list(set(close_cells) & set(cluster))) > 0):
                    # if true, add all these to the cluster and continue
                    for cl in close_cells:
                        if (cl not in cluster):
                            cluster.append(cl)
                    cluster_merged = True
                    # exit the for cluster loop
                    break
            if cluster_merged:
                # continue to while
                continue
            # if there is no cluster merger, add cells to a new cluster
            for cl in close_cells:
                clusters[curr_cluster].append(cl)
            # remove assigned cells from list of still available ones
            for cl in close_cells:
                available_cells.remove(cl)
            # go to next cluster initialization
            curr_cluster += 1

        num_of_miss = 0
        # assign each of the rest of the cells to a cluster
        while (len(available_cells) > 0 and num_of_miss< self.max_iter):
            # pick a random cell
            curr_cell = random.choice(available_cells)
            # find its allowed neighbourhood
            close_cells = np.array(dataset)[distance_matrix[curr_cell, :] < threshold]
            # check if any of these neighbours already belong to a cluster and extract the distance
            intersections = []
            for cluster_id in range(self.n_clusters):
                neighbours_in_clusters = list(set(close_cells) & set(clusters[cluster_id]))
                if len(neighbours_in_clusters) > 0:
                    for neigh in neighbours_in_clusters:
                        intersections.append((cluster_id, distance_matrix[curr_cell, neigh]))
            # if there are any neighbours in clusters find which one is closest to the cell
            # if not, increment num_of_miss and start again
            if (len(intersections) > 0):
                # sort to find lowest distance
                intersections.sort(key=lambda k: k[1], reverse=False)
                # add current cell to the cluster of the lowest distance
                clusters[intersections[0][0]].append(curr_cell)
                # remove current cell from pool
                available_cells.remove(curr_cell)
            else:
                num_of_miss += 1
        # if we stopped clustering but there are still unassigned cells, add them to an extra cluster
        if (len(available_cells) > 0):
            clusters.append(available_cells)
        # assign labels of cells
        labels = np.zeros(shape=np.array(dataset).shape)
        for i in range(len(clusters)):
            labels[np.array(clusters[i])] = i

        # return lables
        return labels