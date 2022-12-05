import numpy as np
import random

## WATERSHEDLIKECLUSTER class
# performs clustering by flooding
# custom clusterization by the test3 task
# understood as flooding
# whenever a new cell is added to the cluster, all of its neightbours are
# added to the cluster, triggering flood of new cells, which neighbours
# will also be added
# algorithm results are reproducible to a degree since the anchor cells
# are randomly chosen.
# algorithm is highly dependant on distance metric and threshold that
# determines the neighbours

class Watershedlikecluster:
    def __init__(self, num_of_clusters):
        self.num_of_clusters = num_of_clusters

    ## FIT
    # @brief Cluster the available data
    # @param dataset - list of numbers range(0, number of cells)
    # @param distance_matrix - matrix of distance metric for all cell combinations
    # @param thereshold - maximum distance for two cells to be in the same cluster
    def fit(self, dataset, distance_matrix, threshold):
        clusters = [[] for _ in range(self.num_of_clusters)]
        curr_cluster = 0
        available_cells = [x for x in dataset]
        while (len(available_cells) > 0 and curr_cluster < self.num_of_clusters):
            # start with random cell from available cells and assign it to a new cluster
            new_cell = random.choice(available_cells)
            cells_to_process = [new_cell]
            while (len(cells_to_process) > 0):
                # define current cell anylized
                curr_cell = cells_to_process[0]
                # extract cells with distance to curr cell smaller than threshold
                close_cells = np.array(dataset)[distance_matrix[curr_cell, :] < threshold]
                # add only cells that are not yet in a cluster for further processing
                diff_from_processing = list(set(close_cells) - set(clusters[curr_cluster]) - set(cells_to_process))
                # for cell in close_cells:
                #     if (cell not in clusters[curr_cluster]) and (cell not in cells_to_process):
                #         cells_to_process.append(cell)
                cells_to_process = cells_to_process + diff_from_processing
                # add current cell to current cluster
                clusters[curr_cluster].append(curr_cell)

                # remove processed cell from all avalable cells and 
                # from current cells_to process for this cluster
                cells_to_process.remove(curr_cell)
                available_cells.remove(curr_cell)


            curr_cluster += 1
        
        curr_cluster -= 1

        # if we do not have enough clusters label leftover cells as extra class
        if (len(available_cells) > 0):
            clusters.append([x for x in available_cells]) # probably could have just appended available_cells
            curr_cluster += 1

        # assign labels of cells
        labels = np.zeros(shape=np.array(dataset).shape)
        for i in range(len(clusters)):
            labels[np.array(clusters[i])] = i

        # return lables
        return labels
