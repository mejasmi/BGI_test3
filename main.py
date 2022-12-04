import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cell
import pca_fun
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random
import kmeans

colors = ['b', 'g', 'r', 'm', 'c', 'k', 'y', '#884488', '#018888', '#2248FF', '#FF3388', '#934752', '#E7B300', '#9012FE', '#3322CC']

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            objects = pickle.load(f)
    except:
        objects = []
    return objects

def save_data(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def SSD(x1,y1, x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)



def main(filename,
         exp_weight,
         percentage_distance_threshold,
         num_of_clusters,
         max_feature_num):
    dfile = pd.read_csv(filename, sep='\t')

    # need the list of all genes to base the feature vector on them
    # this array of gene names (strings) will be positional reference for creating feature vectors
    geneID = np.array(dfile["geneID"])
    unique_genes = np.unique(geneID)

    cellID = np.array(dfile['cell'])
    unique_cells = np.unique(cellID)
    # print(unique_cells.size)

    cell_objects = load_data("cells.dat")
    if (cell_objects == []):

        # cell_objects = []
        i = 0
        for cell_num in unique_cells:
            # init cell
            new_cell = cell.Cell(cell_num)
            cell_rows = dfile.loc[dfile["cell"] == cell_num]
            # read all rows for a signle cell and extract data
            new_cell.read_data(cell_rows)
            # create feature vector for cell
            # each position in feature vector is a number of MIDCounts for all unique genes
            # NOTE: additianly we can add other attributes, total expression for cell,
            # maximum expression, number of different genes expressed etc.
            new_cell.create_gene_feature_vector(gene_names = unique_genes)

            # add other features to feature vector
            # here we are adding total gene expression, maximum gene expression and number of unique genes
            new_cell.add_features(new_cell.total_expression, new_cell.maximum_expression, len(new_cell.genes))
            
            # add new_cell to a list of cells
            cell_objects.append(new_cell)

        # save formatted cells
        save_data("cells.dat", cell_objects)
    
    # stack all data in a 2D array
    dataset = np.zeros(shape=(unique_cells.shape[0], cell_objects[0].feature_vector.size)) # can be done without unique variables
    for it in range(len(cell_objects)):
        dataset[it, :] = np.copy(cell_objects[it].feature_vector)

    # normalize additional attributes
    for i in range(1,4):
        dataset[:, -i] /= dataset[:,-i].max()

    # Seurat advises to prune the features by extracting a fixed number of features
    # that have the highest variance in the cell set
    # calculate column variance
    dataset_column_variance = np.var(dataset, axis=0)
    column_variance_pairs = [(dataset[:,i], dataset_column_variance[i]) for i in range(dataset.shape[1])]
    # sort pairs by variance, highest first
    column_variance_pairs.sort(key=lambda k: k[1], reverse=True)
    # stack columns (features) with highest variance into a new dataset
    pruned_dataset = np.hstack((column_variance_pairs[i][0][:, np.newaxis] for i in range(0,max_feature_num)))    

    # run PCA
    pca_dataset = load_data("pca_2Dcomp.dat")
    if (pca_dataset == []):
        # perform PCA on feature vectors
        pca_dataset = pca_fun.pca_transform(pruned_dataset, num_of_pca=2)
        save_data("pca_2Dcomp.dat", pca_dataset)

    norm_pca_dataset = load_data("norm_after_pca.dat")
    if (norm_pca_dataset == []):
        # normalize the area to fit x,y coordinates
        norm_pca_dataset = np.copy(pca_dataset)
        x_values = np.array(dfile["x"])
        y_values = np.array(dfile["y"])
        a = norm_pca_dataset[:,0].min()
        b = norm_pca_dataset[:,0].max()
        c = x_values.min()
        d = x_values.max()
        norm_pca_dataset[:,0] = (1/(b-a))*((d-c)*norm_pca_dataset[:,0] + (b*c - a*d))

        a = norm_pca_dataset[:,1].min()
        b = norm_pca_dataset[:,1].max()
        c = y_values.min()
        d = y_values.max()
        norm_pca_dataset[:,1] = (1/(b-a))*((d-c)*norm_pca_dataset[:,1] + (b*c - a*d))

        save_data("norm_after_pca.dat", norm_pca_dataset)


    # distance metric
    distance_matrix = np.zeros(shape=(len(cell_objects), len(cell_objects)))
    exp_plane_coords = np.copy(norm_pca_dataset)
    spatial_plane_coords = np.array([[cell.spx, cell.spy] for cell in cell_objects])
    # I need norm_pca_dataset with values for 2D exp domain
    # I need array of cell center of masses for 2D spatial domain
    for it in range(len(cell_objects)):
        helper_array = np.ones(shape=(len(cell_objects), ))
        distance_matrix[:, it] = np.sqrt(np.square(spatial_plane_coords[it,0]*helper_array - spatial_plane_coords[:,0]) + np.square(spatial_plane_coords[it,1]*helper_array - spatial_plane_coords[:,1])) * (1-exp_weight) + \
            np.sqrt(np.square(exp_plane_coords[it, 0]*helper_array - exp_plane_coords[:,0]) + np.square(exp_plane_coords[it,1]*helper_array - exp_plane_coords[:,1])) * exp_weight

    max_total_distance = np.ravel(distance_matrix).max()
    threshold = max_total_distance * percentage_distance_threshold

    # perform clusterization

    # # try knn with just xy plane
    # # for wanted conditions in the task k=1

    # # scale the features for better clusterization results
    # # this is done becuase Kmeans uses distance metrics
    # scaler = StandardScaler()
    # spatial_plane_coords = scaler.fit_transform(spatial_plane_coords)

    # xy_kmeans = KMeans(n_clusters=num_of_clusters,
    #                      init="random",
    #                      n_init=10,
    #                      max_iter=1000,
    #                      random_state=9,
    #                      algorithm="lloyd")
    # xy_clusters = xy_kmeans.fit_predict(spatial_plane_coords)

    # # display
    # colors = ['b', 'g', 'r', 'm', 'c', 'k', 'y', '#884488', '#018888', '#2248FF']
    # cluster_labels = np.unique(xy_clusters)
    # for cluster_id in cluster_labels:
    #     plt.scatter(spatial_plane_coords[xy_clusters == cluster_id,0], spatial_plane_coords[xy_clusters == cluster_id,1], marker='o', c=colors[np.where(cluster_labels == cluster_id)[0][0]], label=str(cluster_id))
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    # # try knn with just exp plane
    
    # # scale the features for better clusterization results
    # # this is done becuase Kmeans uses distance metrics
    # scaler = StandardScaler()
    # exp_plane_coords = scaler.fit_transform(exp_plane_coords)
    
    # exp_kmeans = KMeans(n_clusters=num_of_clusters,
    #                      init="random",
    #                      n_init=10,
    #                      max_iter=1000,
    #                      random_state=9,
    #                      algorithm="lloyd")
    # exp_clusters = exp_kmeans.fit_predict(exp_plane_coords)

    # # display

    # cluster_labels = np.unique(exp_clusters)
    # for cluster_id in cluster_labels:
    #     plt.scatter(spatial_plane_coords[exp_clusters == cluster_id, 0], spatial_plane_coords[exp_clusters == cluster_id, 1], marker='o', c=colors[np.where(cluster_labels == cluster_id)[0][0]], label='feature'+ str(cluster_id))
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    # # custom clusterization by the test3 taks
    # # max_iter = 300
    # # curr_pass = 0
    # clusters = [[] for _ in range(num_of_clusters)]
    # curr_cluster = 0
    # available_cells = [x for x in range(len(cell_objects))]
    # while (len(available_cells) > 0 and curr_cluster < num_of_clusters):
    #     # start with random cell from available cells and assign it to a new cluster
    #     new_cell = random.choice(available_cells)
    #     cells_to_process = [new_cell]
    #     while (len(cells_to_process) > 0):
    #         # define current cell anylized
    #         curr_cell = cells_to_process[0]
    #         # extract cells with distance to curr cell smaller than threshold
    #         close_cells = np.array(range(len(cell_objects)))[distance_matrix[curr_cell, :] < threshold]
    #         # add only cells that are not yet in a cluster for further processing
    #         diff_from_processing = list(set(close_cells) - set(clusters[curr_cluster]) - set(cells_to_process))
    #         # for cell in close_cells:
    #         #     if (cell not in clusters[curr_cluster]) and (cell not in cells_to_process):
    #         #         cells_to_process.append(cell)
    #         cells_to_process = cells_to_process + diff_from_processing
    #         # add current cell to current cluster
    #         clusters[curr_cluster].append(curr_cell)

    #         # remove processed cell from all avalable cells and 
    #         # from current cells_to process for this cluster
    #         cells_to_process.remove(curr_cell)
    #         available_cells.remove(curr_cell)


    #     curr_cluster += 1
    
    # curr_cluster -= 1

    # # if we do not have enough clusters label leftover cells as extra class
    # if (len(available_cells) > 0):
    #     clusters.append([x for x in available_cells]) # probably could have just appended available_cells
    #     curr_cluster += 1
    #     # curr_pass += 1
    
    # # display
    # colors = ['b', 'g', 'r', 'm', 'c', 'k', 'y', '#884488', '#018888', '#2248FF', '#FF3388']
    # cluster_labels = np.array(range(curr_cluster+1))
    # for cluster_id in cluster_labels:
    #     plt.scatter(spatial_plane_coords[clusters[cluster_id], 0], spatial_plane_coords[clusters[cluster_id], 1], marker='o', c=colors[np.where(cluster_labels == cluster_id)[0][0]], label=str(cluster_id))
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    available_cells = [x for x in range(len(cell_objects))]
    knmeans_obj = kmeans.KMeans(n_clusters=num_of_clusters, max_iter=300)
    knmeans_obj.fit(available_cells, distance_matrix)
    _, labels = knmeans_obj.evaluate(available_cells, distance_matrix)

    # display
    figure = plt.gcf() # get current figure
    figure.set_size_inches(19.2, 9.83)
    cluster_labels = np.unique(labels)
    for cluster_id in cluster_labels:
        plt.scatter(spatial_plane_coords[np.where(labels==cluster_id)[0], 0], spatial_plane_coords[np.where(labels==cluster_id)[0], 1], marker='o', c=colors[np.where(cluster_labels == cluster_id)[0][0]], label=str(cluster_id))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("custom_kmeans_exp_weight_%.1f_num_cluster_%d_maxfnum_%d_log_genes_3addfeat.png" % (exp_weight, num_of_clusters, max_feature_num))
    # plt.show()
    plt.savefig("custom_kmeans_exp_weight_%.1f_num_cluster_%d_maxfnum_%d_log_genes_3addfeat.png" % (exp_weight, num_of_clusters, max_feature_num), dpi=600)
    plt.close()



    return


if __name__ == '__main__':
    # open the input file
    filename = "../input.tsv"
    pd.options.display.max_rows = 9999
    percentage_distance_threshold = 0.007
    max_feature_num = 2000
    num_of_clusters = 10
    # exp_weight = 0.5
    for exp_weight in [x/10 for x in range(0,11)]:
    #     for num_of_clusters in range(5,15,1):



        main(filename=filename,
            exp_weight = exp_weight, 
            percentage_distance_threshold = percentage_distance_threshold,
            num_of_clusters = num_of_clusters,
            max_feature_num = max_feature_num)
    # print("Number of increases is %d.\n" % result)