import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cell
import pca_fun
import pickle


def load_data():
    try:
        with open("cells.dat") as f:
            objects = pickle.load(f)
    except:
        objects = []
    return objects

def save_data(data):
    with open("cells.dat", "wb") as f:
        pickle.dump(data, f)



def main(filename):
    dfile = pd.read_csv(filename, sep='\t')

    # need the list of all genes to base the feature vector on them
    # this array of gene names (strings) will be positional reference for creating feature vectors
    geneID = np.array(dfile["geneID"])
    unique_genes = np.unique(geneID)

    cellID = np.array(dfile['cell'])
    unique_cells = np.unique(cellID)
    # print(unique_cells.size)

    cell_objects = load_data()
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
            new_cell.create_feature_vector(gene_names = unique_genes)
            # calculate center of mass for the cell
            # some cells have different positions for differenc gene expressions
            # this offsets are small. There are two ways to calculate center of mass:
            # taking into account the MIDCount for gene at the position, or just using
            # each gene expressiong position with same unity weight
            new_cell.calc_center_of_mass(weighted = True)
            # add new_cell to a list of cells
            cell_objects.append(new_cell)
            # if (len(cell_objects) > 49 ):
            #     break

        # save formatted cells
        save_data(cell_objects)
    

    dataset = np.zeros(shape=(unique_cells.shape[0], unique_genes.shape[0])) # can be done without unique variables
    for it in range(len(cell_objects)):
        dataset[it, :] = np.copy(cell_objects[it].feature_vector)
    
    # create feature vector the size of all unique genes, and additional value total_expression
    # fill those feature vectors and run PCA
    

    # perform PCA on feature vectors
    pca_dataset = pca_fun.pca_transform(dataset, explained_variance = 0.95)

    # perform clusterization




    return


if __name__ == '__main__':
    # open the input file
    filename = "../input.tsv"
    pd.options.display.max_rows = 9999
    main(filename=filename)
    # print("Number of increases is %d.\n" % result)