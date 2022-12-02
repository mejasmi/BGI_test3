import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cell


def main(filename):

    dfile = pd.read_csv(filename, sep='\t')
    # print(dfile.head(10))
    # print(dfile.tail(10))
    # print(dfile.info())
    # print(dfile.to_string())

    # p = dfile.duplicated()
    # for value in p:
    #     if value:
    #         print("duplicate found")
    # dfile.plot(kind='scatter', x='x', y='y')
    # plt.show()

    # dfile.plot(kind='scatter', x='geneID', y='cell')
    # plt.show()

    geneID = np.array(dfile['geneID'])
    unique_genes, un_genes_cnt = np.unique(geneID, return_counts=True)
    # print(unique_genes.size)
    # print(un_genes_cnt.max())

    cellID = np.array(dfile['cell'])
    unique_cells, gene_per_cell_cnt = np.unique(cellID, return_counts=True)
    print(unique_cells.size)
    # plt.hist(gene_per_cell_cnt)
    # plt.show()

    # total gene expression per cell
    # I want to create an array the size of unique cell number
    # go through database and for each row with this cell ID accumulate MIDCounts
    # this will provide total gene expression per cell
    # plot hist of it
    total_expression_per_cell = np.zeros(shape=(1,unique_cells.size), dtype=np.int64)
    cell_objects = []
    for it, cell_num in zip(range(unique_cells.size), unique_cells):
        # init cell
        new_cell = cell.Cell(cell_num)
        cell_rows = dfile.loc[dfile["cell"] == cell_num]
        new_cell.read_data(cell_rows)        

        total_expression_per_cell[0,it] = cell_rows["MIDCounts"].sum()

    plt.hist(total_expression_per_cell)
    plt.show()

    # create feature vector the size of all unique genes, and additional value total_gene_expression
    # fill those feature vectors and run PCA


    return


if __name__ == '__main__':
    # open the input file
    filename = "../input.tsv"
    pd.options.display.max_rows = 9999
    main(filename=filename)
    # print("Number of increases is %d.\n" % result)