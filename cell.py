import numpy as np

class Cell(object):

    def __init__(self, cellID):
        self.id = cellID
        self.genes = None
        self.positions = None
        self.spx = 0
        self.spy = 0
        self.gnx = 0
        self.gny = 0
        self.feature_vector = None
        self.total_expression = 0
        self.maximum_expression = 0
    
    def read_data(self, pdata):
        data_rows = pdata.shape[0]
        if  (data_rows > 0):

            # genes stores the information on gene names and expression level
            self.genes = {}
            # postions store all (x,y) data on the cell, for each gene expression,
            # together with expression level in order to create center of mass.
            # NOTE: should gene expression level influence center of mass?
            self.positions = np.zeros(shape=(data_rows,3))
            for i in range(pdata.shape[0]):
                row = pdata.loc[pdata.index[i]]
                self.genes[row["geneID"]] = row["MIDCounts"]
                self.positions[i,0] = row["x"]
                self.positions[i,1] = row["y"]
                self.positions[i,2] = row["MIDCounts"]

            # sum of all gene MIDCounts for the cell   
            self.total_expression = self.positions[:,2].sum()

            # highest expression of a single gene
            self.maximum_expression = self.positions[:,2].max()

            # calculate center of mass, default is weighted
            self.calc_center_of_mass(weighted=True)
       
            return 1
        else:
            print("Bad data for cell %s.\n" % self.id)
            return 0

    ## CREATE_GENE_FEATURE_VECTOR
    # @brief create feature vector with all genes expresions for cell
    # each position in feature vector is a number of MIDCounts for all unique genes
    # Seurat advises that the gene expressions per gene should be normalized
    # with total cell gene expression. Then multiplied with a scale factor
    # (default 10 000) and log transformed with log(1+x) function.
    def create_gene_feature_vector(self, gene_names, scale_factor = 10000):
        self.feature_vector = np.zeros(shape=gene_names.shape)
        for name in self.genes:
            self.feature_vector[np.where(gene_names==name)] = self.genes[name]
        
        # as done in Seurat, normalize the data with total gene expression for the cell
        # then multiply by a scale factor a perform log transformation
        self.feature_vector /= self.total_expression
        self.feature_vector *= scale_factor
        # log is done with numpy log1p(x) = natural logarithm log(1+x)
        self.feature_vector = np.log1p(self.feature_vector)
        return

    ## ADD FEATURES
    # @brief add custom features to gene feature vector
    # add features like total gene expression, number of unique genes
    def add_features(self, *args):
        for feature in args:
            self.feature_vector = np.append(self.feature_vector, feature)
        return

    ## CALC_CENTER_OF_MASS
    # @brief calculate the reference position of the cell
    # some cells have different positions for differenc gene expressions
    # this offsets are small. There are two ways to calculate center of mass:
    # taking into account the MIDCount for gene at the position, or just using
    # each gene expressiong position with same unity weight
    def calc_center_of_mass(self, weighted=True):
        if (weighted):
            # befor summing
            # multiply the position with MIDCount weight
            # and then norm the result with total weight (MIDCounts)
            self.spx = (self.positions[:,0] * self.positions[:,2]).sum() / self.positions[:,2].sum()
            self.spy = (self.positions[:,1] * self.positions[:,2]).sum() / self.positions[:,2].sum()
        else:
            # sum all x (y) positions and divide with number of positions summed.
            self.spx = self.positions[:,0].sum() / self.positions.shape[0]
            self.spy = self.positions[:,1].sum() / self.positions.shape[0]

        return
 

