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
        
            return 1
        else:
            print("Bad data for cell %s.\n" % self.id)
            return 0

    ## CREATE_FEATURE_VECTOR
    # @brief
    def create_feature_vector(self, gene_names):
        self.feature_vector = np.zeros(shape=gene_names.shape)
        for name in self.genes:
            self.feature_vector[np.where(gene_names==name)] = self.genes[name]
        return
    

    ## CALC_CENTER_OF_MASS
    # @brief calculate the reference position of the cell
    # 
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
