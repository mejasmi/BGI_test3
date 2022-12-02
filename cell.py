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
                row = pdata.loc[i]
                self.genes[row["geneID"]] = row["MIDCounts"]
                self.positions[i,0] = row["x"]
                self.positions[i,1] = row["y"]
                self.positions[i,2] = row["MIDCounts"]
        
            return 1
        else:
            print("Bad data for cell %s.\n" % self.id)
            return 0

    def create_feature_vector(self):
        pass
    

    def calc_center_of_mass(self):
        pass