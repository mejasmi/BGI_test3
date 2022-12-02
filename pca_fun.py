# PCA - Principal Component Analysis
# code acquired from https://vitalflux.com/feature-extraction-pca-python-example/
#                and https://vitalflux.com/pca-explained-variance-concept-python-example/
import numpy as np
from sklearn.preprocessing import StandardScaler

def pca_transform (dataset, num_of_pca=0, explained_variance=0.75):

    #
    # Standardize the dataset; This is very important before you apply PCA
    #
    sc = StandardScaler()
    sc.fit(dataset)
    dataset_std = sc.transform(dataset)

    #
    # Import eigh method for calculating eigenvalues and eigenvectirs
    #
    from numpy.linalg import eigh
    #
    # Determine covariance matrix
    #
    cov_matrix = np.cov(dataset_std, rowvar=False)
    #
    # Determine eigenvalues and eigenvectors
    #
    egnvalues, egnvectors = eigh(cov_matrix)
    #
    # Determine explained variance and select the most important eigenvectors based on explained variance
    #
    total_egnvalues = sum(egnvalues)
    var_exp = [(i/total_egnvalues) for i in sorted(egnvalues, reverse=True)]
    
    #
    # Plot the explained variance against cumulative explained variance
    #
    import matplotlib.pyplot as plt
    cum_sum_exp = np.cumsum(var_exp)
    print("With 2 components, explained variance is: %f" % cum_sum_exp[1])
    plt.bar(range(0,len(var_exp)), var_exp, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_exp)), cum_sum_exp, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    #
    # Construct projection matrix using the num_of_pca eigenvectors that correspond to the top num_of_pca eigenvalues (largest)
    # or the number of pca that cover explained_variance percentage to capture at least the explained_variance of the variance in this dataset
    #
    
    egnpairs = [(np.abs(egnvalues[i]), egnvectors[:, i])
                    for i in range(len(egnvalues))]
    egnpairs.sort(key=lambda k: k[0], reverse=True)
    if (num_of_pca != 0):
        projectionMatrix = np.hstack((egnpairs[i][1][:, np.newaxis] for i in range(0,num_of_pca)))
    else:
        num_pca_for_var_exp = np.where(cum_sum_exp>=explained_variance)[0][0]
        print("For explained variance %f %d components are needed" % (explained_variance, num_pca_for_var_exp))
        projectionMatrix = np.hstack((egnpairs[i][1][:, np.newaxis] for i in range(0,num_pca_for_var_exp)))

    #
    # Transform the training data set
    #
    dataset_pca = dataset_std.dot(projectionMatrix)



    return dataset_pca