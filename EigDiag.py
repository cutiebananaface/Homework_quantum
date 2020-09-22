import numpy as np
import scipy.linalg as la
def EigDiag(bigm):
    '''this function will extract the eigenvalues, sort them in ascending order, and place them in diagonal matrix with the eigenvalues
    ascending down the diagonal- Does not sort the eigenvectors'''
    bigm_eig, evecs = la.eig(bigm)  # extract the eigenvalues
    bigm_eig = bigm_eig.real #they are real values

    idx= np.argsort(bigm_eig)
    sorted_eig= bigm_eig[idx]
    sorted_vecs= evecs[:,idx]
    # sorted_eig = np.sort(bigm_eig) #sort them in ascending order
    diag_eig = np.diag(sorted_eig) #put the sorted values in a diagonal array

    return diag_eig, sorted_eig, sorted_vecs
