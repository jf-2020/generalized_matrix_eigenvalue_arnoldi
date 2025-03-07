# generalized_matrix_eigenvalue_arnoldi.py - apply Arnoldi to the generalized
# eigenvalue problem
#
# jf - Spring '25

import numpy as np
from math import ceil
from pprint import pprint

#######################
### DATA GENERATION ###
#######################

def generate_eigenmatrix(dimension, min, max):
    '''Generate a random matrix over Z^(2n), capping range of values'''

    random_integers = [np.random.randint(min, max) for _ in range(dimension)]
    return np.diag(random_integers)

def generate_random_similarity(dimension, condition_number):
    '''Given a dimension & condition number, generate a random similarity matrix
       by constructing a random SVD with U & V factors coming from a QR
       factorization of a random matrix itself. The condition number is obtained
       via a random SIGMA matrix.

       ASSUMPTION: condition_number > 1
    '''

    U = np.random.rand(dimension, dimension)
    V = np.random.rand(dimension, dimension)
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)

    ### TODO ###
    # utilize exponential decay for drawing random values here.
    # apply inverse sampling to do so.
    ############

    # construct the SIGMA matrix via a uniform distribution
    samples = np.random.uniform(low = 1, high = condition_number,
                                size = dimension)
    # and for simplicity, push values into Z
    samples = [ceil(i) for i in samples]
    sigma = np.diag(samples)

    # finally calculate the similarity matrix & it's inverse
    S = np.matmul(np.matmul(U, sigma), V.T)
    # ... via properties of the underlying SVD
    sigma_inv = np.linalg.inv(sigma)
    S_inv = np.matmul(np.matmul(V, sigma_inv), U.T)

    return S, S_inv

def generate_ABCDS(dimension, condition_number):
    '''generate random (A, B) and C, so that A=BC, for the matrix eigenvalue
    problem. the condition number will be used to control the similarity matrix
    as well as B. note that it'll be helpful to have future access to D and S as
    well, so those are returned too.
    '''

    # first compute the random eigenmatrix (over Z^(2n)), specifying 1 as the
    # smallest value, the condition number being the highest
    D = generate_eigenmatrix(dimension, 1, condition_number)

    # then compute C via a random similarity transform with specified condition
    # number
    S, S_inv = generate_random_similarity(dimension, condition_number)
    C = np.matmul(np.matmul(S, D), S_inv)

    # next compute B just like we did S
    B, _ = generate_random_similarity(dimension, condition_number)

    # compute A = BC
    A = np.matmul(B, C)

    return A, B, C, D, S, S_inv

###############
### ARNOLDI ###
###############

def arnoldi_iteration(A, b, n, convergence=False):
    '''Given a matrix, A, starting vector, b, and the number of iterations, n,
       apply the Arnoldi iteration per Trefethen p252.

       if convergence = True, then measure convergence of ritz pairs every so
       often. we take every so often to mean every other iteration.
    '''

    m = A.shape[0]
    Q = np.zeros((m, n+1)) # zero init Q to store ortho cols via MGS
    H = np.zeros((n+1, n)) # similarly, zero into H to get Hessenberg matrix

    ritz_convergence_measurements = []

    # normalize first col of Q, the init vector (1st Krylov vect)
    Q[:, 0] = b / np.linalg.norm(b)

    # apply MGS per 33.4 p252 Trefethen
    for k in range(n):
        
        # apply A to previous iterate & project
        v = A @ Q[:, k]
        for j in range(k+1):
            H[j, k] = np.dot(Q[:, j], v)
            v -= H[j, k]*Q[:,j]

        # avoid blowup
        if np.linalg.norm(v) < 1e-12:
            break

        # update Hessenberg & Q
        H[k+1, k] = np.linalg.norm(v)
        Q[:, k+1] = v / H[k+1, k]

        # measure convergence if required
        if convergence and (k%2==0 or k==n):

            current_iterate = []

            # first compute the ritz pairs
            evals, evects = np.linalg.eig(H[:n,:n])

            # zip them up
            for i in range(len(evals)):
                ritz_val = evals[i]
                evect = evects[:,i]
                current_iterate.append((ritz_val, evect))

            ritz_convergence_measurements.append(current_iterate)

    if convergence:
        return Q, H[:n, :n], ritz_convergence_measurements
    else:
        return Q, H[:n, :n]

def obtain_Arnoldi_matrix(A, B, sigma):
    '''given the pair, (A,B), and a shift, sigma, obtain an LU factorization of
       A - sigma*B. Invert it & then right-multiply by B, obtaining
       ((A-sigma*B)^(-1))*B.

       we call this matrix the "Arnoldi matrix"

       NOTE: per my understanding, calling linalg.inv from numpy, calls a LAPACK
       routine that itself performs an LU decomposition.
    '''

    # first apply the shift
    M = A - sigma*B
    # then invert it ***AGAIN, an LU factorization is obtained underneath***
    M_inv = np.linalg.inv(M)

    # lastly, post-multiply by B
    return np.matmul(M_inv, B)

def apply_arnoldi(A, B, shift):
    '''apply the arnoldi iteration to the pair (A, B), accumulating convergence
       estimates as it goes.'''
    
    # first apply the shift
    M = obtain_Arnoldi_matrix(A, B, shift)

    # then apply Arnoldi
    size = A.shape[0]
    b = np.eye(1, size, 0)
    Q, H , raw_measurements = arnoldi_iteration(M, b, size, convergence = True)

    # calculate the convergence measurements
    convergence_estimates = []
    for i in range(len(raw_measurements)):
        convergence_estimate = []

        for ritz_pairs in raw_measurements[i]:
            
            # grab the estimates
            ritz_val = ritz_pairs[0]
            ritz_vect = ritz_pairs[1]
            
            # calculate the norm of the numerator
            numerator_left_summand = np.matmul(M, ritz_vect.T)
            numerator_right_summand = ritz_val*ritz_vect
            numerator = numerator_left_summand - numerator_right_summand
            numerator_val = np.linalg.norm(numerator, ord=2)
            
            # and then the denominator
            denominator_right_multiplicand = np.linalg.norm(ritz_vect, ord=2)
            denominator_left_summand = np.linalg.norm(M, ord=2)
            denominator_right_summand = np.linalg.norm(ritz_val)
            denominator_val = (denominator_left_summand + 
                               denominator_right_summand)*denominator_right_multiplicand
            
            # and lastly the required quotient
            quotient = numerator_val/denominator_val

            convergence_estimate.append(quotient)

        convergence_estimates.append(convergence_estimate)

    return M, Q, H, convergence_estimates            
            
###############
### HELPERS ###
###############

def get_spectrum_matrix_pair(A, B):
    '''while spectra (partially) revealed by underlying generators for B & C, this
    breaks out the computation for clarity -- O(n^3) worst case
    '''

    spec_A = np.linalg.eigvals(A)
    spec_B = np.linalg.eigvals(B)

    return spec_A, spec_B

def test_data_generation():
    dimension = 100
    condition_number = 1000
    A, B, C, D, S, S_inv = generate_ABCDS(dimension, condition_number)

    spec_A, spec_B = get_spectrum_matrix_pair(A, B)

    print("Spec(A):", spec_A)
    print()

    print("Spec(B):", spec_B)
    print()

def test_arnoldi_iteration():
    A = np.array([[7,5,3,9],[5,4,6,8],[3,6,2,5],[9,8,5,6]])
    b = np.array([[1,0,0,0]])

    Q, H = arnoldi_iteration(A, b, 3)

    evals = np.linalg.eigvals(H)
    print(evals)

def test_obtain_Arnoldi_matrix():

    A = np.array([[7,5,3,9],[5,4,6,8],[3,6,2,5],[9,8,5,6]])
    B = np.array([[6,0,0,0],[0,3,0,0],[0,0,1,0],[0,0,0,5]])
    sigma = 1

    M = obtain_Arnoldi_matrix(A, B, sigma)
    print(M)

def test_arnoldi_iteration_with_convergence_measurement():
    A = np.array([[7,5,3,9],[5,4,6,8],[3,6,2,5],[9,8,5,6]])
    b = np.array([[1,0,0,0]])

    Q, H, raw_convergence_measurements = arnoldi_iteration(A, b, 3, convergence=True)

    # pprint(raw_convergence_measurements)
    # pprint(raw_convergence_measurements[0])
    ritz_pairs = raw_convergence_measurements[0]
    # pprint(ritz_pair)
    ritz_pair = ritz_pairs[0]
    # pprint(ritz_pair)
    ritz_val, ritz_vect = ritz_pair[0], ritz_pair[1]
    print(ritz_val)
    pprint(ritz_vect)

    # evals = np.linalg.eigvals(H)
    # print(evals)

def test_apply_arnoldi():
    A = np.array([[7,5,3,9],[5,4,6,8],[3,6,2,5],[9,8,5,6]])
    B = np.array([[6,0,0,0],[0,3,0,0],[0,0,1,0],[0,0,0,5]])

    M, Q, H, convergence_estimates = apply_arnoldi(A, B, 1)

    evals, evects = np.linalg.eig(M)
    # pprint(evals)
    # pprint(evects)

    print(convergence_estimates)

############
### MAIN ###
############

def main():
    # test_data_generation()
    # test_arnoldi_iteration()
    # test_obtain_Arnoldi_matrix()
    # test_arnoldi_iteration_with_convergence_measurement()
    test_apply_arnoldi()

if __name__ == "__main__":
    main()