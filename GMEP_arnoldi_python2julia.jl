#= GMEP_arnoldi_python2julia.jl - attempt at converting GMEP/Arnoldi Python prgm
                                  to Julia. effectively my "Hello, World!" here!

 jf - March '25
=#

using Distributions
using LinearAlgebra
using Random

#######################
### DATA GENERATION ###
#######################

function generate_eigenmatrix(dim, min, max)
    # Generate a random matrix over Z^(2n), capping range of values

    random_integers = rand(min:max, dim)
    return diagm(random_integers)
end

function generate_random_similarity(dimension, condition_number)
    #= Given a dimension & condition number, generate a random similarity matrix
       by constructing a random SVD with U & V factors coming from a QR
       factorization of a random matrix itself. The condition number is obtained
       via a random SIGMA matrix.

       ASSUMPTION: condition_number > 1
    =#

    U = rand(dimension, dimension)
    V = rand(dimension, dimension)
    U, _ = qr(U)
    V, _ = qr(V)

    ### TODO ###
    # utilize exponential decay for drawing random values here. apply inverse
    # sampling to do so.

    # construct the SIGMA matrix via a uniform distribution
    samples = rand(Uniform(1, condition_number), dimension)
    # and for simplicity, push values into Z
    samples = [ceil(i) for i in samples]
    sigma = diagm(samples)

    # finally calculate the similarity matrix & it's inverse
    S = U*sigma*transpose(V)
    # ... via properties of the underlying SVD
    sigma_inv = inv(sigma)
    S_inv = V*sigma_inv*transpose(U)

    return S, S_inv
end

function generate_ABCDS(dimension, condition_number)
    #= generate random (A, B) and C, so that A=BC, for the matrix eigenvalue
       problem. the condition number will be used to control the similarity matrix
       as well as B. note that it'll be helpful to have future access to D and S as
       well, so those are returned too.
    =#

    # first compute the random eigenmatrix (over Z^(2n)), specifying 1 as the
    # smallest value, the condition number being the highest
    D = generate_eigenmatrix(dimension, 1, condition_number)

    # then compute C via a random similarity transform with specified condition
    # number
    S, S_inv = generate_random_similarity(dimension, condition_number)
    C = S*D*S_inv
    
    # next compute B just like we did S
    B, _ = generate_random_similarity(dimension, condition_number)

    # compute A = BC
    A = B*C

    return A, B, C, D, S, S_inv
end

###############
### ARNOLDI ###
###############

function arnoldi_iteration(A, b, n, convergence=false)
    #= Given a matrix, A, starting vector, b, and the number of iterations, n,
       apply the Arnoldi iteration per Trefethen p252.

       if convergence = True, then measure convergence of ritz pairs every so
       often. we take every so often to mean every other iteration.
    =#
    
    m = size(A, 1)
    Q = zeros(m, n+1) # zero init Q to store ortho cols via MGS
    H = zeros(n+1, n) # similarly, zero into H to get Hessenberg matrix

    ritz_convergence_measurements = []

    # normalize first col of Q, the init vector (1st Krylov vect)
    col1_norm_factor = 1/norm(Q[:, 1])
    new_col = col1_norm_factor*Q[:,1]
    Q[:, 1] = new_col
end

############################
########## IGNORE ##########
############################

# def arnoldi_iteration(A, b, n, convergence=False):

#     # normalize first col of Q, the init vector (1st Krylov vect)
#     Q[:, 0] = b / np.linalg.norm(b)

#     # apply MGS per 33.4 p252 Trefethen
#     for k in range(n):
        
#         # apply A to previous iterate & project
#         v = A @ Q[:, k]
#         for j in range(k+1):
#             H[j, k] = np.dot(Q[:, j], v)
#             v -= H[j, k]*Q[:,j]

#         # avoid blowup
#         if np.linalg.norm(v) < 1e-12:
#             break

#         # update Hessenberg & Q
#         H[k+1, k] = np.linalg.norm(v)
#         Q[:, k+1] = v / H[k+1, k]

#         # measure convergence if required
#         if convergence and (k%2==0 or k==n):

#             current_iterate = []

#             # first compute the ritz pairs
#             evals, evects = np.linalg.eig(H[:n,:n])

#             # zip them up
#             for i in range(len(evals)):
#                 ritz_val = evals[i]
#                 evect = evects[:,i]
#                 current_iterate.append((ritz_val, evect))

#             ritz_convergence_measurements.append(current_iterate)

#     if convergence:
#         return Q, H[:n, :n], ritz_convergence_measurements
#     else:
#         return Q, H[:n, :n]

################################
########## END IGNORE ##########
################################

############
### MAIN ###
############

function main()
    
    ### test generate_eigenmatrix ###
    # M = generate_eigenmatrix(10, 0, 100)
    # display(M)

    ### test generate_random_similarity ###
    # S1, S2 = generate_random_similarity(5, 10)

    ### test generate_ABCDS ###
    # generate_ABCDS(5, 100)

    ### test arnoldi_iteration ###
    A = [7 5 3 9; 5 4 6 8; 3 6 2 5; 9 8 5 6]
    b = [1 0 0 0]
    arnoldi_iteration(A, b, 3)

    # def test_arnoldi_iteration():
    # A = np.array([[7,5,3,9],[5,4,6,8],[3,6,2,5],[9,8,5,6]])
    # b = np.array([[1,0,0,0]])

    # Q, H = arnoldi_iteration(A, b, 3)

    # evals = np.linalg.eigvals(H)
    # print(evals)

end

main()