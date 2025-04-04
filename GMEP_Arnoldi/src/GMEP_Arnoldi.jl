module GMEP_Arnoldi

#= GMEP_arnoldi_python2julia.jl - attempt at converting GMEP/Arnoldi Python prgm
                                  to Julia. effectively my "Hello, World!" here!

 jf - March '25
=#

using Distributions
using LinearAlgebra: checksquare
using Random

export generate_eigenmatrix,
    generate_random_similarity, generate_ABCDS, arnoldi_iteration

#######################
### DATA GENERATION ###
#######################

function generate_eigenmatrix(dim, min, max)
    # Generate a random matrix over Z^(2n), capping range of values
    random_integers = rand(min:max, dim)
    # return diagm(random_integers)
    return Diagonal(random_integers)
end

# I don't love the use of dimension here when there is a very strong
# mathematical convention to use m for the rows in a matrix.
# I'd suggest "condition_bound" since this generates a matrix
# with condition less than this parameter.
#
# The extra type parameter allows you to generate complex matrices.
#
# function generate_random_similarity(dimension, condition_number)
function generate_random_similarity(dimension, condition_number, ::Type{E}=Float64) where {E}
    #= Given a dimension & condition number, generate a random similarity matrix
       by constructing a random SVD with U & V factors coming from a QR
       factorization of a random matrix itself. The condition number is obtained
       via a random SIGMA matrix.

       ASSUMPTION: condition_number > 1
    =#

    # use randn for generating Q.
    # U = rand(dimension, dimension)
    # V = rand(dimension, dimension)
    U = randn(E, dimension, dimension)
    V = randn(E, dimension, dimension)
    U, _ = qr(U)
    V, _ = qr(V)

    ### TODO ###
    # utilize exponential decay for drawing random values here. apply inverse
    # sampling to do so.

    # construct the SIGMA matrix via a uniform distribution
    samples = rand(Uniform(1, condition_number), dimension)

    # and for simplicity, push values into Z
    samples = [ceil(i) for i in samples]

    # diagm allocates a matrix and will be slower to multiply with.
    # sigma = diagm(samples)
    sigma = Diagonal(samples)

    # finally calculate the similarity matrix & it's inverse
    #
    # Don't use transpose unless you know for sure that the matrix is
    # real.   ' is more general, more idiomatic, and more concise even for real matrices.
    # S = U * sigma * transpose(V)
    S = U * sigma * V'
    # ... via properties of the underlying SVD

    # Avoid inv and the explicit computation of inverses.  It is
    # unstable in general, although not necessarily a problem for a
    # Diagonal.  But I would question returning S_inv at all,
    # since using it in matrix multiplication will be unstable, even
    # if the inversion of sigma_inv is OK.

    # sigma_inv = inv(sigma)
    # S_inv = V * sigma_inv * transpose(U)

    # return S, S_inv
    return S
end

function generate_ABCDS(dimension, condition_number, ::Type{E}=Float64) where {E}
    #= generate random (A, B) and C, so that A=BC, for the matrix eigenvalue
       problem. the condition number will be used to control the similarity matrix
       as well as B. note that it'll be helpful to have future access to D and S as
       well, so those are returned too.
    =#

    # first compute the random eigenmatrix (over Z^(2n)), specifying 1 as the
    # smallest value, the condition number being the highest
    #
    # We probably do want complex eigenvalues.  I think we discussed 2×2 blocks in D?
    #
    D = generate_eigenmatrix(dimension, 1, condition_number)

    # then compute C via a random similarity transform with specified condition
    # number
    # S, S_inv = generate_random_similarity(dimension, condition_number)
    # C = S * D * S_inv
    S = generate_random_similarity(dimension, condition_number, E)
    C = (S * D) / S

    # next compute B just like we did S
    B, _ = generate_random_similarity(dimension, condition_number, E)

    # compute A = BC
    A = B * C

    # return A, B, C, D, S, S_inv
    return A, B, C, D, S
end

###############
### ARNOLDI ###
###############

# The types give you access to the type variable E in the body of the function.
function arnoldi_iteration(
    A::AbstractMatrix{E},
    b::AbstractVector{E},
    n::Int,
    convergence = false,
) where {E<:Number}
    #= Given a matrix, A, starting vector, b, and the number of iterations, n,
       apply the Arnoldi iteration per Trefethen p252.

       if convergence = True, then measure convergence of ritz pairs every so
       often. we take every so often to mean every other iteration.
    =#

    # checksquare will throw an exception if A is not square.
    m = checksquare(A)

    # By default, without a type parameter, zeros produces a double precision
    # matrix.  If you pass in a complex matrix, you will have mixed types
    # which might not be a problem, but probably isn't what you want
    # and can mess with type inference and slow down the code.
    # Q = zeros(m, n + 1) # zero init Q to store ortho cols via MGS
    Q = zeros(E, m, n + 1) # zero init Q to store ortho cols via MGS

    # dots broadcast and avoid allocating a new vector for b/norm(b).
    # This will turn into a loop that divides each element of b by the
    # norm and stores the result in Q[:,1].  The @. macro introduces
    # broadcasting on every operation.  Unfortunately it broadcasts
    # norm over b, so it means that it computes the norm of each
    # element of b.  So I have put dots in the right place manually.
    Q[:, 1] .= b ./ norm(b)

    H = zeros(n + 1, n) # similarly, zero into H to get Hessenberg matrix

    ritz_convergence_measurements = []

    # normalize first col of Q, the init vector (1st Krylov vect)
    col1_norm_factor = 1 / norm(Q[:, 1])
    new_col = col1_norm_factor * Q[:, 1]
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

end # module GMEP_Arnoldi
