#= GMEP_arnoldi_python2julia.jl - attempt at converting GMEP/Arnoldi Python prgm
                                  to Julia. effectively my "Hello, World!" here!

 jf - March '25
=#

using Distributions
using LinearAlgebra
using Random

#############
### PROCS ###
#############

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

############
### MAIN ###
############

function main()
    
    ### test generate_eigematrix ###
    # M = generate_eigenmatrix(10, 0, 100)
    # display(M)

    ### test generate_random_similarity ###
    # S1, S2 = generate_random_similarity(5, 10)

    ### test generate_ABCDS ###
    # generate_ABCDS(5, 100)

end

main()