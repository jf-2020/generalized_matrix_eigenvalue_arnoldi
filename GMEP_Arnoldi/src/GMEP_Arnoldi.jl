module GMEP_Arnoldi

#= GMEP_arnoldi_python2julia.jl
    
        Convert GMEP/Arnoldi from Python to Julia.
        This is basically a Julia "Hello, World!".

 jf - March '25
=#

using Distributions
using LinearAlgebra
using LinearAlgebra: checksquare
using Random

export generate_eigenmatrix,
       generate_random_similarity,
       generate_ABCDS,
       arnoldi_iteration

"Generate a random matrix over Z^(2n), capping the range of values."
function generate_eigenmatrix(m, min, max) 
    random_integers = rand(min:max, m)
    return Diagonal(random_integers)
end

"""
Generate a random similarity matrix by constructing a random SVD with U & V
factors coming from a QR factorization of a random matrix itself. The condition
number is obtained via a random SIGMA matrix.

ASSUMPTION: condition_number > 1
"""
function generate_random_similarity(m,
                                    condition_number,
                                    ::Type{E}=Float64
                                    ) where {E}
    # The extra type parameter allows you to generate complex matrices.
    
    U = randn(E, m, m)
    V = randn(E, m, m)
    U, _ = qr(U)
    V, _ = qr(V)

    #= TODO
    utilize exponential decay for drawing random values here. apply inverse
    sampling to do so.
    =#

    # construct the SIGMA matrix via a uniform distribution
    samples = rand(Uniform(1, condition_number), m)

    # and for simplicity, push values into Z
    samples = [ceil(i) for i in samples]
    
    sigma = Diagonal(samples)

    # finally calculate the similarity matrix & it's inverse
    S = U * sigma * V'

    return S, U, sigma, V
end

"""
Generate a random (A, B) and C, so that A=BC. The condition number will be used
to control the similarity matrix as well as B. Note that it'll be helpful to
have future access to D and S too, so those are returned as well.
"""
function generate_ABCDS(m,
                        condition_number,
                        ::Type{E}=Float64
                        ) where {E}
    #= TODO
    We probably do want complex eigenvalues.  I think we discussed 2×2 blocks in
    D?
    =#

    # first compute the random eigenmatrix (over Z^(2n)), specifying 1 as the
    # smallest value, the condition number being the highest    
    D = generate_eigenmatrix(m, 1, condition_number)
    S, _, __, ___ = generate_random_similarity(m, condition_number, E)
    C = (S * D) / S

    # then compute B just like we did S
    B, _, __, ___ = generate_random_similarity(m, condition_number, E)
    # which yields A
    A = B * C

    return A, B, C, D, S
end

"""
Given a matrix pair, (A, B), a starting vector, b, and the number of iterations,
n, and the shift parameter, sigma, apply the Arnoldi iteration per Trefethen
p252.

If convergence == True, then measure convergence of ritz pairs every so often.
We take every so often to mean every other iteration.
"""
function arnoldi_iteration(
                            A::AbstractMatrix{E},
                            B::AbstractMatrix{E},
                            b::AbstractVector{E},
                            sigma::Real,
                            n::Int
                            #convergence = false
                            ) where {R<:Real, E<:Union{Complex{R}, R}}
    # The types give you access to the type variable E in the body of the function.

    m = checksquare(A, B)[1] # assumption on regularity: get A's dimension only
    Q = zeros(E, m, n + 1) # zero init Q to store ortho cols via MGS
    H = zeros(E, n + 1, n) # similarly, zero into H to get Hessenberg matrix
    # ritz_convergence_measurements = Vector{Tuple{Complex{R},Vector{Complex{R}}}}[]

    # normalize first col of Q, the init vector (1st Krylov vect)
    Q[:, 1] .= b ./ norm(b) # will prepending -> @. <- not perform the broadcast here?

    # form the spectral shift
    M = A - sigma*B
    F = lu(M) # get LU factorization of spec shift

    # apply MGS per 33.4 p252 Trefethen
    for k in 1:m
        
        # apply M to previous iterate & project
        
        #= TODO
        are we sure F, of "LU type," behaves as expected?
        
        e.g., why not U \ (L \ (B * Q_k)) ??
        
        and in that case, do we need to unravel any permuted rows?
        =#
        v = F \ (B * Q[:, k])
        
        for j in 1:k
            # B * Q_n = (A - sigma*B) * Q_n+1 * H_n+1
            H[j, k] = dot(Q[:, j], v)
            v -= H[j, k] * Q[:, j]
        end

        # update Hessenberg & QR
        H[k+1, k] = norm(v)

        # will prepending -> @. <- not perform the broadcast here?
        Q[:, k+1] .= v ./ H[k+1, k]

        # # measure convergence (if required)
        # if convergence && ((k % 2 == 0) || (k == n))

        #     current_iterate = Tuple{Complex{R},Vector{Complex{R}}}[]

        #     # compute the Ritz pairs
        #     eigen_res = eigen(H[begin:end-1, begin:end])
        #     evals, evects = eigen_res.values, eigen_res.vectors

        #     # zip them up
        #     for i in 1:length(evals)
        #         ritz_val = evals[i]
        #         evect = evects[:, i]
        #         push!(current_iterate, (ritz_val, evect))
        #     end

        #     push!(ritz_convergence_measurements, current_iterate)
        # end
    end

    # return Q, H[begin:end-1, begin:end]
    return Q, H
    # return Q, H[begin:end-1, begin:end], ritz_convergence_measurements
end

end # module GMEP_Arnoldi