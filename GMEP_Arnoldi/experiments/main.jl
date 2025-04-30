############
### MAIN ###
############

include("../src/GMEP_Arnoldi.jl")
using .GMEP_Arnoldi
using LinearAlgebra

function main()
    
    ### test generate_eigenmatrix ###
    # M = generate_eigenmatrix(10, 0, 100)
    # display(M)

    ### test generate_random_similarity ###
    # S1, S2 = generate_random_similarity(5, 10)

    ### test generate_ABCDS ###
    # generate_ABCDS(5, 100)

    #=
    arnoldi iteration passes on below for eigvals -- tested against python's
    scipy eig function.

    HOWEVER, while the spectra agree, the eigvects do not..
    =#

    ### test arnoldi_iteration ###
    A = [7.0 5 3 9;
        5 4 6 8;
        3 6 2 5;
        9 8 5 6]
    
    # B = [1.0 2 1 3;
    #     2 1 2 7;
    #     1 2 1 4;
    #     3 7 4 1]

    # B = I(4) # for some reason this doesn't register as a correct subtype of the
    #            AbstractMatrix type
    
    B = [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    b = [1.0, 0, 0, 0]
    sigma = 0.0
    iters = 4
    Q, H = arnoldi_iteration(A, B, b, sigma, iters)
    display(Q)
    display(H)

    evals_H, evects_H = eigvals(H[begin:4, begin:4]), eigvecs(H[begin:4, begin:4])
    evals_A, evects_A = eigvals(A), eigvecs(A)
    display(evals_A)
    display(evals_H)

end

main()