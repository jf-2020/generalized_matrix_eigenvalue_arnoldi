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
    A = Float64[7 5 3 9;
                5 4 6 8;
                3 6 2 5;
                9 8 5 6]
    b = Float64[1, 0, 0, 0]
    Q, H, conv = arnoldi_iteration(A, b, 4)
    display(Q)
    display(H)
    println(conv)

end

main()