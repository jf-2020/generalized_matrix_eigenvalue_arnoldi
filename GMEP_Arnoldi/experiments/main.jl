############
### MAIN ###
############

using GMEP_Arnoldi
using LinearAlgebra

function main()
    
    ### test generate_eigenmatrix ###
    # M = generate_eigenmatrix(10, 0, 100)
    # display(M)

    ### test generate_random_similarity ###
    # S1, S2 = generate_random_similarity(5, 10)

    ### test generate_ABCDS ###
    # generate_ABCDS(5, 100)

    ### test arnoldi_iteration ###
    A = Float64[7 5 3 9;
                5 4 6 8;
                3 6 2 5;
                9 8 5 6]
    b = Float64[1, 0, 0, 0]
    arnoldi_iteration(A, b, 3)

    # def test_arnoldi_iteration():
    # A = np.array([[7,5,3,9],[5,4,6,8],[3,6,2,5],[9,8,5,6]])
    # b = np.array([[1,0,0,0]])

    # Q, H = arnoldi_iteration(A, b, 3)

    # evals = np.linalg.eigvals(H)
    # print(evals)

end

main()
