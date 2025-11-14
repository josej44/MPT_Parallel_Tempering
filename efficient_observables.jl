# Tensor_trains 
using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress, TruncBondThresh  



# It works for both, simple and parallel systems

"""
    The function calculate all the matrix products 1:k and k:N of the sum matrices in the 
    tensor_train of distribution.

    It uses as argument the tensor B and return a pair (left, righ) each one with the N-1 products.
    each product is a horizontal vector in left and vertical vector in right.
"""
function full_subproducts_system(B)
    left = []
    right = []
    N = length(B.tensors)
    @tullio init_left[a,b] := B.tensors[1][a,b,l]
    @tullio init_right[a,b] := B.tensors[N][a,b,l]
    push!(left, init_left)
    push!(right, init_right)
    for k in 2:N-1
        @tullio next_left[a,b] := B.tensors[k][a,b,l]
        init_left = init_left * next_left
        push!(left, init_left)
        @tullio next_right[a,b] := B.tensors[N+1-k][a,b,l]
        init_right =  next_right * init_right
        push!(right, init_right)
    end
    right = reverse(right)
    return left, right
end




# It works for both, simple and parallel systems

"""
    The function calculate the marginals resulting of each matrix in the tensor train, 
        (i.e. marginals of the simple spins in the simple Gaubler and the pairs of spins in
        the two chains model)
    
    It uses B or the system of subproducts previously calculated. It return a vector with each (pair)
    marginal
"""

function full_marginals_system(B, full_subprod_system = 1, full_system_already_compute = false)

    if full_system_already_compute == false
        left, right = full_subproducts_system(B)
    else
        left, right = full_subprod_system
    end

    N = length(B.tensors)
    marginals = []
    push!(marginals, [( B.tensors[1][:,:, q] * right[1])[1] for q in 1:size(B.tensors[1],3)])
    for k in 2:N-1
        push!(marginals, [( left[k-1] * B.tensors[k][:,:, q] * right[k])[1] for q in 1:size(B.tensors[1],3)])
    end
    push!(marginals, [( left[N-1] * B.tensors[N][:,:, q])[1] for q in 1:size(B.tensors[1],3)])
    return marginals
end




# Just for parallel systems

"""
    Calculate the expected value (marginal magnetization) of each spin. 
    Return a vector with N pairs, each pair j is the expected value of the spin j of both chains.
"""

function marginal_ev_system(marginal_system)
    function expected(dist)
        return ((dist[1] + dist[3]) * (-1) + (dist[2] + dist[4]) * (1), (dist[1] + dist[2]) * (-1) + (dist[3] + dist[4]) * (1))
    end

    expected_values = []
    for k in 1:N
        push!(expected_values, expected(marginal_system[k]))
    end
    return expected_values
end




# It works for both, simple and parallel systems

"""
    The function calculate the marginals resulting of each pair of consecutives matrices in the tensor 
        train,
    
    It uses B or the system of subproducts previously calculated. It return a vector with each (pair)
    marginal
"""

function full_second_order_marginals_system(B, full_subprod_system = 1, full_system_already_compute = false)

    if full_system_already_compute == false
        left, right = full_subproducts_system(B)
    else
        left, right = full_subprod_system
    end

    N = length(B.tensors)
    marginals = []
    
    # Primera marginal (k=1)
    right_2 = right[2]
    @tullio corr_k[a,b] := B.tensors[1][i,j,a] * B.tensors[2][j,k,b] * right_2[k,m]
    push!(marginals, corr_k)

    # Marginales intermedias
    for k in 2:N-2
        left_k = left[k-1]
        right_k = right[k+1]
        @tullio corr_k_temp[a,b] := left_k[i,j] * B.tensors[k][j,m,a] * B.tensors[k+1][m,n,b] * right_k[n,o]
        push!(marginals, corr_k_temp)
    end
    
    # Última marginal
    left_end = left[N-2]
    @tullio corr_k_last[a,b] := left_end[i,j] * B.tensors[N-1][j,k,a] * B.tensors[N][k,m,b]
    push!(marginals, corr_k_last)
    
    return marginals
end




# Just for parallel systems

"""
    Calculate the correlated moment of each pair of consecutives spins. 
    Return a vector with N-1 pairs, each pair j is the correlated moment of the pair (j,j+1) 
    in both chains.
"""

function second_moment_system(second_order_marginals_system)
    function sum_dif(matrix, vec, vec2)
        state = matrix[:,vec[1]] + matrix[:,vec[2]] - matrix[:,vec[3]] - matrix[:,vec[4]]
        state2 = state[vec2[1]] + state[vec2[2]] - state[vec2[3]] - state[vec2[4]]
        return state2[1,1]
    end

    second_moments = []
    for k in 1:N-1
        marginals = second_order_marginals_system[k]
        second_moment_ch_1 = sum_dif(marginals, [2,4,1,3], [2,4,1,3])
        second_moment_ch_2 = sum_dif(marginals, [3,4,1,2], [3,4,1,2])
        push!(second_moments, (second_moment_ch_1, second_moment_ch_2))
    end
    return second_moments
end




# Just for parallel systems

"""
    Calculate the correlation of each pair of consecutives spins. 
    Return a vector with N-1 pairs, each pair j is the correlation of the pair (j,j+1) 
    in both chains.
"""

function correlation_between_spins_system(second_moments_system, expected_values_system)
    correlations = []
    for k in 1:N-1
        second_moment = second_moments_system[k]
        expected_value_1 = expected_values_system[k][1]
        expected_value_2 = expected_values_system[k+1][1]
        corr_ch_1 = (second_moment[1] - expected_value_1 * expected_value_2) / (sqrt(1 - expected_value_1^2) * sqrt(1 - expected_value_2^2))
        expected_value_1 = expected_values_system[k][2]
        expected_value_2 = expected_values_system[k+1][2]
        corr_ch_2 = (second_moment[2] - expected_value_1 * expected_value_2) / (sqrt(1 - expected_value_1^2) * sqrt(1 - expected_value_2^2))
        push!(correlations, (corr_ch_1, corr_ch_2))
    end
    return correlations
end




# Just for parallel systems

"""
    energy_function(full_expected_values, full_second_moments, params)
Calculate the expected energy of both chains from the marginal expected values and
second order, along with the system parameters.
"""

function energy_function(full_expected_values, full_second_moments, params)
    energy_1 = 0.0
    energy_2 = 0.0
    h_vector = params.h_vector
    j_vector = params.j_vector
    for k in 1:params.N
        energy_1 += -h_vector[k] * full_expected_values[k][1]
        energy_2 += -h_vector[k] * full_expected_values[k][2]
        if k < params.N
            energy_1 += -j_vector[k] * full_second_moments[k][1]
            energy_2 += -j_vector[k] * full_second_moments[k][2]
        end
    end
    return (energy_1, energy_2)
end




# Just for parallel systems

"""
    system_description(B, params)
Calculate all the relevant observables of the system represented by the TensorTrain B and the system parameters params.
Returns a tuple with:
- full_marginals
- full_second_order_marginals
- full_expected_values
- full_second_moments
- full_correlations
- energy
"""

function system_description(B, params)
    full_prod_sys = full_subproducts_system(B)
    full_marginals = full_marginals_system(B, full_prod_sys, true)
    full_second_order_marginals = full_second_order_marginals_system(B, full_prod_sys, true)
    full_expected_values = marginal_ev_system(full_marginals)
    full_second_moments = second_moment_system(full_second_order_marginals)
    full_correlations = correlation_between_spins_system(full_second_moments, full_expected_values)
    energy = energy_function(full_expected_values, full_second_moments, params)

    return (
        full_marginals,
        full_second_order_marginals,
        full_expected_values,
        full_second_moments,
        full_correlations,
        energy
    )
end