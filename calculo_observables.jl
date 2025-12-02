# Tensor_trains 
using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress, TruncBondThresh  


# ============================================
# COMPROBACIONES Y DISTRIBUCIONES MARGINALES
# ============================================

"""
    sum_one_check(B): Verifica que la suma sobre todas las configuraciones de espines de la distribución 
    representada por el TensorTrain B sea 1.
"""
function sum_one_check(B)
    N = length(B.tensors)
    Q = size(B.tensors[1], 3)
    sum_tensor = fill(1.0, 1, 1)
    for i in 1:N
        Bi_sum = zeros(size(B.tensors[i], 1), size(B.tensors[i], 2))
        for q in 1:Q
            Bi_sum .+= B.tensors[i][:,:,q,1]
        end
        sum_tensor *= Bi_sum
    end
    return only(sum_tensor)
end

"""
    marginal_distribution(B, k): Calcula la distribución marginal P(σ_k) para el sitio k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""
function marginal_distribution(B,k)
    N = length(B.tensors)
    K = size(B.tensors[1], 3)
    left_distribution = 1
    for i in 1:(k-1)
        Bi_sum = zeros(size(B.tensors[i], 1), size(B.tensors[i], 2))
        for q in 1:K
            Bi_sum .+= B.tensors[i][:,:,q,1]
        end
        left_distribution *= Bi_sum
    end
    right_distribution = 1
    for i in (k+1):N
        Bi_sum = zeros(size(B.tensors[i], 1), size(B.tensors[i], 2))
        for q in 1:K
            Bi_sum .+= B.tensors[i][:,:,q,1]
        end
        right_distribution *= Bi_sum
    end

    distribution = [(left_distribution * B.tensors[k][:,:, q, 1] * right_distribution)[1] for q in 1:K]
    return distribution
end

"""
    marginal_distribution_system(B): Calcula la distribución marginal P(σ_k) para todos los sitios k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""

function marginal_distribution_system(B)
    N = length(B.tensors)
    distributions = []
    for k in 1:N
        push!(distributions, marginal_distribution(B, k))
    end
    return distributions
end

"""
    marginal_expected_value(B, k): Calcula el valor esperado marginal E[σ_k] para el sitio k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""
function marginal_expected_value_simple(B, k)
    return  marginal_distribution(B, k)[1]*(-1) + marginal_distribution(B, k)[2]*(1)
end

"""
    marginal_expected_value(B, k): Calcula el valor esperado marginal E[σ_k^x], E[σ_k^y] para el sitio k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""
function marginal_expected_value_parallel(B, k)
    dist = marginal_distribution(B, k)
    return (
        (dist[1] + dist[3]) * (-1) + (dist[2] + dist[4]) * (1),
        (dist[1] + dist[2]) * (-1) + (dist[3] + dist[4]) * (1)
    )
end

"""
    marginal_expected_value_system(B): Calcula el valor esperado marginal E[σ_k] para todos los sitios k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""
function marginal_ev_system_inef(B)
    N = length(B.tensors)
    expected_values = zeros(N)
    for k in 1:N
        expected_values[k] = marginal_expected_value_simple(B, k)
    end
    return expected_values
end

"""
    marginal_expected_value_system(B): Calcula el valor esperado marginal E[σ_k] para todos los sitios k 
    a partir del TensorTrain B que representa la distribución conjunta.
"""
function marginal_ev_parallel_system(B)
    N = length(B.tensors)
    expected_values = []
    for k in 1:N
        push!(expected_values, marginal_expected_value_parallel(B, k))
    end
    return expected_values
end



"""
    covariance_between_chains(B)
Calcula la correlación entre las dos cadenas en cada sitio k:
Corr(σ_k^X, σ_k^Y) = E[σ_k^X σ_k^Y] - E[σ_k^X] E[σ_k^Y] 
"""

function covariance_between_chains(B)
    N = length(B.tensors)
    marginals = marginal_distribution_system(B)
    simple_ev = marginal_ev_parallel_system(B)
    correlations = [marginals[k][1]+marginals[k][4]-marginals[k][2]-marginals[k][3] - simple_ev[k][1]*simple_ev[k][2] for k in 1:N]
    return correlations
end

"""
    correlation_between_chains(B)
Calcula la correlación normalizada entre las dos cadenas en cada sitio k:
Corr(σ_k^X, σ_k^Y) = Cov(σ_k^X, σ_k^Y) / (sqrt(1 - E[σ_k^X]^2) * sqrt(1 - E[σ_k^Y]^2))
"""

function correlation_between_chains(B)
    N = length(B.tensors)
    simple_ev = marginal_ev_parallel_system(B)
    covariances = covariance_between_chains(B)
    correlations = [ covariances[k] / (sqrt(1 - simple_ev[k][1]^2) * sqrt(1 - simple_ev[k][2]^2))  for k in 1:N]
    return correlations
end

function max_prod_tt(B, k1, k2)
    k1, k2 = min(k1,k2), max(k1,k2)
    init = 1.0
    if k1 < k2 
        for i in k1:k2-1
            @tullio C[a,b] := B.tensors[i][a,b,k]
            init = init * C
        end
    end
    return init
end

function marginal_distribution(B,k)
    N = length(B.tensors)
    K = size(B.tensors[1], 3)

    left_distribution = max_prod_tt(B, 1, k )
    right_distribution = max_prod_tt(B, k +1, N+1) 

    distribution = [(left_distribution * B.tensors[k][:,:, q, 1] * right_distribution)[1] for q in 1:K]
    return distribution
end
        

function marginal_distribution_second_order(B, k1, k2)
    N = length(B.tensors)
    k1, k2 = min(k1,k2), max(k1,k2)
    left = max_prod_tt(B, 1, k1 )
    middle = max_prod_tt(B, k1 + 1, k2 )
    right = max_prod_tt(B, k2 + 1, N+1) 
    @tullio result[a,b] := left * B.tensors[k1][:,:,a] * middle * B.tensors[k2][:,:,b] * right
    return result
end



function covariance_between_spins_same_chain(B, k1, k2)
    marginals = marginal_distribution_second_order(B, k1, k2)
    simple_ev_k1 = marginal_expected_value_parallel(B, k1)
    simple_ev_k2 = marginal_expected_value_parallel(B, k2)
    second_moment_ch_1 = sum_dif(marginals, [2,4,1,3], [2,4,1,3])
    second_moment_ch_2 = sum_dif(marginals, [3,4,1,2], [3,4,1,2])
    covariance_ch_1 = second_moment_ch_1 - simple_ev_k1[1]*simple_ev_k2[1]
    covariance_ch_2 = second_moment_ch_2 - simple_ev_k1[2]*simple_ev_k2[2]
    return (covariance_ch_1, covariance_ch_2)
end
