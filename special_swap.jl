# Tensor_trains 
using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress, TruncBondThresh  



##################################################################################
##################################################################################

# INVERSO DE UN TT

##################################################################################
##################################################################################

identity_tensor_train(N, qs) = [ones(1,1,qs...) for _ in 1:N] |> TensorTrain
identity_tensor_train(N,qs...) = identity_tensor_train(N,qs)
identity_tensor_train(A::AbstractTensorTrain) = identity_tensor_train(length(A), size(A[1])[3:end])

# Función auxiliar para estimar mejor la norma del TT
function estimate_norm_tt(B)
    norm_prod = 1.0
    for i in 1:length(B)
        norm_prod *= maximum(abs.(B[i]))
    end
    return norm_prod * abs(B.z)
end

function divide_by_constant!(B, constant)
    B.z *= constant
    return B
end

function multiply_by_constant!(B, constant)
    B.z /= constant
    return B
end

function inverse_tt(B, steps, bond)
    B0 = 1 / estimate_norm_tt(B)
    
    # Bn = B0 * (2I - B0 * B)
    temp = multiply_by_constant!(deepcopy(B), B0)
    two = multiply_by_constant!(identity_tensor_train(B), 2)
    Bn = multiply_by_constant!( two - temp, B0)
    
    #Bn = Bn - temp
    
    for _ in 1:steps
        # X_{n+1} = X_n * (2I - B * X_n)
        temp1 = B * Bn
        compress!(temp1; svd_trunc=TruncBond(bond))
        Bnn = two - temp1
        
        Bn = Bnn * Bn
        compress!(Bn; svd_trunc=TruncBond(bond))

        normalize_eachmatrix!(Bn)
    end
    
    return Bn
end




##################################################################################
##################################################################################

# BOLTZMAN DISTRIBUTION TT

##################################################################################
##################################################################################


function boltzman_tt(params, Q::Int = 2, σ = x -> 2x - 3)
    # Inferir N del tamaño de h_vector
    N = length(params.h_vector)
    
    function delta_vector(spin_value)
        return [spin_value == σ(q) ? 1.0 : 0.0 for q in 1:Q]'
    end

    tensors = Array{Float64, 3}[]  # Vector de arrays 3D
    
    h_vector = params.h_vector
    j_vector = params.j_vector
    beta = hasproperty(params, :beta_1) ? params.beta_1 : params.beta

    # ============================================
    # Tensor inicial A1
    # ============================================
    A1 = zeros(1, Q, Q)
    
    for sigma in 1:Q          # σ₁ᵗ
        external_force = exp(beta * h_vector[1] * σ(sigma))
        # Vector fila f₁(σ₁ᵗ, σ₁ᵗ⁺¹, ωⱼ) de dimensión 1×Q
        # ωⱼ representa los posibles valores de σ₂ᵗ
        f_vector = zeros(1, Q)
        for j in 1:Q
            omega_j = σ(j)  # Valor de σ₂ᵗ
            f_vector[1, j] = exp(beta * j_vector[1] * σ(sigma) * omega_j) * external_force
        end
        
        A1[1, :, sigma] = f_vector 
    end
    
    push!(tensors, A1)
    
    # ============================================
    # Tensores intermedios A2, ..., A_{N-1}
    # ============================================
    for i in 2:N-1
        Ai = zeros(Q, Q, Q)
        
        for sigma in 1:Q         # σᵢᵗ
            external_force = exp(beta * h_vector[i] * σ(sigma))
            left_factor = delta_vector( σ( sigma))
            right_factor = zeros(1, Q)

            for j in 1:Q
                omega_j = σ(j)  # Valor de σᵢ₊₁ᵗ
                right_factor[1, j] = exp(beta * j_vector[i] * σ(sigma) * omega_j) * external_force
            end

            Ai[:, :, sigma] = left_factor' * right_factor
        end
        
        push!(tensors, Ai)
    end
    
    # ============================================
    # Tensor final AN
    # ============================================
    AN = zeros(Q, 1, Q)
    
    for sigma in 1:Q    
        external_force = exp(beta * h_vector[N] * σ(sigma))
        AN[:, 1, sigma] = delta_vector(σ( sigma))' * external_force
    end
    
    push!(tensors, AN)
    
    # Crear y retornar el TensorTrain
    return TensorTrain(tensors)
end



# ============================================================================
# TENSOR TRAIN DE EVOLUCIÓN DE LA DISTRIBUCIÓN DE PROBABILIDAD CON SWAP CON SALVA OPCIONAL
# ============================================================================
"""
    tensor_b_t_swap(A, P0, t, bond, s, save = true) Evoluciona la distribución de probabilidad inicial P0 a través de t 
    pasos de tiempo usando la matriz de transición A en formato TensorTrain, aplicando un swap con probabilidad s en cada paso.
    Devuelve una lista con el TensorTrain de la distribución en cada paso de tiempo si save es true.
# Argumentos
- `A`: TensorTrain que representa la matriz de transición
- `P0`: Vector de vectores con la distribución de probabilidad inicial en cada sitio
- `t`: Número de pasos de tiempo a evolucionar
- `bond`: Límite máximo para la compresión del TensorTrain
- `s`: Probabilidad de aplicar el swap en cada paso
- `save`: Booleano que indica si se guarda la distribución en cada paso
"""

function tensor_b_t_swap_acc_to_energy(A, P0, t, bond, swap_energy, save = true)
    B = TensorTrain([(@tullio _[1,1,x] := pi[x]) for pi in P0])
    lista_B_T = save ? [B] : nothing
    swap_idx = [1, 3, 2, 4]
    
    @showprogress for _ in 1:t
        # Evolución temporal
        B = map(zip(A.tensors, B.tensors)) do (A_i, B_i)
            @tullio new_[m1,m2,n1,n2,σ_next] := A_i[m1,n1,σ,σ_next] * B_i[m2,n2,σ]
            @cast _[(m1,m2),(n1,n2),σ_next] := new_[m1,m2,n1,n2,σ_next]
        end |> TensorTrain
        
        # Crear B_swap sin deepcopy: solo reindexar cada tensor
        tensors_swap = [T[:, :, swap_idx] for T in B.tensors]
        B_swap = TensorTrain(tensors_swap; z = B.z)
        
        # Aplicar factores y sumar
        B_swap.tensors[1] *= s
        B.tensors[1] *= (1-s)
        B = B + B_swap
        
        compress!(B; svd_trunc=TruncBond(bond))
        normalize!(B)
        save && push!(lista_B_T, B)
    end
    
    return save ? lista_B_T : B
end

