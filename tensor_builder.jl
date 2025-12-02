# Tensor_trains 
using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress, TruncBondThresh  



# ============================================
# CONSTRUCCIÓN DEL TENSOR TRAIN DE TRANSICIÓN
# ============================================

"""
    build_transition_tensorchain(transition_rate, N, params)

Construye un TensorTrain que representa la matriz de transición A(σᵗ, σᵗ⁺¹).

Según la ecuación (4):
- A₁(σ₁ᵗ, σ₁ᵗ⁺¹) = (f₁(σ₁ᵗ, σ₁ᵗ⁺¹, ωⱼ))ⱼ∈[1,Q] · (Iᵣ ⊗ v_σ₁ᵗ)
- Aᵢ(σᵢᵗ, σᵢᵗ⁺¹) = (v_σᵢᵗᵀ ⊗ Iᵣ) · Fᵢ(σᵢᵗ, σᵢᵗ⁺¹) · (Iᵣ ⊗ v_σᵢᵗ)
- Aₙ(σₙᵗ, σₙᵗ⁺¹) = (v_σₙᵗᵀ ⊗ Iᵣ) · (fₙ(σₙᵗ, σₙᵗ⁺¹, ωⱼ))ⱼ∈[1,Q]

donde:
- Q = 2 (número de estados por espín)
- v_y = (δ(y,ω₁), ..., δ(y,ωᵣ)) es el vector delta
- Fᵢ es una matriz con las tasas de transición

# Argumentos
- `transition_rate`: Función (sigma_neighbors, sigma_new, i, params) → probabilidad
- `params`: Parámetros del modelo (j_vector, h_vector, beta)
- `N`: Número de sitios en la cadena
- `Q`: Cantidad de valores posibles de cada espin
- `σ`: Función que asigna a cada índice 1,2,...,Q el valor real del espín (for example (1,2) -> (-1,1))

# Retorna
- `TensorTrain`: Representación compacta de la matriz de transición
"""

function build_transition_tensorchain(transition_rate, params, Q = 2, σ = x -> 2x - 3)
    N = length(params.h_vector)  # Número de sitios en la cadena

    function delta_vector(spin_value)
        return [spin_value == σ(q) ? 1.0 : 0.0 for q in 1:Q]'
    end

    tensors = Array{Float64, 4}[]  # Vector de arrays 4D
    
    # ============================================
    # SITIO 1: A₁[1, Q², σ₁ᵗ, σ₁ᵗ⁺¹]
    # Forma: (f₁(σ₁ᵗ, σ₁ᵗ⁺¹, ωⱼ))ⱼ · (Iᵣ ⊗ v_σ₁ᵗ)
    # f₁ es un vector fila 1×Q
    # v_σ₁ᵗ es un vector fila 1×Q
    # (Iᵣ ⊗ v_σ₁ᵗ) es una matriz Q×Q²
    # Resultado: (1×Q) · (Q×Q²) = 1×Q²
    # ============================================
    A1 = zeros(1, Q^2, Q, Q)
    
    for sigma_t in 1:Q          # σ₁ᵗ
        for sigma_t_plus in 1:Q  # σ₁ᵗ⁺¹
            
            # Vector fila f₁(σ₁ᵗ, σ₁ᵗ⁺¹, ωⱼ) de dimensión 1×Q
            # ωⱼ representa los posibles valores de σ₂ᵗ
            f_vector = zeros(1, Q)
            for j in 1:Q
                omega_j = σ(j)  # Valor de σ₂ᵗ
                neighbors = [σ(sigma_t), omega_j]  # [σ₁ᵗ, σ₂ᵗ]
                f_vector[1, j] = transition_rate(neighbors, σ(sigma_t_plus), 1, params)
            end
            
            # v_σ₁ᵗ es un vector fila 1×Q
            v_sigma = delta_vector(σ(sigma_t))
            
            # (Iᵣ ⊗ v_σ₁ᵗ) es una matriz Q×Q²
            I_kron_v = kron(Matrix(I, Q, Q), v_sigma)  # Q×Q²
            
            # f₁ · (Iᵣ ⊗ v_σ₁ᵗ) = (1×Q) · (Q×Q²) = 1×Q²
            A1[1, :, sigma_t, sigma_t_plus] = f_vector * I_kron_v
        end
    end
    
    push!(tensors, A1)
    
    # ============================================
    # SITIOS INTERMEDIOS: Aᵢ[Q², Q², σᵢᵗ, σᵢᵗ⁺¹]
    # Forma: (v_σᵢᵗᵀ ⊗ Iᵣ) · Fᵢ(σᵢᵗ, σᵢᵗ⁺¹) · (Iᵣ ⊗ v_σᵢᵗ)
    # v_σᵢᵗ es un vector fila 1×Q, entonces v_σᵢᵗᵀ es un vector columna Q×1
    # (v_σᵢᵗᵀ ⊗ Iᵣ) es una matriz Q²×Q
    # Fᵢ es una matriz Q×Q
    # (Iᵣ ⊗ v_σᵢᵗ) es una matriz Q×Q²
    # Resultado: (Q²×Q) · (Q×Q) · (Q×Q²) = Q²×Q²
    # ============================================
    for i in 2:N-1
        Ai = zeros(Q^2, Q^2, Q, Q)
        
        for sigma_t in 1:Q         # σᵢᵗ
            for sigma_t_plus in 1:Q # σᵢᵗ⁺¹
                
                # Fᵢ(σᵢᵗ, σᵢᵗ⁺¹) es una matriz Q×Q
                # F_i[k,l] = fᵢ(ωₖ, σᵢᵗ, σᵢᵗ⁺¹, ωₗ)
                # donde ωₖ = σᵢ₋₁ᵗ y ωₗ = σᵢ₊₁ᵗ
                
                F_matrix = zeros(Q, Q)
                
                for k in 1:Q  # ωₖ = σᵢ₋₁ᵗ (vecino izquierdo)
                    for l in 1:Q  # ωₗ = σᵢ₊₁ᵗ (vecino derecho)
                        omega_k = σ(k)
                        omega_l = σ(l)
                        neighbors = [omega_k, σ(sigma_t), omega_l]  # [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
                        F_matrix[k, l] = transition_rate(neighbors, σ(sigma_t_plus), i, params)
                    end
                end
                
                # v_σᵢᵗ es un vector fila 1×Q
                v_sigma = delta_vector(σ(sigma_t))
                
                # v_σᵢᵗᵀ es un vector columna Q×1
                v_sigma_T = transpose(v_sigma)
                
                # (v_σᵢᵗᵀ ⊗ Iᵣ) es una matriz Q²×Q
                vT_kron_I = kron(v_sigma_T, Matrix(I, Q, Q))
                
                # (Iᵣ ⊗ v_σᵢᵗ) es una matriz Q×Q²
                I_kron_v = kron(Matrix(I, Q, Q), v_sigma)
                
                # Producto: (Q²×Q) · (Q×Q) · (Q×Q²) = Q²×Q²
                Ai[:, :, sigma_t, sigma_t_plus] = vT_kron_I * F_matrix * I_kron_v
            end
        end
        
        push!(tensors, Ai)
    end
    
    # ============================================
    # SITIO N: Aₙ[Q², 1, σₙᵗ, σₙᵗ⁺¹]
    # Forma: (v_σₙᵗᵀ ⊗ Iᵣ) · (fₙ(σₙᵗ, σₙᵗ⁺¹, ωⱼ))ⱼ
    # v_σₙᵗ es un vector fila 1×Q, entonces v_σₙᵗᵀ es un vector columna Q×1
    # (v_σₙᵗᵀ ⊗ Iᵣ) es una matriz Q²×Q
    # fₙ es un vector columna Q×1
    # Resultado: (Q²×Q) · (Q×1) = Q²×1
    # ============================================
    AN = zeros(Q^2, 1, Q, Q)
    
    for sigma_t in 1:Q         # σₙᵗ
        for sigma_t_plus in 1:Q # σₙᵗ⁺¹
            
            # Vector columna fₙ(σₙᵗ, σₙᵗ⁺¹, ωⱼ) de dimensión Q×1
            # ωⱼ representa los posibles valores de σₙ₋₁ᵗ
            f_vector = zeros(Q, 1)
            for j in 1:Q
                omega_j = σ(j)  # Valor de σₙ₋₁ᵗ
                neighbors = [omega_j, σ(sigma_t)]  # [σₙ₋₁ᵗ, σₙᵗ]
                f_vector[j, 1] = transition_rate(neighbors, σ(sigma_t_plus), N, params)
            end
            
            # v_σₙᵗ es un vector fila 1×Q
            v_sigma = delta_vector(σ(sigma_t))
            
            # v_σₙᵗᵀ es un vector columna Q×1
            v_sigma_T = transpose(v_sigma)
            
            # (v_σₙᵗᵀ ⊗ Iᵣ) es una matriz Q²×Q
            vT_kron_I = kron(v_sigma_T, Matrix(I, Q, Q))
            
            # (v_σₙᵗᵀ ⊗ Iᵣ) · fₙ = (Q²×Q) · (Q×1) = Q²×1
            AN[:, 1, sigma_t, sigma_t_plus] = vT_kron_I * f_vector
        end
    end
    
    push!(tensors, AN)
    
    # Crear y retornar el TensorTrain
    return TensorTrain(tensors)
end


# ============================================================================
# TENSOR TRAIN DE EVOLUCIÓN DE LA DISTRIBUCIÓN DE PROBABILIDAD
# ============================================================================

"""
    tensor_b_t(A, P0, t, bond) Evoluciona la distribución de probabilidad inicial P0 a través de t 
    pasos de tiempo usando la matriz de transición A en formato TensorTrain.
# Argumentos
- `A`: TensorTrain que representa la matriz de transición
- `P0`: Vector de vectores con la distribución de probabilidad inicial en cada sitio
- `t`: Número de pasos de tiempo a evolucionar
- `bond`: Límite para la compresión del TensorTrain
"""
function tensor_b_t(A, P0, t, bond)
    N = length(A.tensors)               # Define N como la longitud de A.tensors

    # Construye el TensorTrain inicial para la distribución P0
    # Para cada sitio, crea un tensor de tamaño (1,1,Q) con las probabilidades iniciales.
    B = TensorTrain([(@tullio _[1,1,x] := pi[x]) for pi in P0])        

    # Itera sobre los pasos de tiempo, mostrando una barra de progreso.
    @showprogress for step in 1:t   

        # Para cada sitio, toma el tensor de transición Ai y el tensor de probabilidad Bi
        B = map(zip(A.tensors,B.tensors)) do (A_i, B_i)     
            
            # Realiza la suma sobre σ_t (el estado anterior), multiplicando el tensor de transición 
            # por la distribución. El resultado es un nuevo tensor para el siguiente tiempo.
            @tullio new_tensor_[m1,m2,n1,n2,sigma_t_plus] := A_i[m1,n1,sigma_t,sigma_t_plus] * B_i[m2,n2,sigma_t]

            # Reordena las dimensiones para que los bonds estén agrupados correctamente.
            @cast _[(m1,m2),(n1,n2),sigma_t_plus] := new_tensor_[m1,m2,n1,n2,sigma_t_plus]

        # Crea el nuevo TensorTrain con los tensores actualizados.
        end |> TensorTrain
        compress!(B; svd_trunc=TruncBond(bond)) 
        normalize!(B)
    end
    
    return B
end

# ============================================================================

# ============================================================================================
# TENSOR TRAIN DE EVOLUCIÓN DE LA DISTRIBUCIÓN DE PROBABILIDAD, COS SALVA PARA CADA T
# ============================================================================================

"""
    tensor_b_t(A, P0, t, bond) Evoluciona la distribución de probabilidad inicial P0 a través de t 
    pasos de tiempo usando la matriz de transición A en formato TensorTrain. Devuelve una lista con el 
    TensorTrain de la distribución en cada paso de tiempo.
# Argumentos
- `A`: TensorTrain que representa la matriz de transición
- `P0`: Vector de vectores con la distribución de probabilidad inicial en cada sitio
- `t`: Número de pasos de tiempo a evolucionar
- `bond`: Límite máximo para la compresión del TensorTrain
"""
function tensor_b_t_over_time(A, P0, t, bond)
    N = length(A.tensors)               # Define N como la longitud de A.tensors

    lista_B_T =[]

    # Construye el TensorTrain inicial para la distribución P0
    # Para cada sitio, crea un tensor de tamaño (1,1,Q) con las probabilidades iniciales.
    B = TensorTrain([(@tullio _[1,1,x] := pi[x]) for pi in P0])        
    push!(lista_B_T, B)
    # Itera sobre los pasos de tiempo, mostrando una barra de progreso.
    @showprogress for step in 1:t   

        # Para cada sitio, toma el tensor de transición Ai y el tensor de probabilidad Bi
        B = map(zip(A.tensors,B.tensors)) do (A_i, B_i)     
            
            # Realiza la suma sobre σ_t (el estado anterior), multiplicando el tensor de transición 
            # por la distribución. El resultado es un nuevo tensor para el siguiente tiempo.
            @tullio new_tensor_[m1,m2,n1,n2,sigma_t_plus] := A_i[m1,n1,sigma_t,sigma_t_plus] * B_i[m2,n2,sigma_t]

            # Reordena las dimensiones para que los bonds estén agrupados correctamente.
            @cast _[(m1,m2),(n1,n2),sigma_t_plus] := new_tensor_[m1,m2,n1,n2,sigma_t_plus]

        # Crea el nuevo TensorTrain con los tensores actualizados.
        end |> TensorTrain
        compress!(B; svd_trunc=TruncBond(bond)) 
        normalize!(B)
        push!(lista_B_T, B)
    end
    
    return lista_B_T
end



# ============================================================================

# ============================================
# TRABAJO CON TENSOR TRAINS
# ============================================

"""
* Multiplicación Kronecker de dos tensores en formato Tensor Train con dimensiones físicas coincidentes.
"""
function Base.:*(A::T, B::T) where T<:AbstractTensorTrain
    # Recorre los pares de tensores correspondientes de A y B (uno por cada sitio) usando zip. Para cada par (a, b), ejecuta el bloque siguiente y guarda el resultado en una nueva lista C.
    C = map(zip(A.tensors, B.tensors)) do (a, b)  # ← Usar zip explícitamente
        
        # Verifica que las dimensiones físicas (las últimas) de ambos tensores coincidan (por ejemplo, los índices de espín).
        @assert size(a)[3:end] == size(b)[3:end]

        # Colapsa todas las dimensiones físicas en una sola (de tamaño igual al producto de las dimensiones físicas), dejando los dos primeros ejes (bond dimensions) intactos.
        ar = reshape(a, size(a,1), size(a,2), prod(size(a)[3:end]))
        br = reshape(b, size(b,1), size(b,2), prod(size(b)[3:end]))
        
        # Usa el macro @cast (de TensorCast) para calcular el producto de Kronecker entre los bond dimensions de a y b y multiplica elemento a elemento la dimensión física. 
        # El resultado c tiene bond dimensions combinadas: (ia,ib) y (ja,jb), y la dimensión física x. 
        @cast c[(ia,ib),(ja,jb),x] := ar[ia,ja,x] * br[ib,jb,x]

        # Devuelve el tensor resultante con bond dimensions combinadas y las dimensiones físicas originales (descolapsadas).
        reshape(c, size(c,1), size(c,2), size(a)[3:end]...)
    end

    # Construye y retorna un nuevo TensorTrain (tipo T) usando los tensores C y multiplicando los factores escalares z de ambos operandos.
    T(C; z = A.z * B.z)
end


"""
    mult_sep(A, B): Multiplica dos TensorTrains A y B separando las dimensiones físicas.
"""
function mult_sep(A, B)
    d = map(zip(A.tensors,B.tensors)) do (a,b)
        @tullio c[m1,m2,n1,n2,x,y,x1,y1] := a[m1,n1,x,x1] * b[m2,n2,y,y1]
        @cast _[(m1,m2),(n1,n2),(x,y),(x1,y1)] := c[m1,m2,n1,n2,x,y,x1,y1]
    end
    return TensorTrain(d; z = A.z * B.z)   
end



# ============================================
# TASA DE TRANSICIÓN (Dinámica de Glauber)
# ============================================

"""
    transition_rate(sigma_neighbors, sigma_new, site_index, params)

Calcula la probabilidad de transición P(σᵢᵗ⁺¹ = sigma_new | configuración actual).

# Argumentos
- `sigma_neighbors`: Valores de espines vecinos relevantes
  * Sitio 1: [σ₁ᵗ, σ₂ᵗ]
  * Sitio i (intermedio): [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
  * Sitio N: [σₙ₋₁ᵗ, σₙᵗ]
- `sigma_new`: Nuevo valor del espín (±1)
- `site_index`: Índice del sitio (1 a N)
- `params`: Parámetros (beta, j_vector, h_vector)

# Retorna
- Probabilidad según dinámica de Glauber
"""
function transition_rate(sigma_neighbors, sigma_new, site_index, params)
    N = length(params.h_vector)
    
    if site_index == 1
        # Sitio 1: solo vecino derecho
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = params.j_vector[1] * sigma_neighbors[2] + params.h_vector[1]
        
    elseif site_index == N
        # Sitio N: solo vecino izquierdo
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = params.j_vector[end] * sigma_neighbors[1] + params.h_vector[end]
        
    else
        # Sitio intermedio: vecinos izquierdo y derecho
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = params.j_vector[site_index - 1] * sigma_neighbors[1] + 
                params.j_vector[site_index] * sigma_neighbors[3] + 
                params.h_vector[site_index]
    end
    
    # Dinámica de Glauber
    return (exp(params.beta * sigma_new * h_eff) / (2 * cosh(params.beta * h_eff)))*(1-params.p0) + params.p0* (sigma_new == sigma_neighbors[site_index == 1 ? 1 : site_index == N ? 2 : 2] ? 1.0 : 0.0)
end


function transition_glauber_rate(sigma_neighbors, sigma_new, site_index, params)
    N = length(params.h_vector)
    
    if site_index == 1
        # Sitio 1: solo vecino derecho
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = params.j_vector[1] * sigma_neighbors[2] + params.h_vector[1]
        
    elseif site_index == N
        # Sitio N: solo vecino izquierdo
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = params.j_vector[end] * sigma_neighbors[1] + params.h_vector[end]
        
    else
        # Sitio intermedio: vecinos izquierdo y derecho
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = params.j_vector[site_index - 1] * sigma_neighbors[1] + 
                params.j_vector[site_index] * sigma_neighbors[3] + 
                params.h_vector[site_index]
    end
    
    # Dinámica de Glauber
    return (exp(params.beta * sigma_new * h_eff) / (2 * cosh(params.beta * h_eff))) 
end



# ============================================================
# TENSOR TRAIN DE TRANSICIÓN PARA TEMPERATURAS PARALELAS
# ============================================================

"""
    parallel_transition_tensorchain(transition_rate, N, params)

Construye un TensorTrain que representa la matriz de transición A((x,y)ᵗ, (x,y)ᵗ⁺¹).

# Argumentos
- `transition_rate`: Función (sigma_neighbors, sigma_new, i, params) → probabilidad
- `params`: Parámetros del modelo (j_vector, h_vector, beta_1, beta_2, N)
- `N`: Número de sitios en la cadena
- `Q`: Cantidad de valores posibles de cada espin
- `σ`: Función que asigna a cada índice 1,2,...,Q el valor real del espín (for example (1,2) -> (-1,1))

# Retorna
- `TensorTrain`: Representación compacta de la matriz de transición mixta
"""

function parallel_transition_tensor_train(transition_rate, params, Q = 2, σ = x -> 2x - 3)
    N = params.N
    params_1 = (N = N, beta = params.beta_1, j_vector = params.j_vector, h_vector = params.h_vector, p0 = params.p0)
    params_2 = (N = N, beta = params.beta_2, j_vector = params.j_vector, h_vector = params.h_vector, p0 = params.p0)
    A1 = build_transition_tensorchain(transition_rate, params_1, Q, σ)
    A2 = build_transition_tensorchain(transition_rate, params_2, Q, σ)
    return mult_sep(A1, A2) 
end


# ============================================
# FUNCIONES AUXILIARES
# ============================================

"""
σ(x) Mapea índice de espín a valor físico: 1 → -1, 2 → +1
"""
σ(x) = 2x - 3


# ============================================
# SELECCIÓN DE PARÁMETROS ALEATORIOS
# ============================================

"""
    random_params(N): Genera parámetros aleatorios para el modelo de Ising.
"""
function random_params(N)
    a, b = -1.0, 1.0
    params = (
        N = N, 
        beta = rand(),                              # Inversa de la temperatura (β = 1/kT)
        j_vector = a .+ (b - a) .* rand(N-1) ,        # Acoplamientos J_{i,i+1} (N-1 elementos)
        h_vector = a .+ (b - a) .* rand(N) ,    # Campos externos h_i (N elementos)
        p0 = rand(),                                 # Probabilidad de mantener configuración,
    )
    return params
end

"""
    random_params(N): Genera parámetros aleatorios para el modelo de Ising paralelo.
"""
function parallel_random_params(N)
    a, b = -1.0, 1.0
    params = (
        N = N, 
        beta_1 = rand(),                              # Inversa de la temperatura (β = 1/kT)
        beta_2 = rand(),                              # Inversa de la temperatura (β = 1/kT)
        j_vector = a .+ (b - a) .* rand(N-1) ,        # Acoplamientos J_{i,i+1} (N-1 elementos)
        h_vector = a .+ (b - a) .* rand(N) ,          # Campos externos h_i (N elementos)
        p0 = rand()                                   # Probabilidad de mantener configuración,
    )
    return params
end

"""
    random_P0(N, Q): Genera una distribución de probabilidad inicial aleatoria normalizada para N sitios 
    y Q estados por espín.
"""
function random_P0(N, Q = 2)
    P0 = [rand(Q) for _ in 1:N]
    for i in 1:N
        P0[i] /= sum(P0[i])  # Normaliza cada vector de probabilidad
    end
    return P0
end


"""
    parallel_random_P0_fixed(N): Genera una distribución de probabilidad inicial fija para N sitios 
    en el modelo paralelo.
"""
function parallel_random_P0_fixed(N)
    P0 = [Float64[rand(), 0.0, 0.0, rand()] for _ in 1:N]
    for i in 1:N
        P0[i] ./= sum(P0[i])  # Normaliza cada vector de probabilidad
    end
    return P0
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

function tensor_b_t_swap(A, P0, t, bond, s, save = true)
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







#swap(i) = (i == 1 ? 1 : i == 2 ? 3 : i == 3 ? 2 : i == 4 ? 4 : error("swap solo definido para i=1..4"))


# function doble_tensor_b_t_tplus_swap(A, B, bond, s)
#     swap_A_idx = [1,2,3,4, 9,10,11,12, 5,6,7,8, 13,14,15,16]
    
#     # Crear B_swap sin deepcopy: solo reindexar cada tensor
#     tensors_A_swap = [T[:, :, swap_A_idx] for T in A.tensors]
#     A_swap = TensorTrain(tensors_A_swap; z = A.z) 
    
#     # Evolución temporal
#     B_ = map(zip(A.tensors, B.tensors)) do (A_i, B_i)
#         @tullio new_[m1,m2,n1,n2,σ, σ_next] := A_i[m1,n1,σ,σ_next] * B_i[m2,n2,σ]
#         @cast _[(m1,m2),(n1,n2),(σ,σ_next)] := new_[m1,m2,n1,n2,σ,σ_next]
#     end |> TensorTrain

#     B_swap = map(zip(A_swap.tensors, B.tensors)) do (A_i, B_i)
#         @tullio new_[m1,m2,n1,n2,σ, σ_next] := A_i[m1,n1,σ,σ_next] * B_i[m2,n2,σ]
#         @cast _[(m1,m2),(n1,n2),(σ, σ_next)] := new_[m1,m2,n1,n2,σ, σ_next]
#     end |> TensorTrain
    
#     # Aplicar factores y sumar
#     B_swap.tensors[1] *= s
#     B_.tensors[1] *= (1-s)
#     B = B_ + B_swap
    
#     compress!(B; svd_trunc=TruncBond(bond))
#     normalize!(B)
    
#     return B
# end
