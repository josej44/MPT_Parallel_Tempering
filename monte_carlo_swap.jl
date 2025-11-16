# ============================================
# ESTRUCTURA DE PARÁMETROS
# ============================================
 
"""
    GlauberParams

Parámetros del sistema de espines.

# Campos
- `beta::Float64`: Temperatura inversa (β = 1/kT)
- `j_vector::Vector{Float64}`: Acoplamientos J entre vecinos (longitud N-1)
- `h_vector::Vector{Float64}`: Campos magnéticos locales (longitud N)
- `p0::Float64`: Probabilidad de mantener el espín sin cambio
"""
struct GlauberParamsParallelSwap
    beta_1::Float64
    beta_2::Float64
    j_vector::Vector{Float64}
    h_vector::Vector{Float64}
    p0::Float64
    s::Float64
    
    function GlauberParamsParallelSwap(beta_1, beta_2, j_vector, h_vector, p0=0.0, s=0.1)
        @assert length(j_vector) == length(h_vector) - 1 "j_vector debe tener longitud N-1"
        @assert 0 <= p0 <= 1 "p0 debe estar en [0,1]"
        new(beta_1, beta_2, j_vector, h_vector, p0, s)
    end
end

# ============================================
# TASA DE TRANSICIÓN (Dinámica de Glauber)
# ============================================

"""
    transition_rate(sigma_neighbors, sigma_new, site_index, params)

Calcula la probabilidad de transición P(σᵢᵗ⁺¹ = sigma_new | configuración actual).
"""
function transition_rate(sigma_neighbors, sigma_new, site_index, params)
    N = length(params.h_vector)
    
    if site_index == 1
        # Sitio 1: solo vecino derecho
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = params.j_vector[1] * sigma_neighbors[2] + params.h_vector[1]
        sigma_current = sigma_neighbors[1]
        
    elseif site_index == N
        # Sitio N: solo vecino izquierdo
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = params.j_vector[end] * sigma_neighbors[1] + params.h_vector[end]
        sigma_current = sigma_neighbors[2]
        
    else
        # Sitio intermedio: vecinos izquierdo y derecho
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = params.j_vector[site_index - 1] * sigma_neighbors[1] + 
                params.j_vector[site_index] * sigma_neighbors[3] + 
                params.h_vector[site_index]
        sigma_current = sigma_neighbors[2]
    end
    
    # Dinámica de Glauber con probabilidad p0 de no cambiar
    glauber_prob = exp(params.beta * sigma_new * h_eff) / (2 * cosh(params.beta * h_eff))
    
    # Agregar componente de mantener el estado
    if sigma_new == sigma_current
        return (1 - params.p0) * glauber_prob + params.p0
    else
        return (1 - params.p0) * glauber_prob
    end
end

# ============================================
# INICIALIZACIÓN
# ============================================

"""
    initialize_spins(N, initial_probs)

Inicializa la cadena de espines con distribución producto independiente.

# Argumentos
- `N::Int`: Número de sitios
- `initial_probs::Vector{Float64}`: Probabilidades p_i de que σᵢ = +1 (longitud N)

# Retorna
- Vector de espines iniciales (±1)
"""
function initialize_spins(N::Int, initial_probs::Vector{Float64})
    @assert length(initial_probs) == N "initial_probs debe tener longitud N"
    @assert all(0 .<= initial_probs .<= 1) "Probabilidades deben estar en [0,1]"
    
    spins = zeros(Int, N)
    for i in 1:N
        spins[i] = rand() < initial_probs[i] ? 1 : -1
    end
    return spins
end

# ============================================
# PASO DE EVOLUCIÓN PARALELA
# ============================================

"""
    parallel_update!(spins_new, spins, params, rng)

Realiza un paso de actualización paralela (todos los espines simultáneamente).

# Argumentos
- `spins_new::Vector{Int}`: Vector para almacenar nueva configuración
- `spins::Vector{Int}`: Configuración actual
- `params::GlauberParams`: Parámetros del sistema
- `rng::AbstractRNG`: Generador de números aleatorios

# Modifica
- `spins_new` con la nueva configuración
"""
function parallel_update!(spins_new, spins, params, rng)
    N = length(spins)
    
    for i in 1:N
        # Preparar vecinos según la posición
        if i == 1
            sigma_neighbors = [spins[1], spins[2]]
        elseif i == N
            sigma_neighbors = [spins[N-1], spins[N]]
        else
            sigma_neighbors = [spins[i-1], spins[i], spins[i+1]]
        end
        
        # Calcular probabilidades para σᵢ = +1 y σᵢ = -1
        p_up = transition_rate(sigma_neighbors, 1, i, params)
        p_down = transition_rate(sigma_neighbors, -1, i, params)
        
        # Normalizar (por seguridad numérica)
        p_total = p_up + p_down
        p_up /= p_total
        
        # Muestrear nuevo estado
        spins_new[i] = rand(rng) < p_up ? 1 : -1
    end
end

# ============================================
# SIMULACIÓN MONTE CARLO
# ============================================

"""
    SimulationResult

Resultado de la simulación Monte Carlo.

# Campos
- `trajectories::Array{Int,2}`: Trayectorias (N_samples × N × T_steps)
- `magnetizations::Matrix{Float64}`: Magnetización promedio por sitio (N_samples × N)
- `correlations::Matrix{Float64}`: Correlaciones espaciales promedio
- `params::GlauberParams`: Parámetros usados
"""
struct SimulationResultParallelSwap
    trajectories_x::Array{Int,3}  # (N_samples, N, T_steps)
    trajectories_y::Array{Int,3}  # (N_samples, N, T_steps)
    magnetizations_x::Matrix{Float64}  # (N_samples, N)
    magnetizations_y::Matrix{Float64}  # (N_samples, N)
    params::GlauberParamsParallelSwap
end

"""
    run_monte_carlo(N, params, initial_probs, T_steps; 
                    N_samples=1000, seed=123, save_trajectory=true)

Ejecuta simulación Monte Carlo de la dinámica de Glauber paralela.

# Argumentos
- `N::Int`: Número de espines
- `params::GlauberParams`: Parámetros del sistema
- `initial_probs::Vector{Float64}`: Distribución inicial P₀
- `T_steps::Int`: Número de pasos temporales

# Argumentos opcionales
- `N_samples::Int=1000`: Número de realizaciones independientes
- `seed::Int=123`: Semilla para reproducibilidad
- `save_trajectory::Bool=true`: Si guardar trayectorias completas

# Retorna
- `SimulationResult` con resultados de la simulación
"""
function run_swap_parallel_monte_carlo(N::Int, params::GlauberParamsParallelSwap, initial_probs::Vector{Float64}, 
                        T_steps::Int; N_samples::Int=1000, seed::Int=123, 
                        save_trajectory::Bool=true)
    
    rng = MersenneTwister(seed)
    
    params_1 = (N = N, beta = params.beta_1, j_vector = params.j_vector, h_vector = params.h_vector, p0 = params.p0)
    params_2 = (N = N, beta = params.beta_2, j_vector = params.j_vector, h_vector = params.h_vector, p0 = params.p0)
    
    # Almacenamiento
    if save_trajectory
        trajectories_x = zeros(Int, N_samples, N, T_steps + 1)
        trajectories_y = zeros(Int, N_samples, N, T_steps + 1)
    else
        trajectories_x = zeros(Int, 0, 0, 0)
        trajectories_y = zeros(Int, 0, 0, 0)
    end
    magnetizations_x = zeros(Float64, N_samples, N)
    magnetizations_y = zeros(Float64, N_samples, N)
    
    # Buffers para eficiencia
    spins_x = zeros(Int, N)
    spins_new_x = zeros(Int, N)
    
    spins_y = zeros(Int, N)
    spins_new_y = zeros(Int, N)

    # Simulación
    for sample in 1:N_samples
        # Inicializar
        spins_x .= initialize_spins(N, initial_probs)
        spins_y .= spins_x  # Misma configuración inicial para ambos sistemas
        
        if save_trajectory
            trajectories_x[sample, :, 1] .= spins_x
            trajectories_y[sample, :, 1] .= spins_y
        end
        
        # Evolución temporal
        for t in 1:T_steps
            parallel_update!(spins_new_x, spins_x, params_1, rng)
            parallel_update!(spins_new_y, spins_y, params_2, rng)
            
            spins_x, spins_new_x = spins_new_x, spins_x  # Swap eficiente
            spins_y, spins_new_y = spins_new_y, spins_y  # Swap eficiente
            
            # Intentar swap de configuraciones
            if rand(rng) < params.s
                # Swap de configuraciones
                spins_x, spins_y = spins_y, spins_x
            end

            if save_trajectory
                trajectories_x[sample, :, t+1] .= spins_x  
            end

            if save_trajectory
                trajectories_y[sample, :, t+1] .= spins_y 
            end
        end
        
        # Guardar magnetización final
        magnetizations_x[sample, :] .= spins_x
        magnetizations_y[sample, :] .= spins_y
    end
    
    return SimulationResultParallelSwap(trajectories_x, trajectories_y, magnetizations_x, magnetizations_y, params)
end



# ============================================
# ANÁLISIS Y OBSERVABLES
# ============================================

"""
    compute_magnetization(result)

Calcula magnetización promedio por sitio.
"""
function compute_magnetization_parallel_swap(result::SimulationResultParallelSwap)
    return mean(result.magnetizations_x, dims=1)[1, :], mean(result.magnetizations_y, dims=1)[1, :]
end

"""
    compute_magnetization_error(result)

Calcula error estándar de la magnetización.
"""
function compute_magnetization_error_parallel_swap(result::SimulationResultParallelSwap)
    return std(result.magnetizations_x, dims=1)[1, :] / sqrt(size(result.magnetizations_x, 1)), std(result.magnetizations_y, dims=1)[1, :] / sqrt(size(result.magnetizations_y, 1))
end

"""
    compute_correlation(result, i, j)

Calcula ⟨σᵢ σⱼ⟩ promedio sobre realizaciones.
"""
function compute_correlation_parallel_swap(result::SimulationResultParallelSwap, i::Int, j::Int)
    return mean(result.magnetizations_x[:, i] .* result.magnetizations_x[:, j]), mean(result.magnetizations_y[:, i] .* result.magnetizations_y[:, j])
end

"""
    compute_marginal_magnetization_parallel_swap(result)

Devuelve la magnetización marginal de cada espín en cada instante de tiempo, para ambos sistemas.
Retorna dos matrices: (mag_x, mag_y) de tamaño (N_x, T_x) y (N_y, T_y).
"""
function compute_marginal_magnetization_parallel_swap(result::SimulationResultParallelSwap)
    N_samples = size(result.trajectories_x, 1)
    N_x = size(result.trajectories_x, 2)
    T_x = size(result.trajectories_x, 3)
    N_y = size(result.trajectories_y, 2)
    T_y = size(result.trajectories_y, 3)
    mag_x = zeros(N_x, T_x)
    mag_y = zeros(N_y, T_y)
    for t in 1:T_x
        for i in 1:N_x
            mag_x[i, t] = mean(result.trajectories_x[sample, i, t] for sample in 1:N_samples)
        end
    end
    for t in 1:T_y
        for i in 1:N_y
            mag_y[i, t] = mean(result.trajectories_y[sample, i, t] for sample in 1:N_samples)
        end
    end
    return mag_x, mag_y
end







# ============================================
# OBSERVABLES ADICIONALES PARA PARALLEL SWAP
# ============================================

"""
    compute_energy(spins, params)

Calcula la energía de una configuración de espines.
E = -∑ᵢ Jᵢ σᵢ σᵢ₊₁ - ∑ᵢ hᵢ σᵢ
"""
function compute_energy(spins::Vector{Int}, j_vector::Vector{Float64}, 
                       h_vector::Vector{Float64})
    N = length(spins)
    energy = 0.0
    
    # Término de interacción
    for i in 1:(N-1)
        energy -= j_vector[i] * spins[i] * spins[i+1]
    end
    
    # Término de campo magnético
    for i in 1:N
        energy -= h_vector[i] * spins[i]
    end
    
    return energy
end

"""
    compute_energy_trajectory(result)

Calcula la energía promedio en cada paso temporal para ambas cadenas.
Retorna dos vectores: (energy_x, energy_y) de longitud T_steps+1
"""
function compute_energy_trajectory(result::SimulationResultParallelSwap)
    N_samples = size(result.trajectories_x, 1)
    T_steps = size(result.trajectories_x, 3)
    
    energy_x = zeros(T_steps)
    energy_y = zeros(T_steps)
    
    for t in 1:T_steps
        energies_x_t = zeros(N_samples)
        energies_y_t = zeros(N_samples)
        
        for sample in 1:N_samples
            spins_x = result.trajectories_x[sample, :, t]
            spins_y = result.trajectories_y[sample, :, t]
            
            energies_x_t[sample] = compute_energy(spins_x, 
                                                   result.params.j_vector, 
                                                   result.params.h_vector)
            energies_y_t[sample] = compute_energy(spins_y, 
                                                   result.params.j_vector, 
                                                   result.params.h_vector)
        end
        
        energy_x[t] = mean(energies_x_t)
        energy_y[t] = mean(energies_y_t)
    end
    
    return energy_x, energy_y
end

"""
    compute_energy_trajectory_with_error(result)

Calcula la energía promedio y su error estándar en cada paso temporal.
Retorna cuatro vectores: (energy_x, error_x, energy_y, error_y)
"""
function compute_energy_trajectory_with_error(result::SimulationResultParallelSwap)
    N_samples = size(result.trajectories_x, 1)
    T_steps = size(result.trajectories_x, 3)
    
    energy_x = zeros(T_steps)
    error_x = zeros(T_steps)
    energy_y = zeros(T_steps)
    error_y = zeros(T_steps)
    
    for t in 1:T_steps
        energies_x_t = zeros(N_samples)
        energies_y_t = zeros(N_samples)
        
        for sample in 1:N_samples
            spins_x = result.trajectories_x[sample, :, t]
            spins_y = result.trajectories_y[sample, :, t]
            
            energies_x_t[sample] = compute_energy(spins_x, 
                                                   result.params.j_vector, 
                                                   result.params.h_vector)
            energies_y_t[sample] = compute_energy(spins_y, 
                                                   result.params.j_vector, 
                                                   result.params.h_vector)
        end
        
        energy_x[t] = mean(energies_x_t)
        error_x[t] = std(energies_x_t) / sqrt(N_samples)
        energy_y[t] = mean(energies_y_t)
        error_y[t] = std(energies_y_t) / sqrt(N_samples)
    end
    
    return energy_x, error_x, energy_y, error_y
end

"""
    compute_pairwise_correlations_trajectory(result, i, j)

Calcula ⟨σᵢ(t) σⱼ(t)⟩ en función del tiempo para ambas cadenas.
Retorna dos vectores: (corr_x, corr_y) de longitud T_steps+1
"""
function compute_pairwise_correlations_trajectory(result::SimulationResultParallelSwap, 
                                                   i::Int, j::Int)
    N_samples = size(result.trajectories_x, 1)
    T_steps = size(result.trajectories_x, 3)
    
    corr_x = zeros(T_steps)
    corr_y = zeros(T_steps)
    
    for t in 1:T_steps
        corr_x[t] = mean(result.trajectories_x[sample, i, t] * 
                        result.trajectories_x[sample, j, t] 
                        for sample in 1:N_samples)
        corr_y[t] = mean(result.trajectories_y[sample, i, t] * 
                        result.trajectories_y[sample, j, t] 
                        for sample in 1:N_samples)
    end
    
    return corr_x, corr_y
end

"""
    compute_all_pairwise_correlations(result, t)

Calcula la matriz de correlaciones ⟨σᵢ σⱼ⟩ para todos los pares en el tiempo t.
Retorna dos matrices: (C_x, C_y) de tamaño (N, N)
"""
function compute_all_pairwise_correlations(result::SimulationResultParallelSwap, t::Int)
    N_samples = size(result.trajectories_x, 1)
    N = size(result.trajectories_x, 2)
    
    C_x = zeros(N, N)
    C_y = zeros(N, N)
    
    for i in 1:N
        for j in 1:N
            C_x[i, j] = mean(result.trajectories_x[sample, i, t] * 
                            result.trajectories_x[sample, j, t] 
                            for sample in 1:N_samples)
            C_y[i, j] = mean(result.trajectories_y[sample, i, t] * 
                            result.trajectories_y[sample, j, t] 
                            for sample in 1:N_samples)
        end
    end
    
    return C_x, C_y
end

"""
    compute_nearest_neighbor_correlations_trajectory(result)

Calcula las correlaciones entre vecinos cercanos ⟨σᵢ(t) σᵢ₊₁(t)⟩ promediadas sobre todos los enlaces.
Retorna dos vectores: (corr_x, corr_y) de longitud T_steps+1
"""
function compute_nearest_neighbor_correlations_trajectory(result::SimulationResultParallelSwap)
    N_samples = size(result.trajectories_x, 1)
    N = size(result.trajectories_x, 2)
    T_steps = size(result.trajectories_x, 3)
    
    corr_x = zeros(T_steps)
    corr_y = zeros(T_steps)
    
    for t in 1:T_steps
        nn_corr_x = 0.0
        nn_corr_y = 0.0
        
        for i in 1:(N-1)
            nn_corr_x += mean(result.trajectories_x[sample, i, t] * 
                             result.trajectories_x[sample, i+1, t] 
                             for sample in 1:N_samples)
            nn_corr_y += mean(result.trajectories_y[sample, i, t] * 
                             result.trajectories_y[sample, i+1, t] 
                             for sample in 1:N_samples)
        end
        
        corr_x[t] = nn_corr_x / (N - 1)
        corr_y[t] = nn_corr_y / (N - 1)
    end
    
    return corr_x, corr_y
end

"""
    compute_susceptibility(result, t)

Calcula la susceptibilidad magnética χ = β⟨(M - ⟨M⟩)²⟩ en el tiempo t.
M = ∑ᵢ σᵢ es la magnetización total.
Retorna dos valores: (chi_x, chi_y)
"""
function compute_susceptibility(result::SimulationResultParallelSwap, t::Int)
    N_samples = size(result.trajectories_x, 1)
    N = size(result.trajectories_x, 2)
    
    # Magnetizaciones totales
    M_x = [sum(result.trajectories_x[sample, :, t]) for sample in 1:N_samples]
    M_y = [sum(result.trajectories_y[sample, :, t]) for sample in 1:N_samples]
    
    # Susceptibilidad
    chi_x = result.params.beta_1 * var(M_x)
    chi_y = result.params.beta_2 * var(M_y)
    
    return chi_x, chi_y
end

"""
    compute_susceptibility_trajectory(result)

Calcula la susceptibilidad en función del tiempo.
Retorna dos vectores: (chi_x, chi_y) de longitud T_steps+1
"""
function compute_susceptibility_trajectory(result::SimulationResultParallelSwap)
    T_steps = size(result.trajectories_x, 3)
    
    chi_x = zeros(T_steps)
    chi_y = zeros(T_steps)
    
    for t in 1:T_steps
        chi_x[t], chi_y[t] = compute_susceptibility(result, t)
    end
    
    return chi_x, chi_y
end

"""
    compute_specific_heat(result, t)

Calcula la capacidad calorífica C = β²⟨(E - ⟨E⟩)²⟩ en el tiempo t.
Retorna dos valores: (C_x, C_y)
"""
function compute_specific_heat(result::SimulationResultParallelSwap, t::Int)
    N_samples = size(result.trajectories_x, 1)
    
    # Energías
    energies_x = zeros(N_samples)
    energies_y = zeros(N_samples)
    
    for sample in 1:N_samples
        spins_x = result.trajectories_x[sample, :, t]
        spins_y = result.trajectories_y[sample, :, t]
        
        energies_x[sample] = compute_energy(spins_x, 
                                            result.params.j_vector, 
                                            result.params.h_vector)
        energies_y[sample] = compute_energy(spins_y, 
                                            result.params.j_vector, 
                                            result.params.h_vector)
    end
    
    # Capacidad calorífica
    C_x = result.params.beta_1^2 * var(energies_x)
    C_y = result.params.beta_2^2 * var(energies_y)
    
    return C_x, C_y
end

"""
    compute_autocorrelation_time(mag_trajectory; threshold=exp(-1))

Estima el tiempo de autocorrelación τ de una trayectoria de magnetización.
Calcula C(t) = ⟨m(t₀)m(t₀+t)⟩ - ⟨m⟩² y encuentra cuando C(τ) ≈ C(0)/e
"""
function compute_autocorrelation_time(mag_trajectory::Vector{Float64}; 
                                      threshold::Float64=exp(-1))
    T = length(mag_trajectory)
    mean_mag = mean(mag_trajectory)
    
    # Calcular autocorrelación
    autocorr = zeros(T÷2)  # Solo calculamos hasta T/2
    
    for lag in 0:(T÷2-1)
        sum_prod = 0.0
        count = 0
        for t in 1:(T-lag)
            sum_prod += (mag_trajectory[t] - mean_mag) * 
                       (mag_trajectory[t+lag] - mean_mag)
            count += 1
        end
        autocorr[lag+1] = sum_prod / count
    end
    
    # Normalizar
    autocorr ./= autocorr[1]
    
    # Encontrar tiempo de autocorrelación
    tau_idx = findfirst(x -> x <= threshold, autocorr)
    
    if tau_idx === nothing
        return T÷2  # No decayó suficiente
    else
        return tau_idx - 1
    end
end

"""
    compute_overlap_trajectory(result)

Calcula el overlap Q(t) = (1/N) ∑ᵢ ⟨σᵢˣ(t) σᵢʸ(t)⟩ entre las dos réplicas.
Retorna un vector de longitud T_steps+1
"""
function compute_overlap_trajectory(result::SimulationResultParallelSwap)
    N_samples = size(result.trajectories_x, 1)
    N = size(result.trajectories_x, 2)
    T_steps = size(result.trajectories_x, 3)
    
    overlap = zeros(T_steps)
    
    for t in 1:T_steps
        overlap_sum = 0.0
        for i in 1:N
            overlap_sum += mean(result.trajectories_x[sample, i, t] * 
                               result.trajectories_y[sample, i, t] 
                               for sample in 1:N_samples)
        end
        overlap[t] = overlap_sum / N
    end
    
    return overlap
end

# ============================================
# FUNCIÓN DE UTILIDAD PARA ANÁLISIS COMPLETO
# ============================================

"""
    compute_all_observables(result; include_correlations=false)

Calcula todos los observables principales y retorna un diccionario con los resultados.
"""
function compute_all_observables(result::SimulationResultParallelSwap; 
                                 include_correlations::Bool=false)
    observables = Dict{String, Any}()
    
    # Magnetización
    mag_x, mag_y = compute_marginal_magnetization_parallel_swap(result)
    observables["magnetization_x"] = mag_x
    observables["magnetization_y"] = mag_y
    
    # Energía
    energy_x, error_x, energy_y, error_y = compute_energy_trajectory_with_error(result)
    observables["energy_x"] = energy_x
    observables["energy_error_x"] = error_x
    observables["energy_y"] = energy_y
    observables["energy_error_y"] = error_y
    
    # Correlaciones de vecinos cercanos
    nn_corr_x, nn_corr_y = compute_nearest_neighbor_correlations_trajectory(result)
    observables["nn_correlation_x"] = nn_corr_x
    observables["nn_correlation_y"] = nn_corr_y
    
    # Susceptibilidad
    chi_x, chi_y = compute_susceptibility_trajectory(result)
    observables["susceptibility_x"] = chi_x
    observables["susceptibility_y"] = chi_y
    
    # Overlap entre réplicas
    observables["overlap"] = compute_overlap_trajectory(result)
    
    # Correlaciones completas (opcional, puede ser costoso)
    if include_correlations
        T_final = size(result.trajectories_x, 3)
        C_x, C_y = compute_all_pairwise_correlations(result, T_final)
        observables["correlation_matrix_x"] = C_x
        observables["correlation_matrix_y"] = C_y
    end
    
    return observables
end


























































# # ============================================
# # OBSERVABLES ADICIONALES PARA PARALLEL SWAP
# # ============================================

# """
#     compute_energy(spins, params)

# Calcula la energía de una configuración de espines.
# E = -∑ᵢ Jᵢ σᵢ σᵢ₊₁ - ∑ᵢ hᵢ σᵢ
# """
# function compute_energy(spins::Vector{Int}, j_vector::Vector{Float64}, 
#                        h_vector::Vector{Float64})
#     N = length(spins)
#     energy = 0.0
    
#     # Término de interacción
#     for i in 1:(N-1)
#         energy -= j_vector[i] * spins[i] * spins[i+1]
#     end
    
#     # Término de campo magnético
#     for i in 1:N
#         energy -= h_vector[i] * spins[i]
#     end
    
#     return energy
# end

# """
#     compute_energy_trajectory(result)

# Calcula la energía promedio en cada paso temporal para ambas cadenas.
# Retorna dos vectores: (energy_x, energy_y) de longitud T_steps+1
# """
# function compute_energy_trajectory(result::SimulationResultParallelSwap)
#     N_samples = size(result.trajectories_x, 1)
#     T_steps = size(result.trajectories_x, 3)
    
#     energy_x = zeros(T_steps)
#     energy_y = zeros(T_steps)
    
#     for t in 1:T_steps
#         energies_x_t = zeros(N_samples)
#         energies_y_t = zeros(N_samples)
        
#         for sample in 1:N_samples
#             spins_x = result.trajectories_x[sample, :, t]
#             spins_y = result.trajectories_y[sample, :, t]
            
#             energies_x_t[sample] = compute_energy(spins_x, 
#                                                    result.params.j_vector, 
#                                                    result.params.h_vector)
#             energies_y_t[sample] = compute_energy(spins_y, 
#                                                    result.params.j_vector, 
#                                                    result.params.h_vector)
#         end
        
#         energy_x[t] = mean(energies_x_t)
#         energy_y[t] = mean(energies_y_t)
#     end
    
#     return energy_x, energy_y
# end

# """
#     compute_energy_trajectory_with_error(result)

# Calcula la energía promedio y su error estándar en cada paso temporal.
# Retorna cuatro vectores: (energy_x, error_x, energy_y, error_y)
# """
# function compute_energy_trajectory_with_error(result::SimulationResultParallelSwap)
#     N_samples = size(result.trajectories_x, 1)
#     T_steps = size(result.trajectories_x, 3)
    
#     energy_x = zeros(T_steps)
#     error_x = zeros(T_steps)
#     energy_y = zeros(T_steps)
#     error_y = zeros(T_steps)
    
#     for t in 1:T_steps
#         energies_x_t = zeros(N_samples)
#         energies_y_t = zeros(N_samples)
        
#         for sample in 1:N_samples
#             spins_x = result.trajectories_x[sample, :, t]
#             spins_y = result.trajectories_y[sample, :, t]
            
#             energies_x_t[sample] = compute_energy(spins_x, 
#                                                    result.params.j_vector, 
#                                                    result.params.h_vector)
#             energies_y_t[sample] = compute_energy(spins_y, 
#                                                    result.params.j_vector, 
#                                                    result.params.h_vector)
#         end
        
#         energy_x[t] = mean(energies_x_t)
#         error_x[t] = std(energies_x_t) / sqrt(N_samples)
#         energy_y[t] = mean(energies_y_t)
#         error_y[t] = std(energies_y_t) / sqrt(N_samples)
#     end
    
#     return energy_x, error_x, energy_y, error_y
# end

# """
#     compute_energy_from_moments(mag_x, mag_y, second_moments_x, second_moments_y, params)

# Calcula la energía usando magnetizaciones marginales y segundos momentos,
# consistente con el método de Tensor Trains.

# E = -∑ᵢ hᵢ⟨σᵢ⟩ - ∑ᵢ Jᵢ⟨σᵢ σᵢ₊₁⟩
# """
# function compute_energy_from_moments(mag_x::Vector{Float64}, mag_y::Vector{Float64},
#                                      second_moments_x::Vector{Float64}, 
#                                      second_moments_y::Vector{Float64},
#                                      j_vector::Vector{Float64}, 
#                                      h_vector::Vector{Float64})
#     N = length(h_vector)
    
#     energy_x = 0.0
#     energy_y = 0.0
    
#     # Término de campo magnético
#     for i in 1:N
#         energy_x -= h_vector[i] * mag_x[i]
#         energy_y -= h_vector[i] * mag_y[i]
#     end
    
#     # Término de interacción
#     for i in 1:(N-1)
#         energy_x -= j_vector[i] * second_moments_x[i]
#         energy_y -= j_vector[i] * second_moments_y[i]
#     end
    
#     return energy_x, energy_y
# end

# """
#     compute_energy_trajectory_from_marginals(result)

# Calcula la energía usando las magnetizaciones marginales y correlaciones,
# método consistente con Tensor Trains. Más preciso que promediar energías individuales.
# """
# function compute_energy_trajectory_from_marginals(result::SimulationResultParallelSwap)
#     N_samples = size(result.trajectories_x, 1)
#     N = size(result.trajectories_x, 2)
#     T_steps = size(result.trajectories_x, 3)
    
#     energy_x = zeros(T_steps)
#     energy_y = zeros(T_steps)
    
#     for t in 1:T_steps
#         # Calcular magnetizaciones marginales
#         mag_x = zeros(N)
#         mag_y = zeros(N)
#         for i in 1:N
#             mag_x[i] = mean(result.trajectories_x[sample, i, t] for sample in 1:N_samples)
#             mag_y[i] = mean(result.trajectories_y[sample, i, t] for sample in 1:N_samples)
#         end
        
#         # Calcular segundos momentos (correlaciones)
#         second_moments_x = zeros(N-1)
#         second_moments_y = zeros(N-1)
#         for i in 1:(N-1)
#             second_moments_x[i] = mean(result.trajectories_x[sample, i, t] * 
#                                        result.trajectories_x[sample, i+1, t] 
#                                        for sample in 1:N_samples)
#             second_moments_y[i] = mean(result.trajectories_y[sample, i, t] * 
#                                        result.trajectories_y[sample, i+1, t] 
#                                        for sample in 1:N_samples)
#         end
        
#         # Calcular energía desde momentos
#         energy_x[t], energy_y[t] = compute_energy_from_moments(
#             mag_x, mag_y, second_moments_x, second_moments_y,
#             result.params.j_vector, result.params.h_vector
#         )
#     end
    
#     return energy_x, energy_y
# end

# """
#     compute_pairwise_correlations_trajectory(result, i, j)

# Calcula ⟨σᵢ(t) σⱼ(t)⟩ en función del tiempo para ambas cadenas.
# Retorna dos vectores: (corr_x, corr_y) de longitud T_steps+1
# """
# function compute_pairwise_correlations_trajectory(result::SimulationResultParallelSwap, 
#                                                    i::Int, j::Int)
#     N_samples = size(result.trajectories_x, 1)
#     T_steps = size(result.trajectories_x, 3)
    
#     corr_x = zeros(T_steps)
#     corr_y = zeros(T_steps)
    
#     for t in 1:T_steps
#         corr_x[t] = mean(result.trajectories_x[sample, i, t] * 
#                         result.trajectories_x[sample, j, t] 
#                         for sample in 1:N_samples)
#         corr_y[t] = mean(result.trajectories_y[sample, i, t] * 
#                         result.trajectories_y[sample, j, t] 
#                         for sample in 1:N_samples)
#     end
    
#     return corr_x, corr_y
# end

# """
#     compute_all_pairwise_correlations(result, t)

# Calcula la matriz de correlaciones ⟨σᵢ σⱼ⟩ para todos los pares en el tiempo t.
# Retorna dos matrices: (C_x, C_y) de tamaño (N, N)
# """
# function compute_all_pairwise_correlations(result::SimulationResultParallelSwap, t::Int)
#     N_samples = size(result.trajectories_x, 1)
#     N = size(result.trajectories_x, 2)
    
#     C_x = zeros(N, N)
#     C_y = zeros(N, N)
    
#     for i in 1:N
#         for j in 1:N
#             C_x[i, j] = mean(result.trajectories_x[sample, i, t] * 
#                             result.trajectories_x[sample, j, t] 
#                             for sample in 1:N_samples)
#             C_y[i, j] = mean(result.trajectories_y[sample, i, t] * 
#                             result.trajectories_y[sample, j, t] 
#                             for sample in 1:N_samples)
#         end
#     end
    
#     return C_x, C_y
# end

# """
#     compute_nearest_neighbor_correlations_trajectory(result)

# Calcula las correlaciones entre vecinos cercanos ⟨σᵢ(t) σᵢ₊₁(t)⟩ promediadas sobre todos los enlaces.
# Retorna dos vectores: (corr_x, corr_y) de longitud T_steps+1
# """
# function compute_nearest_neighbor_correlations_trajectory(result::SimulationResultParallelSwap)
#     N_samples = size(result.trajectories_x, 1)
#     N = size(result.trajectories_x, 2)
#     T_steps = size(result.trajectories_x, 3)
    
#     corr_x = zeros(T_steps)
#     corr_y = zeros(T_steps)
    
#     for t in 1:T_steps
#         nn_corr_x = 0.0
#         nn_corr_y = 0.0
        
#         for i in 1:(N-1)
#             nn_corr_x += mean(result.trajectories_x[sample, i, t] * 
#                              result.trajectories_x[sample, i+1, t] 
#                              for sample in 1:N_samples)
#             nn_corr_y += mean(result.trajectories_y[sample, i, t] * 
#                              result.trajectories_y[sample, i+1, t] 
#                              for sample in 1:N_samples)
#         end
        
#         corr_x[t] = nn_corr_x / (N - 1)
#         corr_y[t] = nn_corr_y / (N - 1)
#     end
    
#     return corr_x, corr_y
# end

# """
#     compute_susceptibility(result, t)

# Calcula la susceptibilidad magnética χ = β⟨(M - ⟨M⟩)²⟩ en el tiempo t.
# M = ∑ᵢ σᵢ es la magnetización total.
# Retorna dos valores: (chi_x, chi_y)
# """
# function compute_susceptibility(result::SimulationResultParallelSwap, t::Int)
#     N_samples = size(result.trajectories_x, 1)
#     N = size(result.trajectories_x, 2)
    
#     # Magnetizaciones totales
#     M_x = [sum(result.trajectories_x[sample, :, t]) for sample in 1:N_samples]
#     M_y = [sum(result.trajectories_y[sample, :, t]) for sample in 1:N_samples]
    
#     # Susceptibilidad
#     chi_x = result.params.beta_1 * var(M_x)
#     chi_y = result.params.beta_2 * var(M_y)
    
#     return chi_x, chi_y
# end

# """
#     compute_susceptibility_trajectory(result)

# Calcula la susceptibilidad en función del tiempo.
# Retorna dos vectores: (chi_x, chi_y) de longitud T_steps+1
# """
# function compute_susceptibility_trajectory(result::SimulationResultParallelSwap)
#     T_steps = size(result.trajectories_x, 3)
    
#     chi_x = zeros(T_steps)
#     chi_y = zeros(T_steps)
    
#     for t in 1:T_steps
#         chi_x[t], chi_y[t] = compute_susceptibility(result, t)
#     end
    
#     return chi_x, chi_y
# end

# """
#     compute_specific_heat(result, t)

# Calcula la capacidad calorífica C = β²⟨(E - ⟨E⟩)²⟩ en el tiempo t.
# Retorna dos valores: (C_x, C_y)
# """
# function compute_specific_heat(result::SimulationResultParallelSwap, t::Int)
#     N_samples = size(result.trajectories_x, 1)
    
#     # Energías
#     energies_x = zeros(N_samples)
#     energies_y = zeros(N_samples)
    
#     for sample in 1:N_samples
#         spins_x = result.trajectories_x[sample, :, t]
#         spins_y = result.trajectories_y[sample, :, t]
        
#         energies_x[sample] = compute_energy(spins_x, 
#                                             result.params.j_vector, 
#                                             result.params.h_vector)
#         energies_y[sample] = compute_energy(spins_y, 
#                                             result.params.j_vector, 
#                                             result.params.h_vector)
#     end
    
#     # Capacidad calorífica
#     C_x = result.params.beta_1^2 * var(energies_x)
#     C_y = result.params.beta_2^2 * var(energies_y)
    
#     return C_x, C_y
# end

# """
#     compute_autocorrelation_time(mag_trajectory; threshold=exp(-1))

# Estima el tiempo de autocorrelación τ de una trayectoria de magnetización.
# Calcula C(t) = ⟨m(t₀)m(t₀+t)⟩ - ⟨m⟩² y encuentra cuando C(τ) ≈ C(0)/e
# """
# function compute_autocorrelation_time(mag_trajectory::Vector{Float64}; 
#                                       threshold::Float64=exp(-1))
#     T = length(mag_trajectory)
#     mean_mag = mean(mag_trajectory)
    
#     # Calcular autocorrelación
#     autocorr = zeros(T÷2)  # Solo calculamos hasta T/2
    
#     for lag in 0:(T÷2-1)
#         sum_prod = 0.0
#         count = 0
#         for t in 1:(T-lag)
#             sum_prod += (mag_trajectory[t] - mean_mag) * 
#                        (mag_trajectory[t+lag] - mean_mag)
#             count += 1
#         end
#         autocorr[lag+1] = sum_prod / count
#     end
    
#     # Normalizar
#     autocorr ./= autocorr[1]
    
#     # Encontrar tiempo de autocorrelación
#     tau_idx = findfirst(x -> x <= threshold, autocorr)
    
#     if tau_idx === nothing
#         return T÷2  # No decayó suficiente
#     else
#         return tau_idx - 1
#     end
# end

# """
#     compute_overlap_trajectory(result)

# Calcula el overlap Q(t) = (1/N) ∑ᵢ ⟨σᵢˣ(t) σᵢʸ(t)⟩ entre las dos réplicas.
# Retorna un vector de longitud T_steps+1
# """
# function compute_overlap_trajectory(result::SimulationResultParallelSwap)
#     N_samples = size(result.trajectories_x, 1)
#     N = size(result.trajectories_x, 2)
#     T_steps = size(result.trajectories_x, 3)
    
#     overlap = zeros(T_steps)
    
#     for t in 1:T_steps
#         overlap_sum = 0.0
#         for i in 1:N
#             overlap_sum += mean(result.trajectories_x[sample, i, t] * 
#                                result.trajectories_y[sample, i, t] 
#                                for sample in 1:N_samples)
#         end
#         overlap[t] = overlap_sum / N
#     end
    
#     return overlap
# end

# # ============================================
# # FUNCIÓN DE UTILIDAD PARA ANÁLISIS COMPLETO
# # ============================================

# """
#     compute_all_observables(result; include_correlations=false)

# Calcula todos los observables principales y retorna un diccionario con los resultados.
# """
# function compute_all_observables(result::SimulationResultParallelSwap; 
#                                  include_correlations::Bool=false)
#     observables = Dict{String, Any}()
    
#     # Magnetización
#     mag_x, mag_y = compute_marginal_magnetization_parallel_swap(result)
#     observables["magnetization_x"] = mag_x
#     observables["magnetization_y"] = mag_y
    
#     # Energía (método consistente con Tensor Trains)
#     energy_x, energy_y = compute_energy_trajectory_from_marginals(result)
#     observables["energy_x"] = energy_x
#     observables["energy_y"] = energy_y
    
#     # Correlaciones de vecinos cercanos
#     nn_corr_x, nn_corr_y = compute_nearest_neighbor_correlations_trajectory(result)
#     observables["nn_correlation_x"] = nn_corr_x
#     observables["nn_correlation_y"] = nn_corr_y
    
#     # Susceptibilidad
#     chi_x, chi_y = compute_susceptibility_trajectory(result)
#     observables["susceptibility_x"] = chi_x
#     observables["susceptibility_y"] = chi_y
    
#     # Overlap entre réplicas
#     observables["overlap"] = compute_overlap_trajectory(result)
    
#     # Correlaciones completas (opcional, puede ser costoso)
#     if include_correlations
#         T_final = size(result.trajectories_x, 3)
#         C_x, C_y = compute_all_pairwise_correlations(result, T_final)
#         observables["correlation_matrix_x"] = C_x
#         observables["correlation_matrix_y"] = C_y
#     end
    
#     return observables
# end