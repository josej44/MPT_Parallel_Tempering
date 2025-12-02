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
mutable struct GlauberParams
    beta::Float64
    j_vector::Vector{Float64}
    h_vector::Vector{Float64}
    p0::Float64
    
    function GlauberParams(beta, j_vector, h_vector, p0=0.0)
        @assert length(j_vector) == length(h_vector) - 1 "j_vector debe tener longitud N-1"
        @assert 0 <= p0 <= 1 "p0 debe estar en [0,1]"
        new(beta, j_vector, h_vector, p0)
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

"""
    initialize_spins_uniform(N, p_up=0.5)

Inicializa con probabilidad uniforme.
"""
function initialize_spins_uniform(N::Int, p_up::Float64=0.5)
    return initialize_spins(N, fill(p_up, N))
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
struct SimulationResult
    trajectories::Array{Int,3}  # (N_samples, N, T_steps)
    magnetizations::Matrix{Float64}  # (N_samples, N)
    params::GlauberParams
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
function run_monte_carlo(N::Int, params::GlauberParams, initial_probs::Vector{Float64}, 
                        T_steps::Int; N_samples::Int=1000, seed::Int=123, 
                        save_trajectory::Bool=true)
    
    rng = MersenneTwister(seed)
    
    # Almacenamiento
    if save_trajectory
        trajectories = zeros(Int, N_samples, N, T_steps + 1)
    else
        trajectories = zeros(Int, 0, 0, 0)
    end
    magnetizations = zeros(Float64, N_samples, N)
    
    # Buffers para eficiencia
    spins = zeros(Int, N)
    spins_new = zeros(Int, N)
    
    # Simulación
    for sample in 1:N_samples
        # Inicializar
        spins .= initialize_spins(N, initial_probs)
        
        if save_trajectory
            trajectories[sample, :, 1] .= spins
        end
        
        # Evolución temporal
        for t in 1:T_steps
            parallel_update!(spins_new, spins, params, rng)
            spins, spins_new = spins_new, spins  # Swap eficiente
            
            if save_trajectory
                trajectories[sample, :, t+1] .= spins
            end
        end
        
        # Guardar magnetización final
        magnetizations[sample, :] .= spins
    end
    
    return SimulationResult(trajectories, magnetizations, params)
end

# ============================================
# ANÁLISIS Y OBSERVABLES
# ============================================

"""
    compute_magnetization(result)

Calcula magnetización promedio por sitio.
"""
function compute_magnetization(result::SimulationResult)
    return mean(result.magnetizations, dims=1)[1, :]
end

"""
    compute_magnetization_error(result)

Calcula error estándar de la magnetización.
"""
function compute_magnetization_error(result::SimulationResult)
    return std(result.magnetizations, dims=1)[1, :] / sqrt(size(result.magnetizations, 1))
end

"""
    compute_correlation(result, i, j)

Calcula ⟨σᵢ σⱼ⟩ promedio sobre realizaciones.
"""
function compute_correlation(result::SimulationResult, i::Int, j::Int)
    return mean(result.magnetizations[:, i] .* result.magnetizations[:, j])
end



"""
    compute_trajectory_magnetization(result, sample)

Calcula evolución temporal de magnetización total para una muestra.
"""

function compute_trajectory_magnetization(result::SimulationResult, sample::Int)
    T = size(result.trajectories, 3)
    N = size(result.trajectories, 2)
    mag = zeros(T)
    for t in 1:T
        mag[t] = sum(result.trajectories[sample, :, t]) / N
    end
    return mag
end


function compute_marginal_magnetization_parallel(result)
    N_samples = size(result.trajectories, 1)
    N_x = size(result.trajectories, 2)
    T_x = size(result.trajectories, 3)
    mag_x = zeros(N_x, T_x)
    for t in 1:T_x
        for i in 1:N_x
            mag_x[i, t] = mean(result.trajectories[sample, i, t] for sample in 1:N_samples)
        end
    end
    return mag_x
end






















# ============================================
# DINÁMICA SECUENCIAL ALEATORIA
# ============================================
# Agregar estas funciones AL FINAL de tu archivo .jl existente
# (después de todas las funciones del Glauber paralelo)

"""
    sequential_update!(spins, params, rng)

Realiza un paso de actualización secuencial ALEATORIA de Glauber.
En cada paso temporal se escoge UN espín al azar y solo ese se actualiza.

# Argumentos
- `spins::Vector{Int}`: Configuración actual (se modifica in-place)
- `params::GlauberParams`: Parámetros del sistema
- `rng::AbstractRNG`: Generador de números aleatorios

# Modifica
- `spins` actualizando un único espín escogido aleatoriamenteparams
"""
function sequential_update!(spins, params, rng)
    N = length(spins)
    
    # Escoger un sitio al azar
    i = rand(rng, 1:N)
    
    params.p0 = 0.0
    # params_new = merge(params, (p0 = 0.0,))
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
    
    # Muestrear y actualizar el espín escogido
    spins[i] = rand(rng) < p_up ? 1 : -1
end

"""
    run_monte_carlo_sequential(N, params, initial_probs, T_steps; 
                               N_samples=1000, seed=123, save_trajectory=true)

Ejecuta simulación Monte Carlo de la dinámica de Glauber SECUENCIAL ALEATORIA.

# Argumentos
- `N::Int`: Número de espines
- `params::GlauberParams`: Parámetros del sistema
- `initial_probs::Vector{Float64}`: Distribución inicial P₀
- `T_steps::Int`: Número de pasos temporales (actualizaciones individuales)

# Argumentos opcionales
- `N_samples::Int=1000`: Número de realizaciones independientes
- `seed::Int=123`: Semilla para reproducibilidad
- `save_trajectory::Bool=true`: Si guardar trayectorias completas

# Retorna
- `SimulationResult` con resultados de la simulación

# Nota
En cada paso temporal se escoge UN espín al azar y solo ese se actualiza.
Para comparar con la dinámica paralela, considera usar T_steps ≈ N × T_parallel,
ya que en promedio cada espín es actualizado una vez cada N pasos.
"""
function run_monte_carlo_sequential(N::Int, params::GlauberParams, 
                                   initial_probs::Vector{Float64}, 
                                   T_steps::Int; 
                                   N_samples::Int=1000, 
                                   seed::Int=123, 
                                   save_trajectory::Bool=true)
    
    rng = MersenneTwister(seed)
    
    # Almacenamiento
    if save_trajectory
        trajectories = zeros(Int, N_samples, N, T_steps + 1)
    else
        trajectories = zeros(Int, 0, 0, 0)
    end
    magnetizations = zeros(Float64, N_samples, N)
    
    # Buffer para la configuración
    spins = zeros(Int, N)
    
    # Simulación
    for sample in 1:N_samples
        # Inicializar
        spins .= initialize_spins(N, initial_probs)
        
        if save_trajectory
            trajectories[sample, :, 1] .= spins
        end
        
        # Evolución temporal
        for t in 1:T_steps
            sequential_update!(spins, params, rng)
            
            if save_trajectory
                trajectories[sample, :, t+1] .= spins
            end
        end
        
        # Guardar magnetización final
        magnetizations[sample, :] .= spins
    end
    
    return SimulationResult(trajectories, magnetizations, params)
end

"""
    compare_dynamics(N, params, initial_probs, T_parallel; 
                    N_samples=1000, seed=123)

Compara las dinámicas paralela y secuencial con escalas temporales apropiadas.

# Argumentos
- `T_parallel::Int`: Número de pasos para la dinámica paralela
- Automáticamente usa T_sequential = N × T_parallel

# Retorna
- `(result_parallel, result_sequential)`: Tupla con ambos resultados
"""
function compare_dynamics(N::Int, params::GlauberParams, 
                         initial_probs::Vector{Float64}, 
                         T_parallel::Int; 
                         N_samples::Int=1000, 
                         seed::Int=123)
    
    T_sequential = N * T_parallel  # Escala temporal apropiada
    
    println("Ejecutando dinámica paralela ($T_parallel pasos)...")
    result_parallel = run_monte_carlo(N, params, initial_probs, T_parallel; 
                                     N_samples=N_samples, seed=seed, 
                                     save_trajectory=true)
    
    println("Ejecutando dinámica secuencial ($T_sequential pasos)...")
    result_sequential = run_monte_carlo_sequential(N, params, initial_probs, T_sequential; 
                                                   N_samples=N_samples, seed=seed, 
                                                   save_trajectory=true)
    
    println("\nComparación de magnetizaciones finales:")
    mag_par = compute_magnetization(result_parallel)
    mag_seq = compute_magnetization(result_sequential)
    
    for i in 1:N
        println("Sitio $i: Paralela = $(round(mag_par[i], digits=4)), " *
                "Secuencial = $(round(mag_seq[i], digits=4))")
    end
    
    return (result_parallel, result_sequential)
end




# ============================================
# OBSERVABLES: ENERGÍA
# ============================================

"""
    compute_energy(spins, params)

Calcula la energía total de una configuración de espines.
E = -∑ᵢ Jᵢ σᵢ σᵢ₊₁ - ∑ᵢ hᵢ σᵢ
"""
function compute_energy(spins, params)
    N = length(spins)
    energy = 0.0
    
    # Término de interacción
    for i in 1:N-1
        energy -= params.j_vector[i] * spins[i] * spins[i+1]
    end
    
    # Término de campo externo
    for i in 1:N
        energy -= params.h_vector[i] * spins[i]
    end
    
    return energy
end

"""
    compute_trajectory_energy(result, sample)

Calcula la evolución temporal de la energía para una muestra específica.
"""
function compute_trajectory_energy(result::SimulationResult, sample::Int)
    T = size(result.trajectories, 3)
    energy = zeros(T)
    
    for t in 1:T
        spins = result.trajectories[sample, :, t]
        energy[t] = compute_energy(spins, result.params)
    end
    
    return energy
end

"""
    compute_average_trajectory_energy(result)

Calcula la energía promedio sobre todas las muestras en cada paso temporal.

# Retorna
- `(energy_mean, energy_std)`: Tupla con energía promedio y desviación estándar
"""
function compute_average_trajectory_energy(result::SimulationResult)
    T = size(result.trajectories, 3)
    N_samples = size(result.trajectories, 1)
    
    energy_mean = zeros(T)
    energy_std = zeros(T)
    
    for t in 1:T
        energies = zeros(N_samples)
        for sample in 1:N_samples
            spins = result.trajectories[sample, :, t]
            energies[sample] = compute_energy(spins, result.params)
        end
        energy_mean[t] = mean(energies)
        energy_std[t] = std(energies)
    end
    
    return energy_mean, energy_std
end