# Fast frequency-space convolution routines for pseudogap solver using SparseIR sampling

using SparseIR, FFTW, LinearAlgebra, Statistics
FFTW.set_num_threads(1)

# ======================================================================================
# 1. Precomputation and Data Structures
# ======================================================================================

"""
Precomputes expansion terms for w^2(k - P), where P = Q/2.
Returns a vector of tuples (g_k, f_P) such that w^2(k-P) ≈ Σ g(k) * f(P).
"""
function precompute_w_sq_terms(k_grid, Nk, a)
    ka = k_grid .* a
    c  = cos.(ka)
    s  = sin.(ka)
    c2 = cos.(2 .* ka)
    s2 = sin.(2 .* ka)

    Ckx  = repeat(c, 1, Nk)
    Skx  = repeat(s, 1, Nk)
    C2kx = repeat(c2, 1, Nk)
    S2kx = repeat(s2, 1, Nk)
    
    Cky  = permutedims(Ckx)
    Sky  = permutedims(Skx)
    C2ky = permutedims(C2kx)
    S2ky = permutedims(S2kx)

    half_mat = fill(0.5, Nk, Nk)
    one_mat  = fill(1.0, Nk, Nk)

    terms = Vector{Tuple{Matrix{Float64}, Matrix{Float64}}}()
    sizehint!(terms, 10)

    # x-component squared terms
    push!(terms, (half_mat, one_mat))
    push!(terms, (0.5 .* C2kx, C2kx)) 
    push!(terms, (0.5 .* S2kx, S2kx))

    # y-component squared terms
    push!(terms, (half_mat, one_mat))
    push!(terms, (0.5 .* C2ky, C2ky))
    push!(terms, (0.5 .* S2ky, S2ky))

    # Cross terms (-2 cos(kx)cos(ky)...)
    push!(terms, (-2.0 .* Ckx .* Cky, Ckx .* Cky))
    push!(terms, (-2.0 .* Ckx .* Sky, Ckx .* Sky))
    push!(terms, (-2.0 .* Skx .* Cky, Skx .* Cky))
    push!(terms, (-2.0 .* Skx .* Sky, Skx .* Sky))

    return terms
end

"""
A structure to hold FFT plans and buffers for convolution operations.
Buffers are pre-allocated to prevent garbage collection overhead during loops.
"""
struct Convolver{T, P_Fwd, P_Bwd}
    buf1::Matrix{Complex{T}}
    buf2::Matrix{Complex{T}}
    buf_out::Matrix{Complex{T}}
    plan_fwd::P_Fwd
    plan_bwd::P_Bwd
    Nk::Int
    inv_Nk2::T
end

function build_convolver(Nk::Int; T=Float64)
    tmp = zeros(Complex{T}, Nk, Nk)
    p_fwd = plan_fft!(tmp, flags=FFTW.MEASURE)
    p_bwd = plan_ifft!(tmp, flags=FFTW.MEASURE)
    
    return Convolver(
        similar(tmp), similar(tmp), similar(tmp),
        p_fwd, p_bwd,
        Nk, 1.0 / (Nk^2)
    )
end

function prepare_G!(C::Convolver, G_2Pmk)
    copyto!(C.buf2, G_2Pmk)
    mul!(C.buf2, C.plan_fwd, C.buf2)
end

# ======================================================================================
# 2. Sigma Calculation Routines
# ======================================================================================

function accumulate_sigma_term!(C::Convolver, g_k, f_P, W_P, sig_acc)
    Nk = C.Nk
    @inbounds for i in eachindex(C.buf1)
        C.buf1[i] = f_P[i] * W_P[i]
    end
    mul!(C.buf1, C.plan_fwd, C.buf1)
    
    out = C.buf_out
    buf_W = C.buf1
    buf_G = C.buf2 
    
    @inbounds for iy in 1:Nk
        iy_A = mod(2 * (iy - 1), Nk) + 1  
        iy_G = mod(-(iy - 1), Nk) + 1    
        for ix in 1:Nk
            ix_A = mod(2 * (ix - 1), Nk) + 1
            ix_G = mod(-(ix - 1), Nk) + 1
            out[ix, iy] = buf_W[ix_A, iy_A] * buf_G[ix_G, iy_G]
        end
    end
    mul!(out, C.plan_bwd, out)
    
    inv_Nk2 = C.inv_Nk2
    @inbounds for i in eachindex(sig_acc)
        sig_acc[i] += g_k[i] * (out[i] * inv_Nk2)
    end
end



# ======================================================================================
# 3. Chi Calculation Routines
# ======================================================================================

function accumulate_chi_term!(C::Convolver, g_k, f_P, G_k, chi_acc)
    Nk = C.Nk
    @inbounds for i in eachindex(C.buf1)
        C.buf1[i] = g_k[i] * G_k[i]
    end
    mul!(C.buf1, C.plan_fwd, C.buf1)
    
    out  = C.buf_out
    buf1 = C.buf1
    buf2 = C.buf2 
    
    @inbounds for i in eachindex(out)
        out[i] = buf1[i] * buf2[i]
    end
    mul!(out, C.plan_bwd, out)

    inv_Nk2 = C.inv_Nk2
    @inbounds for iy_P in 1:Nk
        iy_Q = mod(2 * (iy_P - 1), Nk) + 1
        for ix_P in 1:Nk
            ix_Q = mod(2 * (ix_P - 1), Nk) + 1
            chi_acc[ix_P, iy_P] += f_P[ix_P, iy_P] * (out[ix_Q, iy_Q] * inv_Nk2)
        end
    end
end

# ======================================================================================
# 4. Imaginary Time Routines
# ======================================================================================

# Evaluates basis coefficients on an arbitrary time grid
function evaluate_dense(basis, gl_coeffs, tau_points)
    # G(tau) = G_l * U(tau)^T
    u_mat = basis.u(tau_points)
    
    # Flatten non-basis dimensions, multiply, then reshape back
    sz = size(gl_coeffs)
    gl_flat = reshape(gl_coeffs, prod(sz[1:end-1]), sz[end])
    gt_flat = gl_flat * u_mat
    
    return reshape(gt_flat, sz[1:end-1]..., length(tau_points))
end

# Fits basis coefficients from an arbitrary time grid (Least Squares)
function fit_dense(basis, func_tau, tau_points)
    # Solve: Coeffs * U^T = F(tau)  ->  Coeffs = F(tau) / U^T
    u_mat = basis.u(tau_points)
    
    # Flatten non-time dimensions
    sz = size(func_tau)
    ft_flat = reshape(func_tau, prod(sz[1:end-1]), sz[end])
    
    coeffs = ft_flat / u_mat
    
    nl = size(u_mat, 1)
    return reshape(coeffs, sz[1:end-1]..., nl)
end


"""  
Computes Chi using imaginary time convolution.
For pairing: χ(Q, τ) = (1/N) Σ_p |w(p)|² G(p+Q/2, τ) G(-p+Q/2, τ)
Equal-time correlation corresponding to G(iωₙ) G(iΩₘ-iωₙ) in frequency space.
Then transforms back to Matsubara frequencies using fit.
"""
function compute_chi(G_iwn, w_sq_terms, Nk, smpl_wn_f, smpl_wn_b)
    # Build dense tau points
    beta = smpl_wn_b.basis.beta
    # tau_dense = collect(range(0, beta, length=1000))
    tau_dense = TauSampling(smpl_wn_b.basis).sampling_points

    # Fit G(k, iωₙ) to get expansion coefficients
    G_l = fit(smpl_wn_f, G_iwn, dim=3)

    # Evaluate fermionic G 
    G_tau = evaluate_dense(smpl_wn_f.basis, G_l, tau_dense) # Shape: (Nk, Nk, N_tau)
    
    # Prepare Chi in tau space
    Chi_tau = zeros(ComplexF64, Nk, Nk, length(tau_dense))
    
    # For each bosonic tau point (threaded)
    Threads.@threads for i_tau in 1:length(tau_dense)
        # Build local convolver for this thread
        cv = build_convolver(Nk)
        
        G_k_tau_slice = view(G_tau, :, :, i_tau)
        G_2Pmk_tau_slice = view(G_tau, :, :, i_tau)
        chi_acc = view(Chi_tau, :, :, i_tau)
        
        prepare_G!(cv, G_2Pmk_tau_slice)
        for (g_k, f_P) in w_sq_terms
            accumulate_chi_term!(cv, g_k, f_P, G_k_tau_slice, chi_acc)
        end
    end
   
    
    # Fit Chi(P, τ) back to frequency space
    # Note: Chi is bosonic, use basis_b
    Chi_l = fit_dense(smpl_wn_b.basis, Chi_tau, tau_dense)
    Chi_iwm = evaluate(smpl_wn_b, Chi_l, dim=3)
    
    return Chi_iwm
end

"""
Computes Sigma using imaginary time correlation with tau and -tau.
For pairing: Σ(k, τ) = (1/N) Σ_Q W(Q, τ) w²(k-Q/2) G(Q-k, -τ)
The sum Σ_Ωₘ in frequency space becomes W(τ) G(-τ) in tau space.
Then transforms back to Matsubara frequencies using fit.
"""
function compute_sigma(G_iwn, W_iwm, V, w_sq_terms, Nk, smpl_wn_f, smpl_wn_b)
    # Build dense tau points
    beta = smpl_wn_f.basis.beta
    # tau_dense = collect(range(0, beta, length=1000))
    tau_dense = TauSampling(smpl_wn_f.basis).sampling_points

    # Fit G*(k, iωₙ) to get expansion coefficients
    G_rev_iwn = conj(G_iwn)
    G_rev_l = fit(smpl_wn_f, G_rev_iwn, dim=3)

    # Prepare W in tau space
    W_dyn_iwm = W_iwm .- V
    W_l = fit(smpl_wn_b, W_dyn_iwm, dim=3)
    
    # Evaluate W at fermionic tau points (Sigma is fermionic)
    W_tau = evaluate_dense(smpl_wn_b.basis, W_l, tau_dense)  # Evaluate bosonic W at fermionic tau

    # Evaluate G at all fermionic tau points
    G_rev_tau = evaluate_dense(smpl_wn_f.basis, G_rev_l, tau_dense)
    # Prepare Sigma in tau space
    Sigma_tau_dyn = zeros(ComplexF64, Nk, Nk, length(tau_dense))

    # For each fermionic tau point (threaded)
    Threads.@threads for i_tau in 1:length(tau_dense)
        # Build local convolver for this thread
        cv = build_convolver(Nk)

        G_2Pmk_tau_slice = view(G_rev_tau, :, :, i_tau)
        W_tau_slice = view(W_tau, :, :, i_tau)
        sig_acc = view(Sigma_tau_dyn, :, :, i_tau)

        prepare_G!(cv, G_2Pmk_tau_slice)
        for (g_k, f_P) in w_sq_terms
            accumulate_sigma_term!(cv, g_k, f_P, W_tau_slice, sig_acc)
        end
    end
    
    # Fit Sigma(k, τ) back to frequency space
    Sigma_l = fit_dense(smpl_wn_f.basis, Sigma_tau_dyn, tau_dense)
    Sigma_dyn_iwn = evaluate(smpl_wn_f, Sigma_l, dim=3)
    
    # Static part: Σ_static(k) = (1/N) Σ_P V w²(k - P) G(P, τ=0)
    Sigma_static = zeros(ComplexF64, Nk, Nk)
    
    G_static_slice = view(G_rev_tau .- 0.5, :, :, 1) # G at tau=0
    W_static_slice = fill(ComplexF64(V), Nk, Nk) # Constant V
    
    cv_static = build_convolver(Nk)
    prepare_G!(cv_static, G_static_slice)
    
    for (g_k, f_P) in w_sq_terms
        accumulate_sigma_term!(cv_static, g_k, f_P, W_static_slice, Sigma_static)
    end

    Sigma_iwn = Sigma_dyn_iwn .+ Sigma_static 

    return Sigma_iwn 
end

# ======================================================================================
# 5. Chemical Potential Adjustment Routine
# ======================================================================================
"""
Iteratively adjusts the chemical potential μ to match the target particle density.
Optimized to reuse the `G_temp` buffer and minimize allocations.
"""
function solve_chemical_potential(Sigma_iwn, Ek, smpl_wn_f, Nk, beta, n_target;
                                  max_iter=20, 
                                  tol=1e-5)
    
    # Pre-calculate frequency points from sampling
    # smpl_wn_f.sampling_points contains the MatsubaraFreq objects
    wn_vals = [w.n* π / beta for w in smpl_wn_f.sampling_points]
    N_freq = length(wn_vals)
    
    # Pre-allocate a single buffer for G(k, iωn) to avoid GC overhead in the loop
    G_temp = zeros(ComplexF64, Nk, Nk, N_freq)
    
    # Create tau sampling at β for density calculation
    # n = -2 * G(τ=β⁻)
    basis = smpl_wn_f.basis
    smpl_tau_beta = TauSampling(basis; sampling_points=[beta])
    
    # --- Helper Function: Compute n(μ) ---
    function compute_density(mu_guess)
        # Construct G with guess mu
        @inbounds for i in 1:N_freq
            z = im * wn_vals[i] + mu_guess
            S_slice = view(Sigma_iwn, :, :, i)
            G_slice = view(G_temp, :, :, i)
            @. G_slice = 1.0 / (z - Ek - S_slice)
        end
        
        # Fit and Evaluate
        G_l = fit(smpl_wn_f, G_temp, dim=3)
        G_beta = evaluate(smpl_tau_beta, G_l, dim=3)
        
        # n = -2 * G(β)
        n_calc = -2.0 * mean(real.(view(G_beta, :, :, 1)))
        return n_calc, G_l
    end

    # --- Secant Method Initialization ---
    mu_shift_0 = 0.0
    n_0, G_l_0 = compute_density(mu_shift_0)
    G_l_final = G_l_0
    
    if abs(n_0 - n_target) < tol
        return G_l_final, mu_shift_0, n_0
    end

    # Heuristic guess for second point:
    diff_0 = n_0 - n_target
    mu_shift_1 = mu_shift_0 - sign(diff_0) * 0.5 
    
    n_1 = 0.0
    
    # --- Secant Loop ---
    for iter in 1:max_iter
        n_1, G_l_1 = compute_density(mu_shift_1)
        G_l_final = G_l_1
        
        diff_1 = n_1 - n_target
        
        # Check convergence
        if abs(diff_1) < tol
            return G_l_final, mu_shift_1, n_1
        end
        
        denom = n_1 - n_0
        
        # Guard against division by zero (if function is flat)
        if abs(denom) < 1e-12
            # Fallback to a small push if stuck
            mu_new = mu_shift_1 - sign(diff_1) * 0.1
        else
            mu_new = mu_shift_1 - diff_1 * (mu_shift_1 - mu_shift_0) / denom
        end
        
        # Safety clamp to prevent exploding steps in early iterations
        # (Optional, but good for stability)
        max_step = 50.0 # Allow large steps for t=100 case
        step = mu_new - mu_shift_1
        mu_new = mu_shift_1 + clamp(step, -max_step, max_step)
        
        # Update history
        mu_shift_0, n_0 = mu_shift_1, n_1
        mu_shift_1 = mu_new
    end
    
    # If we exit loop, return best effort
    println("Warning: Chemical potential solver did not fully converge. Final error: $(n_1 - n_target)")
    return G_l_final, mu_shift_1, n_1

end
