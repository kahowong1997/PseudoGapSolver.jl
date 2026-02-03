using Test
using PseudoGapSolver
using SparseIR, LinearAlgebra, Statistics

# ======================================================================================
# 1. Helper Functions (Analytic Reference)
# ======================================================================================

"""
Brute force density using Fermi-Dirac distribution (exact for non-interacting)
n = 2 * mean_k [ f(E_k - μ) ]
where f(E) = 1 / (exp(β*E) + 1)
"""
function compute_density_fermi_dirac(Ek, mu, beta)
    fermi_dirac(E) = E * beta > 100 ? 0.0 : 1.0 / (exp(E * beta) + 1.0)
    
    Ek_shifted = Ek .- mu
    occ = fermi_dirac.(Ek_shifted)
    
    return 2.0 * mean(occ)
end


"""
Analytic calculation of Chi in the non-interacting limit (Σ=0).
Performs the Matsubara sum analytically.
"""
function chi_exact_analytic(Ek_grid, wm_list, k_grid, Nk, a, beta)
    b = π / a
    n_bose = length(wm_list)
    Chi_exact = zeros(ComplexF64, Nk, Nk, n_bose)
    
    nf(E) = 1.0 / (exp(beta * E) + 1.0)
    
    for iPx in 1:Nk, iPy in 1:Nk
        Px, Py = k_grid[iPx], k_grid[iPy]
        
        for idx_m in 1:n_bose
            wm, chi_sum = wm_list[idx_m], 0.0im
            
            for ipx in 1:Nk, ipy in 1:Nk
                px, py = k_grid[ipx], k_grid[ipy]
                
                # FIXED: Added space before ^ to prevent Julia 1.12 ParseError
                w_sq_p = (cos(px * a) - cos(py * a)) ^ 2
                
                kp_x, kp_y = mod(px + Px, 2b), mod(py + Py, 2b)
                ikp_x = argmin(abs.(k_grid .- kp_x))
                ikp_y = argmin(abs.(k_grid .- kp_y))
                
                km_x, km_y = mod(-px + Px, 2b), mod(-py + Py, 2b)
                ikm_x = argmin(abs.(k_grid .- km_x))
                ikm_y = argmin(abs.(k_grid .- km_y))
                
                E1, E2 = Ek_grid[ikp_x, ikp_y], Ek_grid[ikm_x, ikm_y]
                
                numerator = nf(E1) + nf(E2) - 1
                denominator = im * wm - (E1 + E2)
                
                term = abs(denominator) < 1e-12 ? 0.0im : numerator / denominator
                chi_sum += w_sq_p * term
            end
            
            Chi_exact[iPx, iPy, idx_m] = chi_sum / (Nk ^ 2)
        end
    end
    return Chi_exact
end


"""
Analytic calculation of Sigma via direct Matsubara summation.
"""
function sigma_analytic_exact(Ek_grid, wn_list, k_grid, Nk, a, beta, V, Delta)
    n_fermi_pts = length(wn_list)
    Sigma_ref = zeros(ComplexF64, Nk, Nk, n_fermi_pts)

    n_f(x) = 1.0 / (exp(clamp(beta * x, -500, 500)) + 1.0)
    n_b(x) = 1.0 / (exp(clamp(beta * x, -500, 500)) - 1.0)

    for iKx in 1:Nk, iKy in 1:Nk
        Kx, Ky = k_grid[iKx], k_grid[iKy]
        for iPx in 1:Nk, iPy in 1:Nk
            Px, Py = k_grid[iPx], k_grid[iPy]
            Qx, Qy = 2*Px, 2*Py
            Omega_2P = 50.0 * (2.0 - cos(Qx*a/2) - cos(Qy*a/2)) + Delta
            
            ipx = mod(2*(iPx-1) - (iKx-1), Nk) + 1
            ipy = mod(2*(iPy-1) - (iKy-1), Nk) + 1
            E_2Pk = Ek_grid[ipx, ipy]
            
            w_sq = (cos((Kx - Px) * a) - cos((Ky - Py) * a)) ^ 2
            nb, nf = n_b(Omega_2P), n_f(E_2Pk)
            
            for idx_n in 1:n_fermi_pts
                iwn = im * wn_list[idx_n]
                term1 = (nb + nf) / (iwn - Omega_2P + E_2Pk)
                term2 = (nb + 1 - nf) / (iwn + Omega_2P + E_2Pk)
                
                Sigma_ref[iKx, iKy, idx_n] += -1 * w_sq * (term1 + term2)
            end
        end
    end
    return Sigma_ref / (Nk ^ 2)
end


# ======================================================================================
# 2. Test Suites
# ======================================================================================

function run_w_sq_tests()
    @testset "Precompute w_sq_terms" begin
        Nk, a = 8, 3.82
        k_grid = collect(0.0 : 2π/(Nk*a) : 2π/a - 2π/(Nk*a))
        
        w_sq_terms = precompute_w_sq_terms(k_grid, Nk, a)
        errors = Float64[]
        
        for ik in 1:Nk, jk in 1:Nk, iP in 1:Nk, jP in 1:Nk
            kx, ky, Px, Py = k_grid[ik], k_grid[jk], k_grid[iP], k_grid[jP]
            w_sq_direct = (cos((kx - Px) * a) - cos((ky - Py) * a)) ^ 2
            w_sq_recon = sum(g_k[ik, jk] * f_P[iP, jP] for (g_k, f_P) in w_sq_terms)
            push!(errors, abs(w_sq_recon - w_sq_direct))
        end
        @test all(errors .< 1e-12)
    end
end


function run_chemical_potential_tests()
    @testset "Chemical Potential Solver" begin
        Nk, beta, a, b, wmax = 24, 0.05802, 3.82, π/3.82, 1000.0
        k_grid = collect(range(0, 2b, length=Nk+1)[1:Nk])
        KX = [kx for kx in k_grid, _ in k_grid]
        KY = [ky for _ in k_grid, ky in k_grid]
        Ek = -200.0 .* (cos.(KX .* a) .+ cos.(KY .* a))
        n_target = 0.918
        
        smpl_wn_f = MatsubaraSampling(FiniteTempBasis(Fermionic(), beta, wmax, 1e-15))
        Sigma_iwn = zeros(ComplexF64, Nk, Nk, length(smpl_wn_f.sampling_points))

        _, mu_found, n_computed = solve_chemical_potential(
            Sigma_iwn, Ek, smpl_wn_f, Nk, beta, n_target; max_iter=30, tol=1e-6
        )
        
        n_fermi_dirac = compute_density_fermi_dirac(Ek, mu_found, beta)

        println("\nChemical Potential Test:")
        println("  Target density: $n_target")
        println("  Found μ: $mu_found meV")
        println("  n from solve_chemical_potential: $n_computed")
        println("  n from Fermi-Dirac: $n_fermi_dirac")
        println("  Error (vs target): $(abs(n_computed - n_target))")
        println("  Error (vs Fermi-Dirac): $(abs(n_computed - n_fermi_dirac))")
        
        @test abs(n_computed - n_target) < 1e-4
        @test abs(n_fermi_dirac - n_target) < 1e-4
        @test abs(n_computed - n_fermi_dirac) < 1e-4
    end
end


function run_analytic_chi_test()
    @testset "Analytic Validation of Chi" begin
        Nk, beta, a, b, wmax = 16, 0.232, 3.82, π/3.82, 2000.0
        k_grid = collect(range(0, 2b, length=Nk+1)[1:Nk])
        Ek_grid = [-200.0 * (cos(kx*a) + cos(ky*a)) - (-50.0) for kx in k_grid, ky in k_grid]
        
        smpl_wn_f = MatsubaraSampling(FiniteTempBasis(Fermionic(), beta, wmax, 1e-15))
        smpl_wn_b = MatsubaraSampling(FiniteTempBasis(Bosonic(), beta, wmax, 1e-15))
        
        wn_list = [w.n * π / beta for w in smpl_wn_f.sampling_points]
        wm_list = [w.n * π / beta for w in smpl_wn_b.sampling_points]
        
        G_iwn = reshape([1.0 / (im*wn - Ek_grid[ix, iy]) for ix in 1:Nk, iy in 1:Nk, wn in wn_list], Nk, Nk, :)
        
        Chi_tau = compute_chi(G_iwn, precompute_w_sq_terms(k_grid, Nk, a), Nk, smpl_wn_f, smpl_wn_b)
        Chi_exact = chi_exact_analytic(Ek_grid, wm_list, k_grid, Nk, a, beta)
        
        max_diff = maximum(abs.(Chi_tau .- Chi_exact))
        max_val_ref = maximum(abs.(Chi_exact))
        rel_err = max_diff / max_val_ref    
        
        println("\nAnalytic Validation Results (Chi):")
        println("  Max Value in Chi_exact: $max_val_ref")
        println("  Max Absolute Diff:  $max_diff")
        println("  Relative Error:     $rel_err")
        
        @test rel_err < 1e-5
    end
end


function run_sigma_direct_sum_test()
    @testset "Validation of Sigma via Direct Summation" begin
        Nk, beta, a, b, wmax, Delta = 8, 0.232, 3.82, π/3.82, 5000.0, 100.0
        k_grid = collect(range(0, 2b, length=Nk+1)[1:Nk])
        Ek_grid = [-200.0 * (cos(kx*a) + cos(ky*a)) - (-50.0) for kx in k_grid, iy in 1:Nk] # Restored logic

        smpl_wn_f = MatsubaraSampling(FiniteTempBasis(Fermionic(), beta, wmax, 1e-15))
        smpl_wn_b = MatsubaraSampling(FiniteTempBasis(Bosonic(), beta, wmax, 1e-15))
        
        wn_list = [w.n * π / beta for w in smpl_wn_f.sampling_points]
        wm_list = [w.n * π / beta for w in smpl_wn_b.sampling_points]
        
        G_iwn = reshape([1.0 / (im*wn - Ek_grid[ix, iy]) for ix in 1:Nk, iy in 1:Nk, wn in wn_list], Nk, Nk, :)
        
        W_iwm = zeros(ComplexF64, Nk, Nk, length(wm_list))
        for i in 1:length(wm_list), ix in 1:Nk, iy in 1:Nk
            Qx, Qy = 2 * k_grid[ix], 2 * k_grid[iy]
            Omega_q = 50.0 * (2.0 - cos(Qx*a/2) - cos(Qy*a/2)) + Delta
            W_iwm[ix, iy, i] = -(2 * Omega_q) / ((im * wm_list[i]) ^ 2 - Omega_q ^ 2)
        end
        
        Sigma_tau = compute_sigma(G_iwn, W_iwm, 0, precompute_w_sq_terms(k_grid, Nk, a), Nk, smpl_wn_f, smpl_wn_b)
        Sigma_ref = sigma_analytic_exact(Ek_grid, wn_list, k_grid, Nk, a, beta, -400.0, Delta)
        
        max_diff = maximum(abs.(Sigma_tau .- Sigma_ref))
        max_val = maximum(abs.(Sigma_ref))
        rel_err = max_diff / max_val
        
        println("\nSigma Validation Results:")
        println("  Max Absolute Diff:  $max_diff")
        println("  Relative Error:     $rel_err")
        
        @test rel_err < 1e-4
    end
end

# --- Entry Point ---
run_w_sq_tests()
run_chemical_potential_tests()
run_analytic_chi_test()
run_sigma_direct_sum_test()
