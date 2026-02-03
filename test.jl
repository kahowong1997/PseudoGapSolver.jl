using Test
include(joinpath(@__DIR__, "FFT_solver.jl"))

function run_w_sq_tests()
    @testset "Precompute w_sq_terms" begin
        Nk = 8
        a = 3.82
        k_grid = collect(0.0:2π/(Nk*a):2π/a - 2π/(Nk*a))

        w_sq_terms = precompute_w_sq_terms(k_grid, Nk, a)

        errors = Float64[]

        for ik in 1:Nk, jk in 1:Nk
            kx, ky = k_grid[ik], k_grid[jk]
            for iP in 1:Nk, jP in 1:Nk
                Px, Py = k_grid[iP], k_grid[jP]
                
                # Direct calculation
                w_sq_direct = (cos((kx - Px)*a) - cos((ky - Py)*a))^2
                
                # Reconstructed from precomputed terms
                w_sq_recon = sum(g_k[ik, jk] * f_P[iP, jP] for (g_k, f_P) in w_sq_terms)
                
                push!(errors, abs(w_sq_recon - w_sq_direct))
            end
        end

        @test all(errors .< 1e-12)
    end
end



"""
Brute force density using Fermi-Dirac distribution (exact for non-interacting)
n = 2 * mean_k [ f(E_k - μ) ]
where f(E) = 1 / (exp(β*E) + 1)
"""
function compute_density_fermi_dirac(Ek, mu, beta)
    fermi_dirac(E) = E * beta > 100 ? 0.0 : 1.0 / (exp(E * beta) + 1.0)
    
    # Apply chemical potential shift
    Ek_shifted = Ek .- mu
    
    # Compute occupancy at each k-point
    occ = fermi_dirac.(Ek_shifted)
    
    # Average and multiply by 2 for spin
    n = 2.0 * mean(occ)
    
    return n
end


"""
Test the chemical potential solver against brute force Fermi-Dirac calculation
in the non-interacting limit (Σ = 0).
"""
function run_chemical_potential_tests()
    @testset "Chemical Potential Solver" begin
        # Test parameters
        Nk = 24
        beta = 0.05802 # Corresponds to T ~ 200K in meV^-1
        a = 3.82
        b = π/a
        wmax = 1000.0
        
        # Setup grid
        k_grid = collect(range(0, 2b, length=Nk+1)[1:Nk])
        KX = [kx for kx in k_grid, _ in k_grid]
        KY = [ky for _ in k_grid, ky in k_grid]
        
        # Simple dispersion: ε(k) = -2t(cos(k_x*a) + cos(k_y*a))
        t = 100.0  # meV
        Ek = -2t .* (cos.(KX .* a) .+ cos.(KY .* a))
        
        # Target density (slightly hole-doped)
        n_target = 0.918
        
        # Setup SparseIR
        b_f = FiniteTempBasis(Fermionic(), beta, wmax, 1e-15)
        smpl_wn_f = MatsubaraSampling(b_f)
        
        fermi_ns = [w.n for w in smpl_wn_f.sampling_points]
        wn_list = [n * π / beta for n in fermi_ns]
        n_freq = length(wn_list)
        
        # Non-interacting case: Σ = 0
        Sigma_iwn = zeros(ComplexF64, Nk, Nk, n_freq)
        
        # Solve for chemical potential
        _, mu_found, n_computed = solve_chemical_potential(
            Sigma_iwn, Ek, smpl_wn_f, Nk, beta, n_target;
            max_iter=30, tol=1e-6
        )
        
        # VERIFICATION 1: Brute force using Fermi-Dirac (exact for Σ=0)
        n_fermi_dirac = compute_density_fermi_dirac(Ek, mu_found, beta)

        println("\nChemical Potential Test:")
        println("  Target density: $n_target")
        println("  Found μ: $mu_found meV")
        println("  n from solve_chemical_potential: $n_computed")
        println("  n from Fermi-Dirac: $n_fermi_dirac")
        println("  Error (vs target): $(abs(n_computed - n_target))")
        println("  Error (vs Fermi-Dirac): $(abs(n_computed - n_fermi_dirac))")
        
        # Tests
        @test abs(n_computed - n_target) < 1e-4
        @test abs(n_fermi_dirac - n_target) < 1e-4
        
        # The two methods should agree
        @test abs(n_computed - n_fermi_dirac) < 1e-4
    end
end

"""
Analytic calculation of Chi in the non-interacting limit (Σ=0).
Performs the Matsubara sum analytically for:
χ(Q, iΩₘ) =  (1/Nk²) Σ_p |w(p)|² * [ n_F(E(p+Q/2)) + n_F(E(-p+Q/2)) - 1 ] 
                / [ iΩₘ - (E(p+Q/2) + E(-p+Q/2)) ]
where n_F(E) is the Fermi-Dirac distribution.
"""
function chi_exact_analytic(Ek_grid, wm_list, k_grid, Nk, a, beta)
    b = π/a
    n_bose = length(wm_list)
    Chi_exact = zeros(ComplexF64, Nk, Nk, n_bose)
    
    # Fermi function helper
    nf(E) = 1.0 / (exp(beta * E) + 1.0)
    
    # Outer loops: External Momentum P (Q/2)
    for iPx in 1:Nk, iPy in 1:Nk
        Px = k_grid[iPx]
        Py = k_grid[iPy]
        
        for idx_m in 1:n_bose
            wm = wm_list[idx_m]
            chi_sum = 0.0im
            
            # Inner loops: Internal Momentum p
            for ipx in 1:Nk, ipy in 1:Nk
                px = k_grid[ipx]
                py = k_grid[ipy]
                
                # |w(p)|² form factor
                w_sq_p = (cos(px*a) - cos(py*a))^2
                
                # ---------------------------------------------------------
                # MOMENTUM INDEXING (Identical to brute force)
                # ---------------------------------------------------------
                
                # p + Q/2 = p + P
                kp_x = mod(px + Px, 2b)
                kp_y = mod(py + Py, 2b)
                ikp_x = argmin(abs.(k_grid .- kp_x))
                ikp_y = argmin(abs.(k_grid .- kp_y))
                
                # -p + Q/2 = -p + P
                km_x = mod(-px + Px, 2b)
                km_y = mod(-py + Py, 2b)
                ikm_x = argmin(abs.(k_grid .- km_x))
                ikm_y = argmin(abs.(k_grid .- km_y))
                
                # Get energies from dispersion input
                E1 = Ek_grid[ikp_x, ikp_y]
                E2 = Ek_grid[ikm_x, ikm_y]
                
                # ---------------------------------------------------------
                # Matsubara Sum replaced by analytic formula
                # ---------------------------------------------------------
                
                numerator = nf(E1) + nf(E2) - 1
                denominator = im * wm - (E1 + E2)
                
                # Handle singularity if denominator is extremely small (optional safety)
                term = abs(denominator) < 1e-12 ? 0.0im : numerator / denominator
                
                chi_sum += w_sq_p * term
            end
            
            # Normalization:
            Chi_exact[iPx, iPy, idx_m] = chi_sum / (Nk^2)
        end
    end
    
    return Chi_exact
end


"""
Analytic validation of the Chi calculation in the Sigma=0 limit.
1. Constructs G(iwn) from non-interacting dispersion.
2. Runs Sparse IR Chi calculation.
3. Computes exact Chi using analytic Matsubara sum.
4. Compares results and reports errors.
"""
function run_analytic_chi_test()
    @testset "Analytic Validation of Chi Sparse IR Method (Sigma=0)" begin
        # 1. Test parameters (Typical meV scales)
        Nk = 16
        beta = 0.232  # T ~ 50 K
        a = 3.82
        b = π/a
        wmax = 2000.0   # Sufficient bandwidth coverage
        
        # Dispersion parameters
        t_hopping = 100.0 # meV
        mu = -50.0        # Chemical potential
        
        # Setup grid
        k_grid = collect(range(0, 2b, length=Nk+1)[1:Nk])
        
        # 2. Build Dispersion Matrix E(k)
        Ek_grid = zeros(Float64, Nk, Nk)
        for ix in 1:Nk, iy in 1:Nk
            kx, ky = k_grid[ix], k_grid[iy]
            Ek_grid[ix, iy] = -2*t_hopping * (cos(kx*a) + cos(ky*a)) - mu
        end
        
        # 3. Setup SparseIR
        basis_f = FiniteTempBasis(Fermionic(), beta, wmax, 1e-15)
        basis_b = FiniteTempBasis(Bosonic(), beta, wmax, 1e-15)
        smpl_wn_f = MatsubaraSampling(basis_f)
        smpl_wn_b = MatsubaraSampling(basis_b)
        
        wn_list = [w.n * π / beta for w in smpl_wn_f.sampling_points]
        wm_list = [w.n * π / beta for w in smpl_wn_b.sampling_points]
        
        # 4. Generate Input G_iwn for Sparse IR Method
        # G0(k, iwn) = 1 / (iwn - Ek)
        G_iwn = zeros(ComplexF64, Nk, Nk, length(wn_list))
        for i in 1:length(wn_list)
            wn = wn_list[i]
            for ix in 1:Nk, iy in 1:Nk
                G_iwn[ix, iy, i] = 1.0 / (im*wn - Ek_grid[ix, iy])
            end
        end
        
        # Precompute w_sq terms
        w_sq_terms = precompute_w_sq_terms(k_grid, Nk, a)
        
        # 5. Run Sparse IR Chi Calculation
        Chi_tau = compute_chi(G_iwn, w_sq_terms, Nk, smpl_wn_f, smpl_wn_b)

        # 6. Compute Exact Analytic Chi
        Chi_exact = chi_exact_analytic(Ek_grid, wm_list, k_grid, Nk, a, beta)
        
        # 7. Compare Results
        max_diff = maximum(abs.(Chi_tau .- Chi_exact))
        max_val_ref = maximum(abs.(Chi_exact))
        max_val_tau = maximum(abs.(Chi_tau))
        rel_err = max_diff / max_val_ref    
        
        println("\nAnalytic Validation Results:")
        println("  Max Value in Chi_exact: $max_val_ref")
        println("  Max Value in Chi_tau:  $max_val_tau")
        println("  Max Absolute Diff:  $max_diff")
        println("  Relative Error:     $rel_err")
        
        @test rel_err < 1e-5
        @test size(Chi_tau) == size(Chi_exact)
    end
end

"""
Analytic calculation of Sigma via direct Matsubara summation in the non-interacting limit with a bosonic mode.
Σ(k, iωₙ) = (1/Nk²) Σ_Q |w(k - Q/2)|² * W(Q, iΩₘ) * G(Q - k, iΩₘ - iωₙ)
where W(Q, iΩₘ) has poles at ±Ω_Q and the Matsubara sum is performed analytically using residues.
"""
function sigma_analytic_exact(Ek_grid, wn_list, k_grid, Nk, a, beta, V, Delta)
    n_fermi_pts = length(wn_list)
    Sigma_ref = zeros(ComplexF64, Nk, Nk, n_fermi_pts)

    # Stability-enhanced distribution functions
    n_f(x) = 1.0 / (exp(clamp(beta * x, -500, 500)) + 1.0)
    n_b(x) = 1.0 / (exp(clamp(beta * x, -500, 500)) - 1.0)

    for iKx in 1:Nk, iKy in 1:Nk
        Kx, Ky = k_grid[iKx], k_grid[iKy]
       
        for iPx in 1:Nk, iPy in 1:Nk
            Px, Py = k_grid[iPx], k_grid[iPy]
            
            # 1. Momentum-dependent energies
            # Q = 2P, so Qx = 2*Px, Qy = 2*Py
            Qx, Qy = 2*Px, 2*Py
            Omega_2P = 50.0 * (2.0 - cos(Qx*a/2) - cos(Qy*a/2)) + Delta
            
            # Internal Fermion momentum p = Q - k = 2P - k
            ipx = mod(2*(iPx-1) - (iKx-1), Nk) + 1
            ipy = mod(2*(iPy-1) - (iKy-1), Nk) + 1
            E_2Pk = Ek_grid[ipx, ipy]
            
            # 2. Form factor: w^2(k - Q/2) = w^2(k - P)
            w_sq = (cos((Kx - Px)*a) - cos((Ky - Py)*a))^2
            
            # 3. Analytic Residues for W(2P, iOm) * G(2P-k, iOm - iwn)
            # Poles of W are at iOm = ±Omega_2P
            nb = n_b(Omega_2P)
            nf = n_f(E_2Pk)
            

            for idx_n in 1:n_fermi_pts
                iwn = im * wn_list[idx_n]  # Imaginary Matsubara frequency
                
                # Two pole contributions from W(iΩₘ) spectral decomposition
                term1 = (nb + nf) / (iwn - Omega_2P + E_2Pk)
                term2 = (nb + 1 - nf) / (iwn + Omega_2P + E_2Pk)
                
                Sigma_ref[iKx, iKy, idx_n] +=  -1 * w_sq * (term1 + term2)
            end
        end
    end


    # Normalize by Nk^2 (the sum over P)
    return Sigma_ref / Nk^2
end

"""
Validation of Sigma via Direct Summation
1. Constructs G(iwn) from non-interacting dispersion.
2. Constructs W(iwm) from synthetic bosonic mode.
3. Runs Sparse IR Sigma calculation.
4. Computes exact Sigma using analytic residue method.
5. Compares results and reports errors.
"""
function run_sigma_direct_sum_test()
    @testset "Validation of Sigma via Direct Summation" begin
        # 1. Parameters (Small Nk for O(N^4) brute force speed)
        Nk = 8
        beta = 0.232 # T ~ 50 K
        a = 3.82
        b = π/a
        wmax = 5000.0   # Sufficient bandwidth coverage
        V = -400.0
        
        # Dispersion parameters
        t_hopping = 100.0 # meV
        mu = -50.0        # Chemical potential

        # Setup grid
        k_grid = collect(range(0, 2b, length=Nk+1)[1:Nk])
        
        # 2. Build Dispersion Matrix E(k)
        Ek_grid = zeros(Float64, Nk, Nk)
        for ix in 1:Nk, iy in 1:Nk
            kx, ky = k_grid[ix], k_grid[iy]
            Ek_grid[ix, iy] = -2*t_hopping * (cos(kx*a) + cos(ky*a)) - mu
        end
        
        # 3. Setup SparseIR
        basis_f = FiniteTempBasis(Fermionic(), beta, wmax, 1e-15)
        basis_b = FiniteTempBasis(Bosonic(), beta, wmax, 1e-15)
        smpl_wn_f = MatsubaraSampling(basis_f)
        smpl_wn_b = MatsubaraSampling(basis_b)
        
        wn_list = [w.n * π / beta for w in smpl_wn_f.sampling_points]
        wm_list = [w.n * π / beta for w in smpl_wn_b.sampling_points]
        
        # 4. Generate Input G_iwn for Sparse IR Method
        # G0(k, iwn) = 1 / (iwn - Ek)
        G_iwn = zeros(ComplexF64, Nk, Nk, length(wn_list))
        for i in 1:length(wn_list)
            wn = wn_list[i]
            for ix in 1:Nk, iy in 1:Nk
                G_iwn[ix, iy, i] = 1.0 / (im*wn - Ek_grid[ix, iy])
            end
        end
        
        # 5. Generate Input Synthetic W for Validation
        # W(2P, iOm) = 1/((iOm)^2 - Omega^2) symmetric form
        Delta = 100.0
        W_iwm = zeros(ComplexF64, Nk, Nk, length(wm_list))
        for i in 1:length(wm_list)
            wm =  wm_list[i]
            for ix in 1:Nk, iy in 1:Nk
                Qx, Qy = 2*k_grid[ix], 2*k_grid[iy]
                Omega_q = 50.0 * (2.0 - cos(Qx*a/2) - cos(Qy*a/2)) + Delta
                # Symmetric Bosonic Propagator
                W_iwm[ix, iy, i] = -(2 * Omega_q) / ((im*wm)^2 - Omega_q^2)
            end
        end
        
        # Precompute w_sq terms
        w_sq_terms = precompute_w_sq_terms(k_grid, Nk, a)
        
        # 5. Run Sparse IR Chi Calculation
        Sigma_tau = compute_sigma(G_iwn, W_iwm, 0, w_sq_terms, Nk, smpl_wn_f, smpl_wn_b) # synthetic mode has no static V.

        # 6. Compute Exact Analytic Chi
        Sigma_ref = sigma_analytic_exact(Ek_grid, wn_list, k_grid, Nk, a, beta, V, Delta)
        
        # 7. Compare Results
        max_diff = maximum(abs.(Sigma_tau .- Sigma_ref))
        max_val = maximum(abs.(Sigma_ref))
        rel_err = max_diff / max_val
        
        sum_ref = sum(Sigma_ref)
        sum_tau = sum(Sigma_tau)
        sum_rel_err = abs(sum_ref - sum_tau) / abs(sum_ref)

        println("\nSigma Validation Results:")
        println(" Brute Force sum : $(sum_ref)")
        println(" Sparse IR sum  : $(sum_tau)")
        println("  Relative Sum Error: $(sum_rel_err)")
        println("  Max Absolute Diff:  $max_diff")
        println("  Relative Error:     $rel_err")
        
        @test rel_err < 1e-4
        @test size(Sigma_tau) == size(Sigma_ref)
    end
end


# Run all tests
function runtests()
    run_w_sq_tests()
    run_chemical_potential_tests()
    run_analytic_chi_test()
    run_sigma_direct_sum_test()
end

runtests();