module PseudoGapSolver

using SparseIR, FFTW, LinearAlgebra, Statistics

# Include the computational routines from the logic file
include("FFT_solver.jl")

# Export the public API for easy access in research scripts
export precompute_w_sq_terms, 
       build_convolver, 
       compute_chi, 
       compute_sigma, 
       solve_chemical_potential

end # module
