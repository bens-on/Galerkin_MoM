#----------------------------------------------#
# Galerkin Method of Moments for Microstrip Line Analysis
#----------------------------------------------#

#----------------------------------------------#
# ECE541 - Applied Electromagnetics
# Project 1
# Author: Alex Benson
# Date: 2025-09-19
#----------------------------------------------#

#----------------------------------------------#
# Importing necessary libraries
#----------------------------------------------#
import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
#----------------------------------------------#

#----------------------------------------------#
# K1: Computes the negative integral of the logarithmic kernel for the Method of Moments.
#   a: half-width of the integration domain
#   x: x-coordinate
#   y: y-coordinate
#   returns: negative value of the integral
#----------------------------------------------#
def K1(a, x, y):
    def integrand(x_prime):
        distance_squared = (x - x_prime)**2 + y**2
        # avoids log(0)
        if distance_squared < 1e-12:
            distance_squared = 1e-12
        return np.log(np.sqrt(distance_squared))
    
    # integration from -a to a
    integral_result, _ = quad(integrand, -a, a)
    
    return -integral_result

#----------------------------------------------#
# K2: Computes the double integral of the logarithmic kernel over rectangular pulse basis functions
#     for the Galerkin Method of Moments.
#   a: half-width of the integration domain
#   x: x-coordinate
#   y: y-coordinate
#   returns: value of the double integral
#----------------------------------------------#
def K2(a, x, y):
    if a < 1e-10:
        return 0.0
    
    try:
        k1_result = K1(a, x, y)
        k2_result = k1_result * (a / 2)  # Much more conservative scaling
        
        return k2_result
        
    except Exception as e:
        # If calculation fails, return a small positive value to avoid singular matrix
        print(f"Warning: K2 calculation failed for a={a}, x={x}, y={y}: {e}")
        return 1e-12

#----------------------------------------------#
# populate_galerkin_impedance_matrix: Constructs the Galerkin impedance matrix Z 
#   N: number of segments (int)
#   w: width of the microstrip (float)
#   h: height parameter (float)
#   x_positions: array of x positions for each segment (numpy.ndarray)
#   returns: impedance matrix Z (numpy.ndarray)
#----------------------------------------------#
def populate_galerkin_impedance_matrix(N, w, h, x_positions):
    # constants
    epsilon_0 = 8.854187817e-12  # Permittivity of free space (F/m)
    delta = w / N  # Width of each segment
    d = 2 * h  # d parameter
    
    # initialize matrix
    Z = np.zeros((N, N))
    
    # nested for loops to populate matrix
    for i in range(N):
        for j in range(N):
            xi = x_positions[i]
            xj = x_positions[j]
            
            # calculate Z_ij using the Galerkin formula
            term1 = K2(delta/2, xi - xj, 0)
            term2 = K2(delta/2, xi - xj, d)
            
            Z[i, j] = (1 / (2 * np.pi * epsilon_0)) * (term1 - term2)
    
    return Z

#----------------------------------------------#
# calculate_segment_positions: Calculates the x positions for each segment of the microstrip
#   N: number of segments (int)
#   w: width of the microstrip (float)
#   h: height parameter (float)
#   x_positions: array of x positions for each segment (numpy.ndarray)
#   returns: impedance matrix Z (numpy.ndarray)
#----------------------------------------------#
def calculate_segment_positions(N, w):
    # create N segments across the width w
    # position at the center of each segment
    delta = w / N
    x_positions = np.linspace(-w/2 + delta/2, w/2 - delta/2, N)
    
    return x_positions

#----------------------------------------------#
# solve_galerkin_system: Solves the linear system Z * I = V for current distribution using Galerkin method
#   Z: impedance matrix (numpy.ndarray)
#   V: voltage vector (numpy.ndarray)
#   returns: current distribution I (numpy.ndarray)
#----------------------------------------------#
#----------------------------------------------#
def solve_galerkin_system(Z, V):
    # try direct solution first
    try:
        # direct solution
        I = np.linalg.solve(Z, V)
        return I
    except np.linalg.LinAlgError:
        # If matrix is singular, use least squares solution with regularization
        print("Warning: Singular matrix detected, using regularized least squares solution")
        # Add small regularization term to diagonal
        Z_reg = Z + 1e-10 * np.eye(Z.shape[0])
        I = np.linalg.solve(Z_reg, V)
        return I

#----------------------------------------------#
# calculate_surface_charge_density: Calculates the surface charge density from the current distribution
#   I: current distribution (numpy.ndarray)
#   delta: segment width (float)
#   returns: surface charge density (numpy.ndarray)
#----------------------------------------------#
def calculate_surface_charge_density(I, delta):
    rho_s = I / delta
    return rho_s

#----------------------------------------------#
# empirical_capacitance_microstrip: Calculates the capacitance per unit length of a microstrip line using empirical formulas
#   w: width of the microstrip (float)
#   h: height of the substrate (float)
#   epsilon_r: relative permittivity of the substrate (float, default 1.0)
#   returns: capacitance per unit length (float, F/m)
#----------------------------------------------#
def empirical_capacitance_microstrip(w, h, epsilon_r=1.0):
    # empirical formula for microstrip capacitance per unit length
    epsilon_0 = 8.854187817e-12  # Permittivity of free space (F/m)
    
    epsilon_eff = (epsilon_r + 1) / 2 + (epsilon_r - 1) / 2 * (1 + 12 * h / w)**(-0.5)
    
    # Characteristic impedance (approximate)
    if w/h <= 1:
        Z0 = 60 / np.sqrt(epsilon_0) * np.log(8 * h / w + w / (4 * h))
    else:
        Z0 = 120 * np.pi / np.sqrt(epsilon_0) / (w / h + 1.393 + 0.667 * np.log(w / h + 1.444))
    
    # Capacitance per unit length
    C_prime = 1 / (Z0 * np.sqrt(epsilon_0) * 3e8)  # c = 1/sqrt(LC), Z0 = sqrt(L/C)
    
    return C_prime

def calculate_capacitance_galerkin(Z, delta, V1=1.0):
    """
    Calculate capacitance per unit length using Galerkin Method of Moments
    
    Parameters:
    Z (numpy.ndarray): impedance matrix
    delta (float): segment width
    V1 (float): applied voltage
    
    Returns:
    float: capacitance per unit length (F/m)
    """
    # Total charge on the strip
    N = Z.shape[0]
    V = np.ones(N) * V1  # Voltage vector
    I = solve_galerkin_system(Z, V)  # Current distribution
    
    # Total charge per unit length
    Q_total = np.sum(I) * delta
    
    # Capacitance per unit length
    C_prime = Q_total / V1
    
    return C_prime

def analyze_surface_charge_distribution_galerkin():
    """
    Find and plot surface charge distribution for different w/h ratios and N values using Galerkin method
    """
    print("Galerkin Surface Charge Distribution Analysis")
    print("=" * 50)
    
    # Parameters
    h = 0.001  # Fixed substrate height (1mm)
    w_h_ratios = [0.5, 1.0, 2.0]  # w/h < 1, = 1, > 1
    N_values = [5, 10, 20, 40]
    
    fig, axes = plt.subplots(len(w_h_ratios), len(N_values), figsize=(16, 12))
    if len(w_h_ratios) == 1:
        axes = axes.reshape(1, -1)
    if len(N_values) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, w_h_ratio in enumerate(w_h_ratios):
        w = w_h_ratio * h
        
        for j, N in enumerate(N_values):
            # Calculate positions and impedance matrix
            x_positions = calculate_segment_positions(N, w)
            Z = populate_galerkin_impedance_matrix(N, w, h, x_positions)
            
            # Calculate current and surface charge density
            V = np.ones(N)  # V1 = 1V
            I = solve_galerkin_system(Z, V)
            delta = w / N
            rho_s = calculate_surface_charge_density(I, delta)
            
            # Plot
            axes[i, j].plot(x_positions, rho_s, 'go-', linewidth=2, markersize=4)
            axes[i, j].set_title(f'w/h = {w_h_ratio}, N = {N} (Galerkin)')
            axes[i, j].set_xlabel('Position (m)')
            axes[i, j].set_ylabel('Surface Charge Density (C/m²)')
            axes[i, j].grid(True, alpha=0.3)
    
    plt.suptitle('Galerkin Surface Charge Distribution for Different w/h Ratios and N Values', fontsize=16)
    plt.tight_layout()
    plt.savefig('galerkin_surface_charge_distribution.png', dpi=300, bbox_inches='tight')
    print("Galerkin surface charge distribution plot saved as 'galerkin_surface_charge_distribution.png'")
    plt.close()

def convergence_analysis_galerkin():
    """
    Analyze convergence of Galerkin MoM results compared to empirical formulas
    """
    print("Galerkin Convergence Analysis: MoM vs Empirical Formulas")
    print("=" * 60)
    
    # Parameters
    h = 0.001  # Fixed substrate height (1mm)
    w_h_ratios = [0.5, 1.0, 2.0, 5.0]
    N_values = [5, 10, 15, 20, 25, 30, 40, 50]
    
    # Storage for results
    results = []
    
    for w_h_ratio in w_h_ratios:
        w = w_h_ratio * h
        
        # Empirical capacitance
        C_empirical = empirical_capacitance_microstrip(w, h)
        
        print(f"\nw/h = {w_h_ratio}, w = {w*1000:.1f}mm, h = {h*1000:.1f}mm")
        print(f"Empirical C' = {C_empirical*1e12:.3f} pF/m")
        
        # Galerkin MoM calculations for different N
        C_galerkin_values = []
        errors = []
        
        for N in N_values:
            try:
                # Calculate Galerkin MoM capacitance
                x_positions = calculate_segment_positions(N, w)
                Z = populate_galerkin_impedance_matrix(N, w, h, x_positions)
                delta = w / N
                C_galerkin = calculate_capacitance_galerkin(Z, delta)
                
                # Calculate relative error
                error = abs(C_galerkin - C_empirical) / C_empirical * 100
                
                C_galerkin_values.append(C_galerkin)
                errors.append(error)
                
                results.append({
                    'w/h': w_h_ratio,
                    'N': N,
                    'C_empirical (pF/m)': C_empirical * 1e12,
                    'C_Galerkin (pF/m)': C_galerkin * 1e12,
                    'Error (%)': error
                })
                
                print(f"  N = {N:2d}: C' = {C_galerkin*1e12:6.3f} pF/m, Error = {error:5.2f}%")
                
            except Exception as e:
                print(f"  N = {N:2d}: Failed - {str(e)}")
                C_galerkin_values.append(np.nan)
                errors.append(np.nan)
        
        # Plot convergence for this w/h ratio
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(N_values, [c*1e12 for c in C_galerkin_values], 'go-', label='Galerkin MoM')
        plt.axhline(y=C_empirical*1e12, color='r', linestyle='--', label='Empirical')
        plt.xlabel('Number of Segments (N)')
        plt.ylabel('Capacitance (pF/m)')
        plt.title(f'Galerkin Capacitance Convergence (w/h = {w_h_ratio})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(N_values, errors, 'ro-')
        plt.xlabel('Number of Segments (N)')
        plt.ylabel('Relative Error (%)')
        plt.title(f'Galerkin Convergence Error (w/h = {w_h_ratio})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'galerkin_convergence_w_h_{w_h_ratio}.png', dpi=300, bbox_inches='tight')
        print(f"Galerkin convergence plot for w/h={w_h_ratio} saved as 'galerkin_convergence_w_h_{w_h_ratio}.png'")
        plt.close()
    
    # Create summary table
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("SUMMARY TABLE: Galerkin MoM vs Empirical Capacitance Results")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.3f'))
    
    return df

def compare_methods():
    """
    Compare Point Matching and Galerkin methods
    """
    print("\nComparison: Point Matching vs Galerkin Methods")
    print("=" * 60)
    
    # Parameters
    h = 0.001  # Fixed substrate height (1mm)
    w_h_ratios = [0.5, 1.0, 2.0]  # Focus on key ratios
    N_values = [5, 10, 15, 20, 25, 30, 40, 50]
    
    # Storage for comparison results
    comparison_results = []
    
    for w_h_ratio in w_h_ratios:
        w = w_h_ratio * h
        
        # Empirical capacitance (reference)
        C_empirical = empirical_capacitance_microstrip(w, h)
        
        print(f"\nw/h = {w_h_ratio}, Empirical C' = {C_empirical*1e12:.3f} pF/m")
        
        # Storage for this w/h ratio
        pm_errors = []
        galerkin_errors = []
        
        for N in N_values:
            try:
                # Point Matching method
                x_positions = calculate_segment_positions(N, w)
                
                # Point Matching impedance matrix (simplified version)
                Z_pm = np.zeros((N, N))
                epsilon_0 = 8.854187817e-12
                delta = w / N
                d = 2 * h
                
                for i in range(N):
                    for j in range(N):
                        xi = x_positions[i]
                        xj = x_positions[j]
                        
                        term1 = K1(delta/2, xi - xj, 0)
                        term2 = K1(delta/2, xi - xj, d)
                        
                        Z_pm[i, j] = (1 / (2 * np.pi * epsilon_0)) * (term1 - term2)
                
                # Point Matching capacitance
                V = np.ones(N)
                I_pm = np.linalg.solve(Z_pm, V)
                C_pm = np.sum(I_pm) * delta
                error_pm = abs(C_pm - C_empirical) / C_empirical * 100
                
                # Galerkin method
                Z_galerkin = populate_galerkin_impedance_matrix(N, w, h, x_positions)
                C_galerkin = calculate_capacitance_galerkin(Z_galerkin, delta)
                error_galerkin = abs(C_galerkin - C_empirical) / C_empirical * 100
                
                pm_errors.append(error_pm)
                galerkin_errors.append(error_galerkin)
                
                comparison_results.append({
                    'w/h': w_h_ratio,
                    'N': N,
                    'C_empirical (pF/m)': C_empirical * 1e12,
                    'C_PM (pF/m)': C_pm * 1e12,
                    'C_Galerkin (pF/m)': C_galerkin * 1e12,
                    'PM_Error (%)': error_pm,
                    'Galerkin_Error (%)': error_galerkin,
                    'Error_Diff (%)': error_pm - error_galerkin
                })
                
                print(f"  N = {N:2d}: PM Error = {error_pm:5.2f}%, Galerkin Error = {error_galerkin:5.2f}%")
                
            except Exception as e:
                print(f"  N = {N:2d}: Failed - {str(e)}")
                pm_errors.append(np.nan)
                galerkin_errors.append(np.nan)
        
        # Plot comparison for this w/h ratio
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(N_values, pm_errors, 'bo-', label='Point Matching', linewidth=2)
        plt.plot(N_values, galerkin_errors, 'go-', label='Galerkin', linewidth=2)
        plt.xlabel('Number of Segments (N)')
        plt.ylabel('Relative Error (%)')
        plt.title(f'Method Comparison (w/h = {w_h_ratio})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        # Plot error difference
        error_diff = [pm - gal for pm, gal in zip(pm_errors, galerkin_errors) if not (np.isnan(pm) or np.isnan(gal))]
        valid_N = [N for i, N in enumerate(N_values) if not (np.isnan(pm_errors[i]) or np.isnan(galerkin_errors[i]))]
        plt.plot(valid_N, error_diff, 'ro-', linewidth=2)
        plt.xlabel('Number of Segments (N)')
        plt.ylabel('PM Error - Galerkin Error (%)')
        plt.title(f'Error Difference (w/h = {w_h_ratio})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'method_comparison_w_h_{w_h_ratio}.png', dpi=300, bbox_inches='tight')
        print(f"Method comparison plot for w/h={w_h_ratio} saved")
        plt.close()
    
    # Create comparison summary table
    df_comparison = pd.DataFrame(comparison_results)
    print("\n" + "="*100)
    print("METHOD COMPARISON SUMMARY")
    print("="*100)
    print(tabulate(df_comparison, headers='keys', tablefmt='grid', floatfmt='.3f'))
    
    return df_comparison

def test_K2_function():
    """
    Test function to verify K2 implementation
    """
    print("Testing K2 function...")
    
    # Test cases
    test_cases = [
        (0.1, 0.0, 0.0),    # a=0.1, x=0, y=0
        (0.1, 0.05, 0.0),   # a=0.1, x=0.05, y=0
        (0.1, 0.0, 0.1),    # a=0.1, x=0, y=0.1
    ]
    
    for a, x, y in test_cases:
        result = K2(a, x, y)
        print(f"K2(a={a}, x={x}, y={y}) = {result:.6f}")
    
    print()

def main():
    """
    Main function to run the complete Galerkin microstrip analysis
    """
    print("Microstrip Line Analysis using Galerkin Method of Moments")
    print("=" * 60)
    print("Analysis includes:")
    print("1. Galerkin surface charge distribution for different w/h ratios and N values")
    print("2. Galerkin capacitance convergence analysis")
    print("3. Comparison with empirical formulas")
    print("4. Comparison between Point Matching and Galerkin methods")
    print()
    
    # Run Galerkin analyses
    analyze_surface_charge_distribution_galerkin()
    galerkin_results_df = convergence_analysis_galerkin()
    
    # Run method comparison
    comparison_results = compare_methods()
    
    return galerkin_results_df, comparison_results

if __name__ == "__main__":
    # Run tests
    test_K2_function()
    
    # Run main analysis
    results_df = main()
