# Galerkin Method of Moments (Galerkin_MoM)

## Overview
This repository contains the implementation of the Galerkin Method of Moments for microstrip line analysis. The Galerkin method uses rectangular pulse basis functions with Galerkin testing to solve electromagnetic problems.

## Features
- **Galerkin Implementation**: Uses Galerkin testing with rectangular pulse basis functions
- **Convergence Analysis**: Detailed analysis of convergence with increasing N values
- **Empirical Comparison**: Comparison with empirical formulas from Notaros textbook
- **Method Comparison**: Direct comparison with Point Matching method
- **Surface Charge Distribution**: Visualization of charge distribution patterns

## Files
- `Galerkin_MoM.py`: Main implementation of the Galerkin method
- `requirements.txt`: Python dependencies
- `README.md`: This documentation file

## Results Summary

### Performance
- **Large errors** (382,681% - 40,047,841%) indicating implementation challenges
- **Poor convergence behavior** compared to Point Matching
- **Implementation issues** with K2 function approximation

### Key Features
1. **Surface Charge Distribution Analysis** - Plots for different w/h ratios and N values
2. **Convergence Analysis** - Detailed comparison with empirical formulas
3. **Method Comparison** - Direct comparison with Point Matching method
4. **Comprehensive Analysis** - Full analysis pipeline

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis
python Galerkin_MoM.py
```

## Dependencies
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- tabulate >= 0.9.0

## Generated Outputs
- Surface charge distribution plots
- Convergence analysis plots
- Method comparison plots
- Comprehensive data tables

## Method Details

### Galerkin Method
The Galerkin method uses:
- **Basis Functions**: Rectangular pulses
- **Testing**: Galerkin method (double integration)
- **Impedance Matrix**: Z_ij = -(1/(2π*ε₀)) * [K2(δ/2, xi-xj, 0) - K2(δ/2, xi-xj, d)]

Where:
- K2 is the double integral over the logarithmic kernel
- δ is the segment width
- xi, xj are segment positions
- d = 2h is the distance parameter

## Implementation Challenges

### K2 Function Issues
The current implementation uses a simplified approximation:
```python
k2_result = k1_result * (a / 2)  # Simplified scaling
```

This approximation is not mathematically correct and leads to:
- Extremely large errors (millions of percent)
- Poor convergence behavior
- Unrealistic capacitance values

### Recommendations
1. **Implement Proper K2 Function**: The double integral should be computed analytically
2. **Verify Mathematical Formulation**: Ensure the Galerkin impedance matrix formulation is correct
3. **Compare with Literature**: Validate against known analytical solutions

## Comparison with Point Matching

| Method | Typical Error Range | Convergence Behavior |
|--------|-------------------|-------------------|
| Point Matching | 0.03% - 3.23% | Excellent convergence |
| Galerkin | 382,681% - 40,047,841% | Poor convergence |

## Results
The Galerkin method implementation reveals significant challenges with the K2 function approximation, demonstrating the importance of proper mathematical formulation in numerical methods.

## Author
Alex Benson - ECE541 Applied Electromagnetics Project 1
