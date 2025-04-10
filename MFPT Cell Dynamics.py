#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from scipy.stats import gaussian_kde, lognorm
from scipy.optimize import curve_fit
from scipy.special import erfc, gamma


# ====================== CORE MODEL WITH EXTENSIONS ======================

class AgingModel:
    def __init__(self):
        # Base parameters
        self.Xc_mean = 10
        self.μ0_mean = 0.1
        self.α_mean = 0.08
        self.σ_base = 0.3
        
        # Network parameters
        self.cell_network = nx.watts_strogatz_graph(100, 4, 0.1)  # 100 cells, small-world
        self.k = 0.05  # Coupling strength
        
        # Memory parameters
        self.τ = 5  # Delay time for repair
        self.β = 0.1  # Repair efficiency
        self.memory_alpha = 0.7  # Fractional derivative order

    def fractional_derivative(self, X_hist, alpha): 
    """Stable Grünwald-Letnikov fractional derivative with error handling"""
    n = len(X_hist)
    frac_deriv = np.zeros(n)
    
    # Precompute gamma(alpha+1) once since it's constant
    gamma_alpha_plus_1 = gamma(alpha + 1)
    
    for t in range(n):
        total = 0.0
        for k in range(min(t + 1, 20)):  # Limit terms for stability
            try:
                # Calculate denominator components separately
                gamma_k_plus_1 = gamma(k + 1)
                gamma_alpha_minus_k_plus_1 = gamma(alpha - k + 1)
                
                # Skip if denominator components are invalid
                if not (np.isfinite(gamma_k_plus_1) and np.isfinite(gamma_alpha_minus_k_plus_1)):
                    continue
                    
                denominator = gamma_k_plus_1 * gamma_alpha_minus_k_plus_1
                
                # Skip division by zero or near-zero
                if denominator < 1e-10:  # Small threshold to prevent overflow
                    continue
                    
                coefficient = ((-1)**k) * gamma_alpha_plus_1 / denominator
                x_value = X_hist[t - k] if (t - k) >= 0 else 0.0
                term = coefficient * x_value
                
                # Only add finite terms to the total
                if np.isfinite(term):
                    total += term
                    
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
                
        frac_deriv[t] = total
        
    return frac_deriv[-1]  # Return only the most recent value
    def simulate_individual(self, intervention=None, memory=False, coupled=False):
        """Simulate single individual with optional extensions"""
        # Initialize parameters
        Xc = lognorm.rvs(s=0.3, scale=self.Xc_mean)
        μ0 = lognorm.rvs(s=0.2, scale=self.μ0_mean)
        α = lognorm.rvs(s=0.1, scale=self.α_mean)
        σ = self.σ_base
        
        # Apply interventions
        if intervention == "caloric_restriction":
            σ *= 0.5
        elif intervention == "repair_boost":
            Xc *= 1.5
        
        # Initialize state
        if coupled:
            X = {cell: 0 for cell in self.cell_network.nodes}
        else:
            X = 0
            
        X_hist = []  # For memory effects
        t = 0
        
        while t < 100:
            μ = μ0 * np.exp(α * t)
            
            if coupled:
                # Coupled network dynamics
                new_X = {}
                for cell in self.cell_network.nodes:
                    neighbors = list(self.cell_network.neighbors(cell))
                    coupling = self.k * sum(X[n] - X[cell] for n in neighbors)
                    dW = np.random.normal(0, np.sqrt(0.1))
                    new_X[cell] = X[cell] + (μ + coupling)*0.1 + σ*dW
                
                # Failure condition (any cell reaches threshold)
                if any(x >= Xc for x in new_X.values()):
                    return t
                X = new_X
                
            else:
                # Single-cell dynamics
                dW = np.random.normal(0, np.sqrt(0.1))
                
                if memory:
                    # Delayed repair and fractional memory
                    X_hist.append(X)
                    if len(X_hist) > self.τ:
                        repair = self.β * X_hist[-self.τ]
                        memory_effect = 0.1 * self.fractional_derivative(X_hist, self.memory_alpha)
                    else:
                        repair = 0
                        memory_effect = 0
                    X += (μ - repair + memory_effect)*0.1 + σ*dW
                else:
                    X += μ*0.1 + σ*dW
                
                if X >= Xc:
                    return t
            
            t += 0.1
        return t  # Survived to max age

# ====================== SYNTHETIC DATA GENERATION ======================

def generate_synthetic_data():
    """Generate realistic synthetic mortality data"""
    ages = np.arange(0, 100)
    # Gompertz-like mortality
    hazard = 1e-5 * np.exp(0.1 * ages)
    # Add some noise and late-life plateau
    hazard += np.random.normal(0, 1e-6, size=len(ages))
    hazard[80:] = hazard[80] * (1 - 0.02*(ages[80:] - 80))  # Plateau effect
    return ages, hazard

# ====================== ANALYSIS TOOLS ======================

def calculate_mortality(lifetimes, max_age=100, bw=0.7):
    """Calculate mortality rate from simulated lifetimes"""
    if not lifetimes:
        return np.linspace(0, max_age, 100), np.zeros(100)
    
    age_grid = np.linspace(0, max_age, 100)
    kde = gaussian_kde(lifetimes, bw_method=bw)
    pdf = kde(age_grid)
    survival = 1 - np.cumsum(pdf) * (age_grid[1] - age_grid[0])
    hazard = pdf / (survival + 1e-10)
    return age_grid, hazard

# ====================== VISUALIZATION ======================

def plot_results(model):
    """Run simulations and plot results"""
    plt.figure(figsize=(14, 6))
    
    # Basic single-cell model
    plt.subplot(121)
    lifetimes = [model.simulate_individual() for _ in range(1000)]
    age_grid, hazard = calculate_mortality(lifetimes)
    plt.plot(age_grid, hazard, 'k-', label='Basic model')
    
    # With interventions
    lifetimes_cr = [model.simulate_individual(intervention="caloric_restriction") for _ in range(500)]
    age_grid_cr, hazard_cr = calculate_mortality(lifetimes_cr)
    plt.plot(age_grid_cr, hazard_cr, 'b-', label='Caloric restriction')
    
    lifetimes_repair = [model.simulate_individual(intervention="repair_boost") for _ in range(500)]
    age_grid_repair, hazard_repair = calculate_mortality(lifetimes_repair)
    plt.plot(age_grid_repair, hazard_repair, 'g-', label='Repair boost')
    
    plt.yscale('log')
    plt.xlabel('Age'); plt.ylabel('Mortality Rate')
    plt.legend(); plt.grid(True)
    plt.title('Basic Model + Interventions')
    
    # Network and memory effects
    plt.subplot(122)
    lifetimes_net = [model.simulate_individual(coupled=True) for _ in range(500)]
    age_grid_net, hazard_net = calculate_mortality(lifetimes_net)
    plt.plot(age_grid_net, hazard_net, 'r-', label='Coupled cells')
    
    lifetimes_mem = [model.simulate_individual(memory=True) for _ in range(500)]
    age_grid_mem, hazard_mem = calculate_mortality(lifetimes_mem)
    plt.plot(age_grid_mem, hazard_mem, 'm-', label='Memory effects')
    
    plt.yscale('log')
    plt.xlabel('Age'); plt.ylabel('Mortality Rate')
    plt.legend(); plt.grid(True)
    plt.title('Network and Memory Extensions')
    
    plt.tight_layout()
    plt.savefig("Network and Memory Extensions.png")

# ====================== MAIN EXECUTION ======================

if __name__ == "__main__":
    print("Running probabilistic aging model with extensions...")
    
    # Initialize model
    model = AgingModel()
    
    # Run and plot simulations
    plot_results(model)
    
    # Show synthetic data vs model
    ages, synthetic_hazard = generate_synthetic_data()
    plt.figure(figsize=(7, 5))
    plt.plot(ages, synthetic_hazard, 'ko', markersize=3, label='Synthetic Data')
    
    # Plot model fit
    lifetimes = [model.simulate_individual() for _ in range(1000)]
    age_grid, hazard = calculate_mortality(lifetimes)
    plt.plot(age_grid, hazard, 'r-', label='Model')
    
    plt.yscale('log')
    plt.xlabel('Age'); plt.ylabel('Mortality Rate')
    plt.legend(); plt.grid(True)
    plt.title('Model vs Synthetic Data')
    plt.savefig("Model vs Synthetic Data.png")

