import numpy as np
from .sim import Sim


def final_charts(sim: Sim):
    """Generate and save individual analysis charts for the simulation."""
    try:
        import matplotlib.pyplot as plt
        import os
        
        cfg = sim.cfg
        
        # Check if we have any data to plot
        if not sim.day_population:
            print("No simulation data to plot")
            return
        
        # Create output directory
        output_dir = "simulation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        days = np.arange(len(sim.day_population))
        
        # 1. Population over time
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(days, sim.day_population, lw=3, color='#1f77b4')
        plt.title(f"Population Evolution - {len(sim.day_population)} Days", fontsize=16, fontweight='bold')
        plt.xlabel("Day", fontsize=12)
        plt.ylabel("Number of Agents", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/population_evolution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Fights per day
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(days, sim.day_fights, lw=3, color="#d62728")
        plt.title(f"Combat Frequency - {len(sim.day_population)} Days", fontsize=16, fontweight='bold')
        plt.xlabel("Day", fontsize=12)
        plt.ylabel("Number of Fights", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/combat_frequency.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Mean energy per day
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(days, sim.day_mean_energy, lw=3, color="#2ca02c")
        plt.title(f"Average Energy Levels - {len(sim.day_population)} Days", fontsize=16, fontweight='bold')
        plt.xlabel("Day", fontsize=12)
        plt.ylabel("Mean Energy", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/energy_evolution.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Individual trait evolution plots
        trait_names = {
            "max_energy": "Maximum Energy",
            "metabolism": "Metabolism Rate", 
            "move_cost": "Movement Cost",
            "perception": "Perception Range",
            "strength": "Combat Strength",
            "aggression": "Aggression Level",
            "repro_threshold": "Reproduction Threshold",
            "repro_cost": "Reproduction Cost"
        }
        
        for trait_key, trait_name in trait_names.items():
            if trait_key in sim.trait_series and sim.trait_series[trait_key]:
                plt.figure(figsize=(10, 6), dpi=150)
                
                # Calculate mean for each day
                trait_means = []
                for day_data in sim.trait_series[trait_key]:
                    if day_data.size > 0:
                        trait_means.append(np.mean(day_data))
                    else:
                        trait_means.append(np.nan)
                
                # Plot trait evolution
                plt.plot(days, trait_means, lw=3, color='#9467bd', marker='o', markersize=4)
                plt.title(f"{trait_name} Evolution - {len(sim.day_population)} Days", fontsize=16, fontweight='bold')
                plt.xlabel("Day", fontsize=12)
                plt.ylabel(trait_name, fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save individual trait plot
                filename = f"{output_dir}/{trait_key}_evolution.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
        
        # 5. Summary statistics plot
        plt.figure(figsize=(12, 8), dpi=150)
        
        # Create subplots for key metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
        
        # Population
        ax1.plot(days, sim.day_population, lw=2, color='#1f77b4')
        ax1.set_title("Population", fontweight='bold')
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Agents")
        ax1.grid(True, alpha=0.3)
        
        # Fights
        ax2.plot(days, sim.day_fights, lw=2, color="#d62728")
        ax2.set_title("Daily Fights", fontweight='bold')
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Fights")
        ax2.grid(True, alpha=0.3)
        
        # Energy
        ax3.plot(days, sim.day_mean_energy, lw=2, color="#2ca02c")
        ax3.set_title("Mean Energy", fontweight='bold')
        ax3.set_xlabel("Day")
        ax3.set_ylabel("Energy")
        ax3.grid(True, alpha=0.3)
        
        # Final trait distribution (last day)
        if sim.agents and sim.trait_series["strength"]:
            last_strength = sim.trait_series["strength"][-1]
            if last_strength.size > 0:
                ax4.hist(last_strength, bins=20, alpha=0.7, color='#ff7f0e', edgecolor='black')
                ax4.set_title("Final Strength Distribution", fontweight='bold')
                ax4.set_xlabel("Strength")
                ax4.set_ylabel("Frequency")
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/summary_statistics.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary to console
        print(f"\n=== SIMULATION SUMMARY ===")
        print(f"Days completed: {len(sim.day_population)}")
        print(f"Final population: {len(sim.agents) if sim.agents else 0}")
        print(f"Total fights: {sum(sim.day_fights) if sim.day_fights else 0}")
        if sim.agents:
            avg_energy = np.mean([a.energy for a in sim.agents])
            print(f"Average final energy: {avg_energy:.1f}")
        
        print(f"\n📊 Charts saved to '{output_dir}/' directory:")
        print(f"  • population_evolution.png")
        print(f"  • combat_frequency.png") 
        print(f"  • energy_evolution.png")
        print(f"  • [trait]_evolution.png (8 individual trait plots)")
        print(f"  • summary_statistics.png")
        print("========================\n")
        
    except ImportError:
        print("Matplotlib not available for final charts. Install with: pip install matplotlib")
        print(f"\n=== SIMULATION SUMMARY ===")
        print(f"Days completed: {len(sim.day_population)}")
        print(f"Final population: {len(sim.agents) if sim.agents else 0}")
        print(f"Total fights: {sum(sim.day_fights) if sim.day_fights else 0}")
        if sim.agents:
            avg_energy = np.mean([a.energy for a in sim.agents])
            print(f"Average final energy: {avg_energy:.1f}")
        print("========================\n")