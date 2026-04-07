import pandas as pd
import random
from datetime import timedelta
from core_engine import TrafficSimulation

def run_massive_data_generation():
    all_episodes_data = []
    
    # 50 Episodes. With 800 steps (2-hour jumps), each episode simulates ~66 days of traffic.
    # Total dataset: ~9 years worth of combined simulated traffic data.
    total_episodes = 2
    
    print(f"Starting Massive Data Generation: {total_episodes} Episodes...")
    
    for episode in range(total_episodes):
        # Randomize the "Type of Region" or "Macro Trend"
        # > 1.5 is a high-density city, < 1.0 is a quiet suburban highway
        traffic_volume_multiplier = random.uniform(0.5, 2.5) 
        
        # Initialize a fresh world for this episode
        sim = TrafficSimulation()
        
        # Run the simulation for 800 time steps
        for step in range(800):
            # Calculate what time of day it is currently in the simulation
            current_hour = (sim.start_time + timedelta(hours=step * 2)).hour
            
            # 1. Determine BASE spawn rate based on the time of day (Rush hour logic)
            if 8 <= current_hour <= 10 or 17 <= current_hour <= 19:
                base_spawn = random.randint(3, 8)  # Heavy base traffic
            elif 0 <= current_hour <= 5:
                base_spawn = 1 if random.random() > 0.5 else 0  # Dead of night
            else:
                base_spawn = random.randint(1, 4)  # Normal midday traffic
                
            # 2. Apply the Episode's unique volume multiplier
            adjusted_spawn = int(base_spawn * traffic_volume_multiplier)
            
            # Spawn the cars if the adjusted volume is greater than 0
            if adjusted_spawn > 0:
                sim.spawn_cars(adjusted_spawn)
            
            # Move the simulation forward one tick (calculates physics, weather, and gridlock target)
            sim.step() 
            
        # Extract the data log from this specific episode simulation
        for row in sim.data_log:
            # Tag the generated data so Mahi can use these as macro-features if she wants
            row['Episode_ID'] = episode
            row['Regional_Traffic_Multiplier'] = round(traffic_volume_multiplier, 2)
            all_episodes_data.append(row)
            
        print(f"Episode {episode+1}/{total_episodes} complete | Multiplier: {traffic_volume_multiplier:.2f} | Cars processed: {sim.car_counter}")
        
    # Compile everything into one massive DataFrame
    print("\nCompiling data...")
    final_df = pd.DataFrame(all_episodes_data)
    
    # Save to the file
    filename = "training_dataset.csv"
    final_df.to_csv(filename, index=False)
    
    print(f"SUCCESS: Generated {len(final_df)} rows of advanced, temporally-aware traffic data.")
    print(f"File saved as -> {filename}. Ready for Mahi's ml_logic.py!")

if __name__ == "__main__":
    run_massive_data_generation()