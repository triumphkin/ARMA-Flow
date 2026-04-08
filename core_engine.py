import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import behavior

random.seed = 96

class Car:
    def __init__(self, car_id, spawn_lane):
        self.id = car_id
        self.lane = spawn_lane # 0=Footpath, 1=Left, 2=Center(Merge), 3=Right
        self.position = 0.0 # Starts at 0 meters
        self.speed = random.uniform(15.0, 25.0) # Meters per second (~50-90 km/h)
        self.target_speed = 25.0
        self.ambulance=0
        self.color=0
        
        # --- HUMAN BEHAVIOR PROFILES ---
        self.aggression = random.uniform(0.0, 1.0)
        self.compliance = random.uniform(0.0, 1.0)
        self.impatience = 0.0 
        
        self.hard_brakes = 0
        self.has_finished = False

    def update_impatience(self):
        if self.speed < 10.0 and self.speed > 6:
            self.impatience = min(0.5, self.impatience + 0.05)
        elif self.speed <=6 and self.speed >4:
            self.impatience = min(0.7, self.impatience + 0.05)
        elif self.speed <= 4.0:
            self.impatience = min(0.9, self.impatience + 0.05)
        
        
        else:
            self.impatience = max(0.0, self.impatience - 0.03)

class TrafficSimulation:
    def __init__(self, road_length=1000, merge_point=800):
        self.road_length = road_length
        self.merge_point = merge_point
        self.cars = []
        self.time_step = 0
        self.data_log = []
        self.car_counter = 0
        # --- BATCH ZIPPER STATE (The AI Brain) ---
        self.ai_zipper_active = False  # The boolean condition!
        self.active_merge_lane = 1     # Starts by letting Lane 1 merge
        self.cars_merged_in_batch = 0  # Counter
        self.batch_size = 5            # How many cars per batch
        
        # --- TIMELINE SETUP ---
        # Start at midnight on the first day
        self.start_time = datetime(2026, 4, 1, 0, 0) 

    def spawn_cars(self, num_cars=3):
        for i in range(num_cars):
            # lane = random.choice([1,2,3])
            lane = random.choices([1, 2, 3], weights=[0.40, 0.20, 0.40])[0]
            if(lane==1): 
                color = 1
            elif(lane==2):
                color=2
            else:
                color=3

            new_car = Car(self.car_counter, lane)
            
            # THE FIX: Stagger the cars backwards so they don't spawn inside each other!
            new_car.position = 0.0 - (i * 25.0) 
            new_car.color = color
            self.cars.append(new_car)
            self.car_counter += 1
            
            if self.car_counter % 45 == 0:
                self.cars[-1].ambulance = 1


    def step(self):
        self.time_step += 1
        
        # Determine dynamic weather per step 
        self.weather = random.choices(['Clear', 'Rain', 'Fog'], weights=[0.80, 0.15, 0.05])[0]
        
        self.cars.sort(key=lambda c: c.position, reverse=True)
        lane_leaders = {0: None, 1: None, 2: None, 3: None}
        
        for car in self.cars:
            if car.has_finished:
                continue

            leader = lane_leaders[car.lane]
            
            # --- 1. PHYSICS & CAR FOLLOWING ---
            if leader is None or (leader.position - car.position) > 50:
                acceleration = 1.0 + (car.aggression)
                car.speed = min(car.target_speed, car.speed + acceleration)
            else:
                gap = leader.position - car.position
                safe_gap = 10.0 - (car.aggression * 5.0) 
                
                if gap < safe_gap:
                    brake_force = 5.0 + (car.aggression * 2.0)
                    car.speed = max(0.0, car.speed - brake_force)
                    car.hard_brakes += 1
                else:
                    car.speed = leader.speed

            # --- 2. HYPER-REALISTIC BOTTLENECK (Maximum Throughput Degradation) ---
            # Expanded to a 200-meter "Stress Zone" 
            merge_zone_start = self.merge_point - 200 

            if car.position > merge_zone_start and car.lane in [1, 3]:
                distance_to_end = self.merge_point - car.position
                
                # Late-merge behavior: Aggressive drivers speed to the end, timid drivers hesitate early.
                merge_probability = 0.02 + (car.aggression * 0.1)
                
                # Panic sets in the closer they get
                if distance_to_end < 50:
                    merge_probability += 0.4
                
                # 1. THE DEAD STOP: If they hit the absolute end without merging, they are stuck.
                if distance_to_end <= 5:
                    # If they haven't merged yet, they are forced to yield to Lane 2 traffic.
                    # They have a high chance of having to stop completely to wait for a gap.
                    if random.random() < 0.6: # 60% chance no gap is available this tick
                        car.speed = 0.0 # Complete stop. This blocks ALL cars behind them in Lane 1/3!
                        car.hard_brakes += 1
                        continue # Skip to the next car in the loop; they cannot merge right now.
                    else:
                        merge_probability = 1.0 # They found a gap from a standstill and force their way in.

                # If they decide (or are forced) to merge...
                if random.random() < merge_probability:
                    car.lane = 2
                    
                    # 2. THE CUT-OFF (Speed degradation based on desperation)
                    if distance_to_end > 50:
                        # Early merge: Hesitant, clumsy lane change
                        car.speed *= random.uniform(0.3, 0.6)
                    else:
                        # Late merge / Standstill merge: Violently forcing their way in.
                        # They enter Lane 2 at a crawl, obliterating the flow.
                        car.speed *= random.uniform(0.0, 0.2)
                        car.hard_brakes += 1
                        
                    # 3. THE STARTLE REFLEX (The true throughput killer)
                    # If they decide (or are forced) to merge...
                    if random.random() < merge_probability:
                        car.lane = 2
                        
                        # 2. THE CUT-OFF (Speed degradation based on desperation)
                        if distance_to_end > 50:
                            car.speed *= random.uniform(0.3, 0.6)
                        else:
                            car.speed *= random.uniform(0.0, 0.2)
                            car.hard_brakes += 1
                            
                        # 3. THE STARTLE REFLEX (Inline calculation)
                        # We need to find the closest car in Lane 2 that is behind the merging car.
                        closest_car_behind = None
                        min_distance = float('inf')
                        
                        # NOTE: Replace 'self.cars' with whatever your main list of vehicles is called.
                        for other_car in self.cars: 
                            # Check if the car is in the target lane AND behind our merging car
                            if other_car.lane == 2 and other_car.position < car.position:
                                dist = car.position - other_car.position
                                if dist < min_distance:
                                    min_distance = dist
                                    closest_car_behind = other_car
            
                        # If we found a car right behind them, trigger the human panic braking!
                        if closest_car_behind and min_distance < 40:
                            closest_car_behind.speed *= random.uniform(0.1, 0.3)
                            closest_car_behind.hard_brakes += 11

            # --- 3. BEHAVIOR ---
            active_cars = [c for c in self.cars if not c.has_finished]
        
            if active_cars:
            # Calculate the average speed of the whole road
                 current_global_avg_speed = np.mean([c.speed for c in active_cars])
            else:
            # If the road is empty, default to free-flowing speed
                 current_global_avg_speed = 25.0 
                 
    
            imp=behavior.calculate_impatience(current_global_avg_speed,car.target_speed, car.impatience)
            car.impatience = imp
            if behavior.decide_footpath_violation(car.impatience, car.compliance, car.lane, car.position, self.merge_point):
                car.lane = 0 
                car.speed = 20.0 
                car.impatience = 1.0 
                
            if car.lane == 0 and car.position >= self.merge_point:
                car.lane = 2
                car.speed = 2.0 

            car.position += car.speed
            if car.position >= self.road_length:
                car.has_finished = True

            
            active_cars = [c for c in self.cars if not c.has_finished]
            if not active_cars:
                return
        
            densities = {0: 0, 1: 0, 2: 0, 3: 0}
            for c in active_cars:
                densities[c.lane] += 1
            


            def decide_target_lane(current_lane, lane_densities, aggression):
                """
                Density-based lane change model across all 3 highway lanes.
                Returns the lane number the car should move to, or the current lane if it stays put.
                """
                # 1. PHYSICAL CONSTRAINTS: Which lanes can we actually steer into?
                # Lane 1 (Left) can only go to Lane 2 (Center)
                # Lane 3 (Right) can only go to Lane 2 (Center)
                # Lane 2 (Center) has the choice of going Left (1) or Right (3)
                if current_lane == 1:
                    adjacent_lanes = [2]
                elif current_lane == 3:
                    adjacent_lanes = [2]
                else: 
                    adjacent_lanes = [1, 3]

                # 2. ENVIRONMENTAL AWARENESS: Find the best adjacent lane
                # We look at the dictionary and pick the lane with the absolute lowest number of cars
                best_target_lane = min(adjacent_lanes, key=lambda l: lane_densities.get(l, 0))

                my_lane_density = lane_densities.get(current_lane, 0)
                target_lane_density = lane_densities.get(best_target_lane, 0)

                # 3. THE PRESSURE GAUGE: How much emptier is that other lane?
                density_advantage = my_lane_density - target_lane_density

                # If the best lane is equally crowded or worse, don't bother moving.
                if density_advantage <= 0:
                    return current_lane
                    
                # 4. THE HUMAN FACTOR (Psychology Threshold)
                # This formula scales based on the car's random aggression level (0.0 to 1.0)
                # - A highly aggressive driver (1.0) needs only 2 fewer cars to violently switch lanes.
                # - A passive, polite driver (0.0) needs a massive gap of 8 fewer cars to feel safe switching.
                required_advantage = 8.0 - (aggression * 6.0) 

                # 5. THE FINAL DECISION
                # If the relief of moving outweighs the driver's personal hesitation...
                if density_advantage >= required_advantage:
                    return best_target_lane # SWERVE!
                    
                # Otherwise, accept your fate and stay in the current lane.
                return current_lane             




            lane_leaders[car.lane] = car 

        


        self._record_state()

    def _record_state(self):
        active_cars = [c for c in self.cars if not c.has_finished]
        if not active_cars:
            return

        speeds = [c.speed for c in active_cars]
        avg_speed = np.mean(speeds)
        speed_variance = np.var(speeds)
        footpath_count = sum(1 for c in active_cars if c.lane == 0)
        
        densities = {0: 0, 1: 0, 2: 0, 3: 0}
        for c in active_cars:
            densities[c.lane] += 1

        throughput = sum(1 for c in self.cars if c.has_finished)

        # --- THE 6-MONTH TIMELINE LOGIC ---
        # Advance clock by exactly 2 hours for every simulation step
        current_real_time = self.start_time + timedelta(hours=self.time_step * 2)
        hour_of_day = current_real_time.hour
        month_of_year = current_real_time.month
        
       # --- 1. WEATHER PENALTY (Visibility & Physics) ---
        # Reality: Rain slows down average speed by ~15%, but Fog is incredibly 
        # dangerous and causes massive spacing, killing throughput.
        weather_penalty = 0 
        if self.weather == 'Rain': 
            weather_penalty = 20
        elif self.weather == 'Fog': 
            weather_penalty = 45 

        # --- 2. TIME PENALTY (Macro-Volume Baseline) ---
        # Reality: The road has a physical capacity. Rush hour pushes the road 
        # to 90% capacity automatically. Nighttime empties it.
        time_penalty = 0
        if 8 <= hour_of_day <= 10 or 17 <= hour_of_day <= 19:
            time_penalty = 50  # Massive baseline pressure increase
        elif 11 <= hour_of_day <= 16:
            time_penalty = 15  # Steady, manageable flow
        elif 0 <= hour_of_day <= 5:
            time_penalty = -40 # Dead of night, almost impossible to jam

        # --- 3. BEHAVIOR PENALTY (The Human Element) ---
        # Reality: Impatience causes erratic lane changes. Aggression causes 
        # tailgating and "Phantom Jams" (shockwave braking). Compliance smooths flow.
        avg_impatience = np.mean([c.impatience for c in active_cars]) if active_cars else 0
        avg_aggression = np.mean([c.aggression for c in active_cars]) if active_cars else 0
        avg_compliance = np.mean([c.compliance for c in active_cars]) if active_cars else 1.0

        # We increased the compliance reward. Good drivers actively relieve pressure.
        behavior_penalty = (avg_impatience * 60) + (avg_aggression * 50) - (avg_compliance * 30)    
            
        # --- 4. THE REALISTIC SYSTEM PRESSURE FORMULA ---
        # - We removed the weird decimals (*0.3, *1.4) to let the raw penalties speak.
        # - Speed Variance gets a *1.5 multiplier because "stop-and-go" waves are the #1 indicator of a jam.
        # - Footpath Cheaters get a *25 multiplier. One car merging violently from the footpath ruins the flow for 50 cars behind it.
        system_pressure = densities[1] + densities[2] + (densities[3] * 0.4) + (speed_variance * 1.5) + weather_penalty + (footpath_count * 25) + time_penalty + behavior_penalty

        # --- 5. THE REALISTIC THRESHOLDS ---
        # Because the max theoretical pressure is now much higher (~450+), 
        # we widened the "buckets". This ensures the simulation doesn't just 
        # scream "GRIDLOCK" all the time. It has to actually build up.
        
        if system_pressure > 280:
            target_gridlock = max(0, random.randint(0, 15))  # Imminent failure
        elif system_pressure > 200:
            target_gridlock = random.randint(15, 45)         # Tipping point reached
        elif system_pressure > 130:
            target_gridlock = random.randint(45, 120)        # Heavy, but moving
        else:
            target_gridlock = random.randint(120, 300)       # Smooth sailing 

        # --- LOGGING DATA ---
        for car in active_cars:
            row = {
                "Timestamp_Step": self.time_step,
                "Real_World_Time": current_real_time.strftime("%Y-%m-%d %H:%M:%S"),
                "Month": month_of_year,
                "Hour_Of_Day": hour_of_day,
                "Weather_Condition": self.weather,
                
                "Car_ID": car.id,
                "Position_X": round(car.position, 2),
                "Lane": car.lane,
                "Speed": round(car.speed, 2),
                "Aggression_Profile": round(car.aggression, 2),
                "Compliance_Profile": round(car.compliance, 2),
                "Current_Impatience": round(car.impatience, 2),
                "Hard_Braking_Count": car.hard_brakes,
                
                "Global_Avg_Speed": round(avg_speed, 2),
                "Global_Speed_Variance": round(speed_variance, 2),
                "Footpath_Count": footpath_count,
                "Density_Lane1": densities[1],
                "Density_Lane2": densities[2],
                "Density_Lane3": densities[3],
                "Total_Throughput": throughput,
                "Is_Ambulance": car.ambulance,
                
                "Seconds_To_Gridlock": target_gridlock,
                "Color": car.color 
            }
            self.data_log.append(row)

    def export_data(self, filename="traffic_data.csv"):
        df = pd.DataFrame(self.data_log)
        df.to_csv(filename, index=False)
        print(f"Data successfully exported to {filename} with {len(df)} rows.")

if __name__ == "__main__":
    sim = TrafficSimulation()
    
    # 180 days * (24 hours / 2 hour jumps) = 2160 steps
    total_steps = 300 
    
    print(f"Starting 6-Month Simulation Generator (Total intervals: {total_steps})...")
    
    for step in range(total_steps):
        # We need to fluctuate car spawning based on time of day for realism
        current_hour = (sim.start_time + timedelta(hours=step * 2)).hour
        
        if 8 <= current_hour <= 10 or 17 <= current_hour <= 19:
            sim.spawn_cars(random.randint(3, 8)) # Heavy spawning during rush hour
        elif 0 <= current_hour <= 5:
            if random.random() > 0.5: # 50% chance to spawn 1 car at night
                sim.spawn_cars(1)
        else:
            sim.spawn_cars(random.randint(1, 4)) # Normal daytime spawning
        
        sim.step()
        
    sim.export_data("baseline_chaos_data.csv")