import yaml
import scipy.constants as const
import numpy as np
import matplotlib.pyplot as plt
"""
This code simulates the power drainage of a drone over time with the mission profile programmed in.
"""




def load_configuration():
    """
    Loads the configuration for the simulation.
    """
    
    with open("configuration.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config
   
class Drone:
    def __init__(self, config):
        self.config = config
        
        self.drone = config["drone"]
        self.weight = self.drone["weight"] * const.g  # in N
        
        self.battery_weight = self.config['battery_weight_fraction'] * self.drone['weight']  # in kg
        self.battery_capacity = self.battery_weight * self.config["battery_specific_energy"]  # in Wh
        
        self.current_battery = self.battery_capacity  # in Wh
        self.dt = 0.1 # Time step for simulation in seconds
        self.current_power_consumption = 0.0  # in W
        self.time_elapsed = 0.0  # in seconds
        
        
        self.h = 0 # Altitude in meters
        self.v = 0 # Vertical speed in m/s
        self.a = 0 # Vertical acceleration in m/s^2
        
    def get_hover_thrust(self):
        """
        Returns the thrust needed for hovering.
        """
        return self.weight  # in N, since thrust must equal weight for hover
   
    def get_hover_power(self):
        """
        Returns the power consumption for hovering.
        """

        thrust_per_motor = self.weight / self.config["num_motors"]  # in N        
        get_power_from_thrust = lambda thrust: np.sqrt((thrust ** 3) / (self.config["air_density"] * const.pi * (self.config["propeller_diameter"] / 2) ** 2 * self.config["hover_efficiency"]))  # in W
        power_per_motor = get_power_from_thrust(thrust_per_motor) # in W
        return power_per_motor * self.config["num_motors"]  # Total power for all motors
    
    def add_power(self, power):
        """
        Adds to the current power consumption for the drone.
        """
        self.current_power_consumption += power
    
    
    def update_battery(self):
        """
        Updates the battery level based on the power consumption.
        """
        self.current_battery -= self.current_power_consumption * self.dt/3600 / self.config["battery_efficiency"]  # Convert W to Wh
        if self.current_battery < 0:
            self.current_battery = 0
        self.current_power_consumption = 0  # Reset power consumption after update
        return self.current_battery

    def get_motor_power(self, required_thrust):
        """
        Placeholder for calculating motor power based on required thrust.
        For now, we assume a linear relationship between thrust and power.
        """
        
        get_power_from_thrust = lambda thrust: np.sqrt((thrust ** 3) / (self.config["air_density"] * const.pi * (self.config["propeller_diameter"] / 2) ** 2 * self.config["hover_efficiency"]))  # in W
        power_per_motor = get_power_from_thrust(required_thrust/self.config["num_motors"]) # in W
        return power_per_motor * self.config["num_motors"]  # Total power for all motors
        

    def get_required_thrust(self, required_height):
        """
        Placeholder for calculating required thrust based on mission profile.
        For now, we assume the drone is always hovering, so we just return the hover power.
        """
        
        height_error = required_height - self.h  # Calculate height error
        
        gain = 0.01  # Proportional gain for height control
        
        thrust_needed = np.clip(height_error*gain+self.get_hover_thrust(), 0, self.config["max_thrust_per_motor"] * self.config['num_motors'])  # Ensure thrust is within limits

        return thrust_needed

    


if __name__ == "__main__":
    config = load_configuration()
    
    drone = Drone(config)
    
    print(f"Initial battery level: {drone.current_battery:.1f} Wh")
    print(f"Battery weight: {drone.battery_weight:.1f} kg")
    
    power_over_time = np.array([])
    battery_over_time = np.array([])
    time = np.array([])
    height = np.array([])
    velocity = np.array([])
    thrust_list = np.array([])
    
    
    while drone.current_battery > 0:
        
        thrust = drone.get_required_thrust(100)  # Example: target height of 10 meters
        needed_power = drone.get_motor_power(thrust)
        
        drone.a = (thrust-drone.weight) / drone.drone["weight"]  # Calculate vertical acceleration
        drone.v += drone.a * drone.dt  # Update vertical speed
        drone.h += drone.v * drone.dt  # Update altitude
        
        
        # Load power loading with sensor power load
        # drone.add_power(10)  # Assume 10 W for sensors and other electronics
        drone.add_power(needed_power)  # Add power consumption based on needed thrust
       
        
        current_power_load = drone.current_power_consumption
        drone.update_battery()
        drone.time_elapsed += drone.dt
        power_over_time = np.append(power_over_time, current_power_load)
        battery_over_time = np.append(battery_over_time, drone.current_battery)
        time = np.append(time, drone.time_elapsed)
        height = np.append(height, drone.h)
        velocity = np.append(velocity, drone.v)
        thrust_list = np.append(thrust_list, thrust)
        # print(f"Time: {drone.time_elapsed:.1f} s, Battery: {drone.current_battery:.1f} Wh, Power Load: {current_power_load:.1f} W, Height: {drone.h:.1f} m")
        

    print("------------------------------------------------------------------")
    print("Battery depleted. Simulation ended.")
    print()
    print(f"Total flight time: {drone.time_elapsed:.1f} seconds ({drone.time_elapsed/60:.2f} minutes, {drone.time_elapsed/3600:.2f} hours)")

    plt.plot(time, battery_over_time)
    plt.xlabel("Time (s)")
    plt.ylabel("Battery Level (Wh)")
    plt.title("Drone Battery Drainage")
    plt.show()
    
    plt.plot(time, power_over_time)
    plt.xlabel("Time (s)")
    plt.ylabel("Power Consumption (W)")
    plt.title("Drone Power Consumption")
    plt.show()
    
    plt.plot(time, height)
    plt.plot(time, velocity, color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Drone Height Profile")
    plt.show()
    
    plt.plot(time, thrust_list)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust (N)")
    plt.title("Drone Thrust Profile")
    plt.show()
