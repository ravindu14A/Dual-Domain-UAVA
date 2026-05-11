import yaml
import scipy.constants as const
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
        self.battery_capacity = config["battery_capacity"]  # in mAh
        self.current_battery = self.battery_capacity  # in mAh
        self.power_consumption = config["power_consumption"]  # in mA
        self.weight = config["weight"] * 1000 * const.gravitational_constant  # in N

    def update_battery(self, time_elapsed):
        """
        Updates the battery level based on the time elapsed and power consumption.
        """
        power_used = (self.power_consumption * time_elapsed) / 3600  # Convert to mAh
        self.current_battery -= power_used
        if self.current_battery < 0:
            self.current_battery = 0   
   
    def get_hover_power(self):
        """
        Returns the power consumption for hovering.
        """
        return 100  # Placeholder value

 

if __name__ == "__main__":
    config = load_configuration()
    print(config)