# ==========================================
# DRONE INSPECTION ROUTING - MAIN INTERFACE
# ==========================================
import numpy as np
import matplotlib.pyplot as plt

# Import the core math functions from the separate file
from Test_time_functions import (
    get_distance_lawnmower, 
    get_distance_spiral, 
    get_velocity_rgb_lawnmower, 
    get_velocity_rgb_spiral, 
    get_velocity_event, 
    get_velocity_hyper_lawnmower, 
    time_lawnmower, 
    time_spiral,
    plot_inspection_route
)



# ---  MAIN EXECUTION LOGIC ---
if __name__ == "__main__":
    # ---------------------------------------------------------
    # A. TOWER & TURBINE GEOMETRY CONFIGURATION
    # ---------------------------------------------------------
    R_base = 4.0      # Monopile radius (Constant from seafloor to tower top base)
    R_top = 2.5       # Nacelle interface radius at the top of the cone
    
    H_water = 30.0    # Underwater portion of the monopile
    H_air_cyl = 15.0  # Above-water portion of the monopile (before the cone starts)
    H_air_cone = 70.0 # Height of the tapered tower itself
    
    # Turbine Configuration (Modeled as 3 equivalent cylinders)
    R_blade = 1.2     # Abstract radius of the blade cylinder
    H_blade = 40.0    # Length of each blade
    
    # ---------------------------------------------------------
    # B. ENVIRONMENT & PAYLOAD PARAMETERS 
    # ---------------------------------------------------------
    
    # WATER PROFILE
    water_config = {
        "camera_type": "RGB",         
        "flight_mode": "spiral",   
        "w_arc": 0.8,                 
        "w_flat": 0.5,                
        "v_kin_vert": 0.5,            
        "v_kin_horiz": 0.5,           
        "camera": {
            "gsd": 0.001, "max_blur": 2.0, "shutter": 0.005, 
            "h_fov": 0.6, "overlap": 0.85, "fps": 10,        
            "line_rate": 100, "integration": 0.01            
        }
    }
    
    # AIR PROFILE
    air_config = {
        "camera_type": "EVENT",       
        "flight_mode": "spiral",      
        "w_arc": 2.5,                 
        "w_flat": 1.5,                
        "v_kin_vert": 5.0,            
        "v_kin_horiz": 4.0,           
        "camera": {
            "gsd": 0.001, "max_blur": 2.0, "shutter": 0.001, 
            "h_fov": 1.5, "overlap": 0.60, "fps": 10,        
            "line_rate": 200, "integration": 0.004           
        }
    }
    
    # TURBINE PROFILE
    turbine_config = {
        "camera_type": "RGB",         
        "flight_mode": "lawnmower",   
        "w_arc": 2.0,                 
        "w_flat": 1.0,                
        "v_kin_vert": 3.0,            
        "v_kin_horiz": 3.0,           
        "camera": {
            "gsd": 0.001, "max_blur": 2.0, "shutter": 0.001, 
            "h_fov": 1.5, "overlap": 0.70, "fps": 10,        
            "line_rate": 200, "integration": 0.004           
        }
    }
    
    transition_penalty_seconds = 45.0 



    # DONT CHANGE CODE AFTER THIS POINT
    # ---------------------------------------------------------
    # C. EXECUTE PHASE 1: UNDERWATER
    # ---------------------------------------------------------
    t_water = 0.0
    c_w = water_config["camera"]
    
    if water_config["flight_mode"] == "lawnmower":
        dist_w = get_distance_lawnmower(R_base, R_base, H_water, water_config["w_arc"])
        
        if water_config["camera_type"] == "RGB":
            v_vert = get_velocity_rgb_lawnmower(water_config["v_kin_vert"], c_w["gsd"], c_w["max_blur"], c_w["shutter"], c_w["h_fov"], c_w["overlap"], c_w["fps"])
        elif water_config["camera_type"] == "EVENT":
            v_vert = get_velocity_event(water_config["v_kin_vert"])
        elif water_config["camera_type"] == "HYPERSPECTRAL":
            v_vert = get_velocity_hyper_lawnmower(water_config["v_kin_vert"], c_w["gsd"], c_w["line_rate"], c_w["integration"], c_w["max_blur"])
            
        t_water = time_lawnmower(dist_w["vertical"], dist_w["horizontal"], v_vert, water_config["v_kin_horiz"])

    elif water_config["flight_mode"] == "spiral":
        dist_w = get_distance_spiral(R_base, R_base, H_water, water_config["w_flat"])
        
        if water_config["camera_type"] == "RGB":
            v_path = get_velocity_rgb_spiral(water_config["v_kin_vert"], R_base, water_config["w_flat"], water_config["w_arc"], c_w["gsd"], c_w["max_blur"], c_w["shutter"], c_w["overlap"], c_w["fps"])
        elif water_config["camera_type"] == "EVENT":
            v_path = get_velocity_event(water_config["v_kin_vert"])
            
        t_water = time_spiral(dist_w, v_path)

    # ---------------------------------------------------------
    # D. EXECUTE PHASE 2: AIR
    # ---------------------------------------------------------
    t_air = 0.0
    c_a = air_config["camera"]
    
    if air_config["flight_mode"] == "lawnmower":
        dist_air_cyl = get_distance_lawnmower(R_base, R_base, H_air_cyl, air_config["w_arc"])
        dist_air_cone = get_distance_lawnmower(R_base, R_top, H_air_cone, air_config["w_arc"])
        
        total_vert_air = dist_air_cyl["vertical"] + dist_air_cone["vertical"]
        total_horiz_air = dist_air_cyl["horizontal"] 
        
        if air_config["camera_type"] == "RGB":
            v_vert = get_velocity_rgb_lawnmower(air_config["v_kin_vert"], c_a["gsd"], c_a["max_blur"], c_a["shutter"], c_a["h_fov"], c_a["overlap"], c_a["fps"])
        elif air_config["camera_type"] == "EVENT":
            v_vert = get_velocity_event(air_config["v_kin_vert"])
        elif air_config["camera_type"] == "HYPERSPECTRAL":
            v_vert = get_velocity_hyper_lawnmower(air_config["v_kin_vert"], c_a["gsd"], c_a["line_rate"], c_a["integration"], c_a["max_blur"])
            
        t_air = time_lawnmower(total_vert_air, total_horiz_air, v_vert, air_config["v_kin_horiz"])

    elif air_config["flight_mode"] == "spiral":
        dist_air_cyl = get_distance_spiral(R_base, R_base, H_air_cyl, air_config["w_flat"])
        dist_air_cone = get_distance_spiral(R_base, R_top, H_air_cone, air_config["w_flat"])
        total_dist_air = dist_air_cyl + dist_air_cone
        
        if air_config["camera_type"] == "RGB":
            v_path = get_velocity_rgb_spiral(air_config["v_kin_vert"], R_top, air_config["w_flat"], air_config["w_arc"], c_a["gsd"], c_a["max_blur"], c_a["shutter"], c_a["overlap"], c_a["fps"])
        elif air_config["camera_type"] == "EVENT":
            v_path = get_velocity_event(air_config["v_kin_vert"])
            
        t_air = time_spiral(total_dist_air, v_path)

    # ---------------------------------------------------------
    # E. EXECUTE PHASE 3: TURBINE (3 Independent Blades)
    # ---------------------------------------------------------
    t_turbine = 0.0
    c_t = turbine_config["camera"]
    
    if turbine_config["flight_mode"] == "lawnmower":
        dist_t = get_distance_lawnmower(R_blade, R_blade, H_blade, turbine_config["w_arc"])
        
        if turbine_config["camera_type"] == "RGB":
            v_vert = get_velocity_rgb_lawnmower(turbine_config["v_kin_vert"], c_t["gsd"], c_t["max_blur"], c_t["shutter"], c_t["h_fov"], c_t["overlap"], c_t["fps"])
        elif turbine_config["camera_type"] == "EVENT":
            v_vert = get_velocity_event(turbine_config["v_kin_vert"])
        elif turbine_config["camera_type"] == "HYPERSPECTRAL":
            v_vert = get_velocity_hyper_lawnmower(turbine_config["v_kin_vert"], c_t["gsd"], c_t["line_rate"], c_t["integration"], c_t["max_blur"])
            
        t_turbine = 3 * time_lawnmower(dist_t["vertical"], dist_t["horizontal"], v_vert, turbine_config["v_kin_horiz"])

    elif turbine_config["flight_mode"] == "spiral":
        dist_t = get_distance_spiral(R_blade, R_blade, H_blade, turbine_config["w_flat"])
        
        if turbine_config["camera_type"] == "RGB":
            v_path = get_velocity_rgb_spiral(turbine_config["v_kin_vert"], R_blade, turbine_config["w_flat"], turbine_config["w_arc"], c_t["gsd"], c_t["max_blur"], c_t["shutter"], c_t["overlap"], c_t["fps"])
        elif turbine_config["camera_type"] == "EVENT":
            v_path = get_velocity_event(turbine_config["v_kin_vert"])
            
        t_turbine = 3 * time_spiral(dist_t, v_path)

    # ---------------------------------------------------------
    # F. FINAL OUTPUT
    # ---------------------------------------------------------
    total_inspection_time = t_water + transition_penalty_seconds + t_air + t_turbine
    
    print("=== INSPECTION ROUTE RESULTS ===")
    print(f"Water Phase   ({water_config['camera_type']} / {water_config['flight_mode'].capitalize()}): \t{t_water / 60:.2f} min")
    print(f"Transition Penalty: \t\t\t{transition_penalty_seconds / 60:.2f} min")
    print(f"Air Phase     ({air_config['camera_type']} / {air_config['flight_mode'].capitalize()}): \t{t_air / 60:.2f} min")
    print(f"Turbine Phase ({turbine_config['camera_type']} / {turbine_config['flight_mode'].capitalize()}): \t{t_turbine / 60:.2f} min")
    print("-" * 50)
    print(f"TOTAL ESTIMATED TIME: \t\t\t{total_inspection_time / 60:.2f} min")
    
    # ---------------------------------------------------------
    # G. VISUALIZE FLIGHT PATH
    # ---------------------------------------------------------
    print("\nGenerating 3D interactive plot...")
    plot_inspection_route(R_base, R_top, H_water, H_air_cyl, H_air_cone, H_blade, R_blade, water_config, air_config, turbine_config)