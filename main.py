import flowpathflood as ff

# --- CONFIGURATION ---
DEM = 'DEM.tif'        # Path to your DEM
FDIR = 'fdir.tif'      # Path to your Flow Direction
STREAMS = 'streams.shp' # Path to your Stream Shapefile
Q_DATA = 'qmax.csv'    # Path to your Discharge CSV

# --- EXECUTION ---

# Phase 1: Spatial Setup
print("Starting Preprocessing...")
ff.run_preprocessing(DEM, FDIR, STREAMS)

# Phase 2: Terrain Normalization
print("Starting HAND Analysis...")
ff.run_hand_analysis(depth=1.0, resolution=1.0)

# Phase 3: Hydraulic Modeling
print("Starting Manning Analysis...")
ff.run_manning_analysis(Q_DATA, n_roughness=0.20, output_name='flood_final.tif')

print("\nFinished! Your results are in the 'flowpath_segments' folder.")
print("The final combined map is: flood_final.tif")