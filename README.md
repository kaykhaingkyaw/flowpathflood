# flowpathflood : Flow Path Flood Extension Algorithm (Urban areas)
# Overview
This algorithm simulates flooding along flow paths that connect depressions in urban areas. These paths typically correspond to primary roads and lanes within the urban fabric.
## 📊 Data Availability

The example dataset required to run this algorithm (including the 1m resolution LiDAR DEM, Streams, Flow Direction rasters, and Discharge lookup tables) is hosted on Zenodo.

**Dataset DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18484142.svg)](https://doi.org/10.5281/zenodo.18484142)

### Instructions to Setup Data:
1. **Download** the zip file from [Zenodo (DOI: 10.5281/zenodo.18484142)](https://doi.org/10.5281/zenodo.18484142).
2. **Create** a folder named `data` in your project root directory.
3. **Extract** the contents into that folder. Your directory should look like this:
   ```text
   /flowpathflood-project
     ├── data/
     │    ├── DEM.tif
     │    ├── fdir.tif
     │    ├── streams.shp
     │    └── qmax.csv
     ├── requirements.txt
     ├── main.py
     ├── flowpathflood.py
     └── README.md
   ```
# 🚀 Methodology: The Three Phases
# Phase1: Preprocessing 
This phase prepares the raw geospatial data for analysis. It focuses on converting flow direction encodings and breaking the flow path into discrete segments for better accuracy and localized calculation.
### Step 1: Stream splitting
```python
import flowpathflood as ff

# Path to your urban flow path shapefile (e.g., roads/lanes)
input_shp = "data/streams.shp"

# Execute Step 1: Split the master file into individual node shapefiles
node_files = ff.split_streams_by_node(input_shp, output_dir="output_streams")

print(f"Successfully split into {len(node_files)} individual node files.")
```
#### Expected Terminal Output:

<img width="729" height="683" alt="01" src="https://github.com/user-attachments/assets/82be80f4-2313-4af1-a16d-55aa5b7b0e9c" />

---

### Step 2: Flow Direction Conversion

```python
import flowpathflood as ff
import numpy as np

# Define paths
input_fdir = "data/fdir.tif"
output_fdir = "fdir_pysheds.tif"

# Execute Conversion
output_data, meta = ff.convert_d8_encoding(input_fdir, output_fdir)

# Print Conversion Summary
direction_names = {
    1: "E", 2: "SE", 4: "S", 8: "SW",
    16: "W", 32: "NW", 64: "N", 128: "NE"
}

print("Conversion Summary:")
print("New FDIR values and their counts:")

unique, counts = np.unique(output_data, return_counts=True)
for val, count in zip(unique, counts):
    if val in direction_names:
        print(f"{direction_names[val]} ({val}): {count} pixels")
```
#### Expected Terminal Output:
```text
Conversion Summary:
New FDIR values and their counts:
E (1): 178101 pixels
SE (2): 131524 pixels
S (4): 263404 pixels
SW (8): 150896 pixels
W (16): 430662 pixels
NW (32): 196602 pixels
N (64): 343007 pixels
NE (128): 134554 pixels
```

---


### Step 3: Stream-Specific Flow Direction Extraction

```python
import flowpathflood as ff

# 1. Define paths
fdir_raster = "fdir_pysheds.tif"     # Generated in Step 2
stream_dir = "output_streams"        # Generated in Step 1
output_dir = "output_rasters"

# 2. Execute extraction and masking
ff.extract_stream_fdir(
    fdir_raster_path=fdir_raster, 
    stream_shp_dir=stream_dir, 
    output_raster_dir=output_dir,
    verbose=True
)
```

#### Expected Terminal Output:
```text
Flow Direction Statistics for nodeid=1:
   Direction W (16): 2 pixels
   Direction NW (32): 10 pixels
   Direction N (64): 19 pixels
   Direction NE (128): 3 pixels
   Total stream pixels: 34
   Saved raster: output_rasters/stream_fdir_nodeid_1.tif

Processing complete! All flow direction rasters have been saved.
```

---

### Step 4: Flow Path Segmentation

```python
import flowpathflood as ff

# 1. Define paths
input_dir = "output_rasters"       # Generated in Step 3
output_base = "flowpath_segments"

# 2. Execute segmentation
# This will group segments by Node ID in the output directory
ff.segment_flowpaths(
    input_dir=input_dir, 
    base_output_dir=output_base, 
    segment_size=110
)
```

#### Expected Terminal Output:
```text
Processing flowpath raster: stream_fdir_nodeid_4.tif (Node ID: 4)
  Saved TIFF file for Segment 1: flowpath_segments/nodeid_4/tiff_segments/flowpath_segment_1.tif
  Saved TIFF file for Segment 2: flowpath_segments/nodeid_4/tiff_segments/flowpath_segment_2.tif
  Saved shapefile for Node ID 4: flowpath_segments/nodeid_4/flowpath_segments_nodeid_4.shp
  Total segments created for Node ID 4: 2

Segmentation complete.
```
---
### Step 5: Segment Slope Calculation

This step calculates the longitudinal slope ($S$) for every individual segment identified in Step 4. By extracting elevation values from the Digital Elevation Model (DEM) along the flow path coordinates, the algorithm determines the average gradient (rise/run), which is a critical input for the Manning's hydraulic equation in Phase 3.



```python
import flowpathflood as ff

# 1. Define paths
dem_file = "data/DEM.tif"
segments_base_dir = "flowpath_segments"

# 2. Execute slope calculation
# This reads the DEM and calculates the gradient for each segment SHP
ff.calculate_segment_slopes(
    dem_path=dem_file, 
    base_dir=segments_base_dir
)
```

#### Expected Terminal Output:
```text
Saved slope values to text file for Node ID nodeid_4: flowpath_segments/nodeid_4/Slopes.txt
Slope calculation complete.
```

---
### Step 6: Buffering and Cropping

This final preprocessing step creates a spatial "buffer" around each flow path segment (default: 10 meters). The algorithm then crops the global DEM and Flow Direction (FDIR) rasters to these localized zones. This dramatically optimizes Phase 2 by ensuring the HAND (Height Above Nearest Drainage) analysis only processes the immediate terrain surrounding the urban conduits.



```python
import flowpathflood as ff

# 1. Define paths
dem_file = "data/urban_dem.tif"
fdir_file = "fdir_pysheds.tif"      # From Step 2
segments_dir = "flowpath_segments"

# 2. Execute Buffering and Cropping
# This creates localized DEM and FDIR clips for every segment
ff.create_buffered_environmental_data(
    dem_file=dem_file, 
    fdir_file=fdir_file, 
    base_dir=segments_dir, 
    buffer_dist=10
)
```

#### Expected Terminal Output:

<img width="699" height="589" alt="02-01" src="https://github.com/user-attachments/assets/189345c1-4fd2-47e5-b345-85b18c389278" />

<img width="770" height="589" alt="03-01" src="https://github.com/user-attachments/assets/55d601dd-76dd-44da-beb7-3d553e829856" />

---

# Phase 2: The Height Above Nearest Drainage (HAND) calculation
### Step7: Localized HAND Calculation

This is the core hydrological step. Using the **Pysheds** library, the algorithm calculates the relative elevation of every pixel in the buffered zone compared to its "downstream" flow path pixel. The result is a HAND raster (Nobre et al., 2011b).

![05](https://github.com/user-attachments/assets/99e624ee-8088-40bb-b75c-3dd92cc5b0a8)




```python
import flowpathflood as ff

# 1. Path to the segments processed in Phase 1
segments_dir = "flowpath_segments"

# 2. Execute HAND calculation
# This iterates through each node and segment to compute relative heights
ff.compute_hand_for_segments(base_dir=segments_dir)
```

#### Expected Terminal Output:

<img width="556" height="870" alt="04-01" src="https://github.com/user-attachments/assets/a41a64dd-2534-43a0-8ecc-0855aa9ef25d" />

---

### Step 8: Constant Depth Inundation Mapping

In this step, the algorithm converts the HAND rasters into flood depth maps. By applying a water level (constant depth) to the segment, the algorithm identifies all surrounding pixels where the terrain is lower than the water surface. 


```python
import flowpathflood as ff

# 1. Path to the processed segments
segments_dir = "flowpath_segments"

# 2. Execute Inundation Mapping
# Set the constant_depth (in meters) to simulate different flood scenarios
ff.generate_inundation_extents(
    base_dir=segments_dir, 
    constant_depth=1.0  # Simulate 1 meter of water depth
)
```

#### Expected Terminal Output:

<img width="837" height="790" alt="06" src="https://github.com/user-attachments/assets/7f46ea6d-11fa-4e32-a1e0-179014217675" />

----
### Step 9: Inundation Geometry Statistics

This step quantifies the flood footprint. By analyzing the HAND rasters against the constant water depth and the original flow path centerlines, the algorithm calculates:
1.  **Segment Length**: The physical length of the road centerline.
2.  **Inundation Area**: The total surface area covered by water (based on pixel resolution).
3.  **Average Width**: The calculated width of the flood spread ($Width = Area / Length$).



```python
import flowpathflood as ff

# 1. Path to the segments folder
segments_dir = "flowpath_segments"

# 2. Execute Statistics Calculation
# constant_depth: Water level in meters
# pixel_res: The resolution of your DEM (e.g., 1.0 for a 1m pixel)
ff.calculate_inundation_stats(
    base_dir=segments_dir, 
    constant_depth=1.0, 
    pixel_res=1.0
)
```

#### Expected Terminal Output:
```text
Processing Node ID: nodeid_14
File: HAND_cropped_dem_flowpath_buffer_segment_1.tif, Segment ID: 1
Segment Length: 47.21
Total inundation area: 795.00
Approximate inundation width: 16.84

Inundation results saved to: flowpath_segments/nodeid_14/Inundation_results/Inundation_results.csv
```

---
# Phase 3: Manning's Equation
### Step 10: Hydraulic Water Depth Solver

In this step, the algorithm moves beyond constant depths and solves for the **real-world water depth ($h$)** . It uses Manning's Equation to balance the discharge (flow rate), the roughness of the urban surface (Manning's $n$), and the physical slope of the street.



The solver uses the `fsolve` numerical method to find the depth ($h$) in the equation:
$$Q = \frac{1}{n} A R^{2/3} S^{1/2}$$
Where:
* **$Q$**: Discharge from the lookup table (provided by user).
* **$n$**: Surface roughness (e.g., $0.20$ for complex urban lanes).
* **$A$**: Cross-sectional area ($Width \times h$).
* **$R$**: Hydraulic radius ($A / Wetted\ Perimeter$).
* **$S$**: Slope (calculated in Step 5).

```python
import flowpathflood as ff

# 1. Define input paths
q_lookup = "data/discharge_lookup.csv"  # File containing nodeid and q
segments_dir = "flowpath_segments"

# 2. Run the Manning Depth Solver
# n_roughness: Manning's n (0.013 for smooth concrete, ~0.2 for cluttered lanes)
ff.solve_manning_depths(
    q_lookup_file=q_lookup, 
    base_dir=segments_dir, 
    n_roughness=0.20
)
```

#### Expected Terminal Output:
```text
Processing Node ID: nodeid_10
File: HAND_cropped_dem_flowpath_buffer_segment_1.tif, Segment ID: 1
Using pre-calculated width: 19.3953 meters
Using pre-calculated slope: 0.018736
Using flow rate (Q): 0.0000 m³/s
Calculated water depth (h): 0.0000 meters

Water depths saved to flowpath_segments/nodeid_10/Water_Depths.csv

Manning Depth solver complete.
```

---

### Step 11: Inundation Mapping based on final water depth

This is the step of the `flowpathflood` pipeline. Instead of using a uniform "guess" for water depth, this function uses the unique water depth calculated for each specific segment in Step 10. By subtracting the HAND terrain values from these dynamic depths, it generates the most accurate representation of the flood extent based on local hydraulics.



```python
import flowpathflood as ff

# 1. Path to the processed segments
segments_dir = "flowpath_segments"

# 2. Execute Dynamic Mapping
# This uses the specific depths from Water_Depths.csv
ff.map_hydraulic_inundation(base_dir=segments_dir)
```

#### Expected Terminal Output:
```text
Hydraulic inundation mapping complete.
```
<img width="838" height="790" alt="07" src="https://github.com/user-attachments/assets/9369752f-da36-4017-8af3-cbebe6037e2e" />

----
### Step 12: Merging Final Flood Maps

After processing each segment individually to ensure high precision, this final step mosaics all the small inundation rasters back into a single, study-area-wide flood map. This master TIFF file is ready for use in professional GIS software (like QGIS or ArcGIS) for final presentation and risk assessment.



```python
import flowpathflood as ff

# 1. Define input directory and final output name
segments_dir = "flowpath_segments"
final_output = "results/urban_flood_master_map.tif"

# 2. Execute the Merge
# This scans all subfolders and combines the results into a seamless mosaic
ff.merge_flood_map(
    base_dir=segments_dir, 
    output_filename=final_output
)
```

#### Expected Terminal Output:

<img width="768" height="790" alt="08" src="https://github.com/user-attachments/assets/e3a75f6e-386c-448f-9793-6f938c88cc58" />

---
