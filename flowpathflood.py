
import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import re
from collections import deque
from shapely.geometry import LineString, shape
from rasterio.features import shapes
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.merge import merge
from pysheds.grid import Grid
from scipy.optimize import fsolve

# ==========================================
# PHASE 1: PREPROCESSING
# ==========================================

def split_streams_by_node(input_shapefile, output_dir="output_streams"):
    """
    Reads a stream shapefile and splits it into individual files based on nodeid.
    """
    os.makedirs(output_dir, exist_ok=True)
    streams = gpd.read_file(input_shapefile, on_invalid="ignore")

    # Cleaning
    streams['dstrnodeid'] = streams['dstrnodeid'].fillna(-1).astype(int)
    streams = streams[streams['geometry'].notnull() & streams.is_valid]

    if streams.empty:
        raise ValueError("The input shapefile contains no valid features.")

    unique_nodeids = streams['nodeid'].unique()
    saved_files = []

    for nodeid in unique_nodeids:
        filtered = streams[streams['nodeid'] == nodeid]
        out_path = os.path.join(output_dir, f"stream_nodeid_{nodeid}.shp")
        filtered.to_file(out_path)
        saved_files.append(out_path)

    return saved_files

def convert_d8_encoding(input_raster, output_path):
    """
    Converts CLSA encoding to Pysheds-compatible FDIR encoding.
    """
    with rasterio.open(input_raster) as src:
        flow_dir = src.read(1)
        meta = src.meta.copy()

    # Your mapping logic
    conversion_dict = {0: 64, 1: 128, 2: 1, 3: 2, 4: 4, 5: 8, 6: 16, 7: 32, 8: 0}
    output = np.zeros_like(flow_dir)

    for old_val, new_val in conversion_dict.items():
        output[flow_dir == old_val] = new_val

    meta.update(dtype=rasterio.int32)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(output.astype(np.int32), 1)

    return output, meta

def extract_stream_fdir(fdir_raster_path, stream_shp_dir, output_raster_dir="output_rasters", verbose=True):
    """
    Masks the flow direction raster using stream shapefiles and prints statistics.
    """
    os.makedirs(output_raster_dir, exist_ok=True)

    with rasterio.open(fdir_raster_path) as src:
        fdir = src.read(1)
        meta = src.meta.copy()
        transform = src.transform

    direction_names = {
        1: "E", 2: "SE", 4: "S", 8: "SW",
        16: "W", 32: "NW", 64: "N", 128: "NE"
    }

    for shapefile in os.listdir(stream_shp_dir):
        if not shapefile.endswith('.shp'):
            continue

        nodeid = os.path.splitext(shapefile)[0].split('_')[-1]
        streams = gpd.read_file(os.path.join(stream_shp_dir, shapefile))

        if streams.empty:
            continue

        # Rasterize
        stream_mask = rasterize(
            [(geom, 1) for geom in streams.geometry],
            out_shape=fdir.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        stream_fdir = np.where(stream_mask == 1, fdir, 0)

        # Save output
        out_path = os.path.join(output_raster_dir, f'stream_fdir_nodeid_{nodeid}.tif')
        meta.update(dtype=rasterio.int32, nodata=0)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(stream_fdir.astype(np.int32), 1)

        # --- Your Statistics Code ---
        if verbose:
            unique_values, counts = np.unique(stream_fdir[stream_fdir > 0], return_counts=True)
            print(f"\nFlow Direction Statistics for nodeid={nodeid}:")
            for value, count in zip(unique_values, counts):
                direction = direction_names.get(value, "Unknown")
                print(f"   Direction {direction} ({value}): {count} pixels")
            print(f"   Total stream pixels: {np.sum(stream_mask)}")
            print(f"   Saved raster: {out_path}")

    print("\nProcessing complete! All flow direction rasters have been saved.")

from collections import deque
from shapely.geometry import LineString
import geopandas as gpd

def segment_flowpaths(input_dir, base_output_dir='flowpath_segments', segment_size=110):
    """
    Divides flow path rasters into linear segments of a fixed pixel size.
    """
    os.makedirs(base_output_dir, exist_ok=True)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def bfs(start_pixel, visited, flowpath_binary, nrows, ncols):
        queue = deque([start_pixel])
        pixels = []
        while queue and len(pixels) < segment_size:
            r, c = queue.popleft()
            if not visited[r, c]:
                visited[r, c] = True
                pixels.append((r, c))
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < nrows and 0 <= nc < ncols and flowpath_binary[nr, nc] == 1 and not visited[nr, nc]:
                        queue.append((nr, nc))
        return pixels

    for flowpath_file in os.listdir(input_dir):
        if not flowpath_file.endswith('.tif'): continue

        nodeid = flowpath_file.split("_")[-1].replace(".tif", "")
        nodeid_dir = os.path.join(base_output_dir, f'nodeid_{nodeid}')
        tiff_dir = os.path.join(nodeid_dir, 'tiff_segments')
        os.makedirs(tiff_dir, exist_ok=True)

        with rasterio.open(os.path.join(input_dir, flowpath_file)) as src:
            flowpath_binary = (src.read(1) > 0).astype(int)
            transform, crs, meta = src.transform, src.crs, src.meta
            nrows, ncols = flowpath_binary.shape

        visited = np.zeros_like(flowpath_binary, dtype=bool)
        flowpath_pixels = np.argwhere(flowpath_binary == 1)
        segment_id = 1
        segments_data = []

        for pixel in flowpath_pixels:
            if visited[pixel[0], pixel[1]]: continue

            pixels = bfs(tuple(pixel), visited, flowpath_binary, nrows, ncols)
            if len(pixels) > 1:
                coords = [rasterio.transform.xy(transform, r, c) for r, c in pixels]
                segments_data.append({'segment_id': segment_id, 'geometry': LineString(coords)})

                # Save Segment Raster
                seg_raster = np.zeros((nrows, ncols), dtype=np.uint8)
                for r, c in pixels: seg_raster[r, c] = 1

                out_meta = meta.copy()
                out_meta.update({'dtype': 'uint8', 'count': 1, 'compress': 'lzw'})
                with rasterio.open(os.path.join(tiff_dir, f'flowpath_segment_{segment_id}.tif'), 'w', **out_meta) as dst:
                    dst.write(seg_raster, 1)
                segment_id += 1

        gdf = gpd.GeoDataFrame(segments_data, crs=crs)
        gdf.to_file(os.path.join(nodeid_dir, f'flowpath_segments_nodeid_{nodeid}.shp'))
    print("Segmentation complete.")

def calculate_segment_slopes(dem_path, base_dir='flowpath_segments'):
    """
    Calculates the average slope for each segment using the provided DEM.
    """
    with rasterio.open(dem_path) as src:
        dem, transform, dem_crs = src.read(1), src.transform, src.crs

    for nodeid_folder in os.listdir(base_dir):
        nodeid_path = os.path.join(base_dir, nodeid_folder)
        if not os.path.isdir(nodeid_path): continue

        shp_path = os.path.join(nodeid_path, f'flowpath_segments_{nodeid_folder}.shp')
        if os.path.exists(shp_path):
            gdf = gpd.read_file(shp_path).to_crs(dem_crs)
            slope_results = []

            for _, row in gdf.iterrows():
                px = [rasterio.transform.rowcol(transform, *c) for c in row['geometry'].coords]
                elevs = [dem[r, c] for r, c in px if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]]

                slope = (max(elevs) - min(elevs)) / len(elevs) if len(elevs) > 1 else 0
                slope_results.append(f"{row['segment_id']},{slope:.6f}")

            with open(os.path.join(nodeid_path, 'Slopes.txt'), 'w') as f:
                f.write("Segment_id,Slope\n" + "\n".join(slope_results))
    print("Slope calculation complete.")

def create_buffered_environmental_data(dem_file, fdir_file, base_dir='flowpath_segments', buffer_dist=10):
    """
    Buffers flow paths and crops DEM/FDIR rasters to that buffer.
    """
    for nodeid_folder in os.listdir(base_dir):
        nodeid_path = os.path.join(base_dir, nodeid_folder)
        tiff_dir = os.path.join(nodeid_path, 'tiff_segments')
        if not os.path.exists(tiff_dir): continue

        dem_out = os.path.join(nodeid_path, 'cropped_dems')
        fdir_out = os.path.join(nodeid_path, 'cropped_fdir')
        os.makedirs(dem_out, exist_ok=True)
        os.makedirs(fdir_out, exist_ok=True)

        for f in os.listdir(tiff_dir):
            if not f.endswith('.tif'): continue
            seg_id = f.split('_')[-1].replace('.tif', '')

            with rasterio.open(os.path.join(tiff_dir, f)) as src:
                data, crs, trans = src.read(1), src.crs, src.transform
                geoms = [shape(g).buffer(buffer_dist) for g, v in shapes(data, mask=(data==1), transform=trans)]

            if not geoms: continue

            # Save Buffer Shapefile
            buf_gdf = gpd.GeoDataFrame(geometry=[gpd.GeoSeries(geoms).unary_union], crs=crs)
            buf_gdf.to_file(os.path.join(nodeid_path, f'buffer_seg_{seg_id}.shp'))

            # Crop Environmental Rasters
            for inp, outp, name in [(dem_file, dem_out, 'dem'), (fdir_file, fdir_out, 'fdir')]:
                with rasterio.open(inp) as src:
                    out_img, out_trans = mask(src, buf_gdf.geometry, crop=True)
                    meta = src.meta.copy()
                    meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_trans})
                    with rasterio.open(os.path.join(outp, f'crop_{name}_{seg_id}.tif'), 'w', **meta) as dst:
                        dst.write(out_img)
    print("Buffering and Cropping complete.")

def run_preprocessing(dem_file, fdir_file, streams_shp, seg_size=110, buff_dist=10):
    """
    Groups all 6 preprocessing steps into one command.
    """
    print("🚀 Starting Preprocessing Pipeline...")

    # Step 1: Split the shapefile into individual stream nodes
    split_streams_by_node(streams_shp, output_dir="output_streams")

    # Step 2: Convert the original Flow Direction to Pysheds format
    convert_d8_encoding(fdir_file, "fdir_pysheds.tif")

    # Step 3: Extract Flow Direction only for the stream paths
    extract_stream_fdir("fdir_pysheds.tif", "output_streams", "output_rasters")

    # Step 4: Divide the flow paths into 110-pixel segments
    segment_flowpaths(input_dir="output_rasters", segment_size=seg_size)

    # Step 5: Calculate the slope (rise/run) for every segment
    calculate_segment_slopes(dem_path=dem_file)

    # Step 6: Create 10m buffers and crop the DEM/FDIR rasters to them
    create_buffered_environmental_data(
        dem_file=dem_file,
        fdir_file="fdir_pysheds.tif",
        buffer_dist=buff_dist
    )

    print("\n✅ Preprocessing Complete! Your data is ready for HAND.")

# ==========================================
# PHASE 2: HAND ANALYSIS
# ==========================================

def compute_hand_for_segments(base_dir='flowpath_segments'):
    """
    Computes HAND (Height Above Nearest Drainage) for each segment.
    """
    for nodeid_folder in os.listdir(base_dir):
        nodeid_path = os.path.join(base_dir, nodeid_folder)
        if not os.path.isdir(nodeid_path): continue

        # Internal directories created in previous steps
        dem_folder = os.path.join(nodeid_path, 'cropped_dems')
        fdir_folder = os.path.join(nodeid_path, 'cropped_fdir')
        flowpath_folder = os.path.join(nodeid_path, 'tiff_segments')

        if not all(os.path.exists(d) for d in [dem_folder, fdir_folder, flowpath_folder]):
            continue

        output_folder = os.path.join(nodeid_path, 'HAND_output')
        os.makedirs(output_folder, exist_ok=True)

        dem_files = sorted([f for f in os.listdir(dem_folder) if f.endswith('.tif')])
        fdir_files = sorted([f for f in os.listdir(fdir_folder) if f.endswith('.tif')])
        flowpath_files = sorted([f for f in os.listdir(flowpath_folder) if f.endswith('.tif')])

        for d_f, fd_f, fp_f in zip(dem_files, fdir_files, flowpath_files):
            grid = Grid.from_raster(os.path.join(dem_folder, d_f))
            dem = grid.read_raster(os.path.join(dem_folder, d_f))
            fdir = grid.read_raster(os.path.join(fdir_folder, fd_f))
            stream_fdir = grid.read_raster(os.path.join(flowpath_folder, fp_f))

            hand = grid.compute_hand(fdir, dem, stream_fdir)
            hand = np.clip(hand, 0, None)

            # Save
            out_path = os.path.join(output_folder, f'HAND_{d_f}')
            with rasterio.open(os.path.join(dem_folder, d_f)) as src:
                meta = src.meta.copy()
            meta.update({'dtype': 'float32', 'nodata': -9999})
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(hand.astype('float32'), 1)
    print("HAND calculation complete.")

def generate_inundation_extents(base_dir='flowpath_segments', constant_depth=1.0):
    """
    Generates inundation rasters based on a constant water depth.
    """
    for nodeid_folder in os.listdir(base_dir):
        nodeid_path = os.path.join(base_dir, nodeid_folder)
        hand_folder = os.path.join(nodeid_path, 'HAND_output')
        if not os.path.exists(hand_folder): continue

        out_folder = os.path.join(nodeid_path, 'Inundation_output')
        os.makedirs(out_folder, exist_ok=True)

        for hand_file in [f for f in os.listdir(hand_folder) if f.endswith('.tif')]:
            with rasterio.open(os.path.join(hand_folder, hand_file)) as src:
                hand_data = np.clip(src.read(1), 0, None)
                meta = src.meta.copy()

            # Water depth = constant_depth - HAND (where HAND < depth)
            inundation = np.where(hand_data < constant_depth, constant_depth - hand_data, np.nan)

            out_path = os.path.join(out_folder, f'Inundation_{hand_file}')
            meta.update({'dtype': 'float32', 'nodata': -9999})
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(inundation.astype('float32'), 1)
    print(f"Inundation mapping complete (Depth: {constant_depth}m).")

def calculate_inundation_stats(base_dir='flowpath_segments', constant_depth=1.0, pixel_res=1.0):
    """
    Calculates Area, Length, and Width for each inundated segment.
    """
    cell_area = pixel_res * pixel_res

    for nodeid_folder in os.listdir(base_dir):
        nodeid_path = os.path.join(base_dir, nodeid_folder)
        hand_folder = os.path.join(nodeid_path, 'HAND_output')
        if not os.path.exists(hand_folder): continue

        # Find the centerline shapefile
        shp_files = [f for f in os.listdir(nodeid_path) if f.endswith('.shp') and 'segments' in f]
        if not shp_files: continue
        gdf = gpd.read_file(os.path.join(nodeid_path, shp_files[0]))

        csv_rows = []
        for hand_file in [f for f in os.listdir(hand_folder) if f.endswith('.tif')]:
            # Extract Segment ID
            match = re.search(r'_(\d+)\.tif$', hand_file)
            if not match: continue
            seg_id = int(match.group(1))

            with rasterio.open(os.path.join(hand_folder, hand_file)) as src:
                hand_data = np.clip(src.read(1), 0, None)

            inundated_cells = np.count_nonzero(hand_data < constant_depth)
            area = inundated_cells * cell_area

            # Calculate Length from Shapefile
            seg_geom = gdf[gdf['segment_id'] == seg_id].geometry.iloc[0]
            coords = np.array(seg_geom.coords)
            length = 0
            for i in range(1, len(coords)):
                d = np.linalg.norm(coords[i] - coords[i-1])
                length += d

            width = area / length if length > 0 else 0
            csv_rows.append([seg_id, length, area, width])

        # Save results
        out_res_folder = os.path.join(nodeid_path, 'Inundation_results')
        os.makedirs(out_res_folder, exist_ok=True)
        header = "Segment_id,Segment_length,Inundation_area,Width"
        np.savetxt(os.path.join(out_res_folder, 'Inundation_results.csv'),
                   csv_rows, delimiter=",", fmt="%.6f", header=header, comments="")
    print("Inundation statistics calculation complete.")

def run_hand_analysis(base_dir='flowpath_segments', depth=1.0, resolution=1.0):
    """
    Master function for the HAND analysis phase.
    1. Computes HAND rasters.
    2. Generates inundation extents for a given depth.
    3. Calculates Area, Length, and Width statistics.
    """
    print(f"🌊 Starting HAND Analysis (Constant Depth: {depth}m)...")

    # Step 7: Compute HAND
    compute_hand_for_segments(base_dir=base_dir)

    # Step 8: Generate Inundation Rasters
    generate_inundation_extents(base_dir=base_dir, constant_depth=depth)

    # Step 9: Calculate Geometric Statistics (CSV output)
    calculate_inundation_stats(base_dir=base_dir, constant_depth=depth, pixel_res=resolution)

    print("\n✅ HAND Analysis Phase Complete.")

# ==========================================
# PHASE 3: MANNING ANALYSIS
# ==========================================

def solve_manning_depths(q_lookup_file, base_dir='flowpath_segments', n_roughness=0.20):
    """
    Calculates hydraulic water depth (h) for each segment using Manning's Equation.
    """
    q_lookup_df = pd.read_csv(q_lookup_file)

    for nodeid_folder in os.listdir(base_dir):
        nodeid_path = os.path.join(base_dir, nodeid_folder)
        if not os.path.isdir(nodeid_path): continue

        # 1. Get NodeID and Discharge (Q)
        nodeid = int(nodeid_folder.split('_')[-1])
        q_row = q_lookup_df[q_lookup_df['nodeid'] == nodeid]
        if q_row.empty: continue
        Q = q_row['q'].values[0]

        # 2. Load Slopes and Geometry Widths
        slope_file = os.path.join(nodeid_path, 'Slopes.txt')
        inund_file = os.path.join(nodeid_path, 'Inundation_results', 'Inundation_results.csv')
        if not (os.path.exists(slope_file) and os.path.exists(inund_file)): continue

        slopes_df = pd.read_csv(slope_file)
        widths_df = pd.read_csv(inund_file)

        water_depths = []
        for _, row in slopes_df.iterrows():
            seg_id = row['Segment_id']
            slope = row['Slope']

            w_row = widths_df[widths_df['Segment_id'] == seg_id]
            if w_row.empty or slope <= 0:
                water_depths.append((seg_id, 0))
                continue

            width = w_row['Width'].values[0]

            # Manning's Solver
            def manning_eq(h):
                if h <= 0: return Q
                A = width * h
                P = width + 2 * h
                R = A / P
                # Q = (1/n) * A * R^(2/3) * S^(1/2)
                return Q - (1 / n_roughness) * A * (R ** (2 / 3)) * np.sqrt(slope)

            h_sol = fsolve(manning_eq, 0.1)[0]
            water_depths.append((seg_id, max(h_sol, 0)))

        # Save Results
        out_df = pd.DataFrame(water_depths, columns=['Segment_id', 'Water_Depth'])
        out_df.to_csv(os.path.join(nodeid_path, 'Water_Depths.csv'), index=False)

    print("Manning Depth solver complete.")

def map_hydraulic_inundation(base_dir='flowpath_segments'):
    """
    Creates inundation rasters using calculated Manning depths per segment.
    """
    for nodeid_folder in os.listdir(base_dir):
        nodeid_path = os.path.join(base_dir, nodeid_folder)
        depth_file = os.path.join(nodeid_path, 'Water_Depths.csv')
        hand_folder = os.path.join(nodeid_path, 'HAND_output')

        if not (os.path.exists(depth_file) and os.path.exists(hand_folder)): continue

        depths_dict = pd.read_csv(depth_file).set_index('Segment_id')['Water_Depth'].to_dict()
        out_folder = os.path.join(nodeid_path, 'Inundation_output')
        os.makedirs(out_folder, exist_ok=True)

        for hand_file in [f for f in os.listdir(hand_folder) if f.endswith('.tif')]:
            seg_id = int(hand_file.split('_')[-1].replace('.tif', ''))
            water_depth = depths_dict.get(seg_id, 0)

            with rasterio.open(os.path.join(hand_folder, hand_file)) as src:
                hand_data = np.clip(src.read(1), 0, None)
                meta = src.meta.copy()

            # Apply Manning Depth
            inundation = np.where(hand_data < water_depth, water_depth - hand_data, np.nan)

            meta.update({'dtype': 'float32', 'nodata': np.nan})
            with rasterio.open(os.path.join(out_folder, f'Inundation_seg_{seg_id}.tif'), 'w', **meta) as dst:
                dst.write(inundation.astype('float32'), 1)
    print("Hydraulic inundation mapping complete.")

def merge_flood_map(base_dir='flowpath_segments', output_filename='final_flood_map.tif'):
    """
    Merges all segment inundation rasters into a single study-area-wide flood map.
    """
    tiff_files = []
    for root, _, files in os.walk(base_dir):
        if 'Inundation_output' in root:
            tiff_files.extend([os.path.join(root, f) for f in files if f.endswith('.tif')])

    if not tiff_files:
        print("No inundation files found to merge.")
        return

    datasets = [rasterio.open(f) for f in tiff_files]
    mosaic, out_trans = merge(datasets, nodata=np.nan)

    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "nodata": np.nan,
        "compress": "lzw"
    })

    with rasterio.open(output_filename, 'w', **out_meta) as dest:
        dest.write(mosaic[0].astype(np.float32), 1)

    for ds in datasets: ds.close()
    print(f"Final flood map saved as: {output_filename}")

def run_manning_analysis(q_lookup_file, base_dir='flowpath_segments', n_roughness=0.20, output_name='final_flood_map.tif'):
    """
    Master function for the Manning hydraulic analysis phase.
    1. Solves Manning's equation for each segment to find specific depth (h).
    2. Maps inundation using the calculated depths.
    3. Merges all segment rasters into a single final study-area map.
    """
    print(f"🌊 Starting Manning's Hydraulic Analysis (Roughness n: {n_roughness})...")

    # Step 10: Solve Manning's Equation for h
    solve_manning_depths(q_lookup_file, base_dir=base_dir, n_roughness=n_roughness)

    # Step 11: Create specific inundation rasters per segment
    map_hydraulic_inundation(base_dir=base_dir)

    # Step 12: Merge segments into a single mosaic
    merge_flood_map(base_dir=base_dir, output_filename=output_name)

    print(f"\n✅ Manning Analysis Phase Complete. Final map saved as: {output_name}")