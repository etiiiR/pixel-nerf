import vtk
import numpy as np
import os
import argparse
import math
import sys # Import sys for sys.exit

# --- Helper Functions ---

def get_fibonacci_sphere_points(samples):
    """Generates points on a sphere using the Fibonacci lattice method."""
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)

def vtk_matrix_to_numpy(vtk_matrix):
    """Converts a VTK 4x4 matrix to a NumPy array."""
    mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            mat[i, j] = vtk_matrix.GetElement(i, j)
    return mat

def make_dirs(path):
    """Creates directories if they don't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

# --- Core Dataset Generation Logic for a Single Mesh ---

def generate_dataset_for_single_mesh(stl_file_path, args):
    """Generates the dataset structure for a single input STL file."""
    print(f"\n--- Processing Mesh: {os.path.basename(stl_file_path)} ---")

    # --- 1. Load STL and Get Bounds ---
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file_path)
    try:
        reader.Update()
    except Exception as e:
        print(f"VTK Error reading STL file {stl_file_path}: {e}")
        return # Skip this file

    polydata = reader.GetOutput()

    if not polydata or polydata.GetNumberOfPoints() == 0:
        print(f"Error: Could not read STL file or STL file is empty: {stl_file_path}")
        return # Skip this file

    bounds = polydata.GetBounds()
    center = np.array(polydata.GetCenter())
    print(f"Original Bounds: {bounds}")
    print(f"Original Center: {center}")

    # --- 2. Calculate Transform (Center and Scale) ---
    translate_transform = vtk.vtkTransform()
    translate_transform.Translate(-center[0], -center[1], -center[2])

    dims = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    dims[dims <= 0] = 1e-6 # Replace zero or negative dimensions with a tiny number
    max_dim = np.max(dims)

    scale_factor = args.target_size / max_dim
    print(f"Scaling factor: {scale_factor} (Max dim: {max_dim})")

    scale_transform = vtk.vtkTransform()
    scale_transform.Scale(scale_factor, scale_factor, scale_factor)

    combined_transform = vtk.vtkTransform()
    combined_transform.Concatenate(scale_transform)
    combined_transform.Concatenate(translate_transform)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(combined_transform)
    transform_filter.Update()

    transformed_polydata = transform_filter.GetOutput()
    transformed_bounds = transformed_polydata.GetBounds()
    print(f"Transformed Bounds: {transformed_bounds}")

    # --- 3. Setup VTK Rendering Pipeline ---
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transform_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(args.bg_color[0], args.bg_color[1], args.bg_color[2])

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(args.img_width, args.img_height)
    render_window.SetOffScreenRendering(1)

    # --- 4. Setup Camera and Calculate Base Intrinsics Values ---
    camera = renderer.GetActiveCamera()
    camera.ParallelProjectionOff()
    camera.SetViewAngle(args.view_angle)

    fov_y_rad = np.deg2rad(args.view_angle)
    tan_half_fovy = np.tan(fov_y_rad / 2.0)
    focal_length_y = args.img_height / (2.0 * tan_half_fovy) if tan_half_fovy > 1e-6 else 0
    focal_length_x = focal_length_y # Assume square pixels
    cx = args.img_width / 2.0
    cy = args.img_height / 2.0
    width = args.img_width
    height = args.img_height

    print(f"Calculated Intrinsics Values: fx={focal_length_x:.8f}, fy={focal_length_y:.8f}, cx={cx:.8f}, cy={cy:.8f}, w={width}, h={height}")

    # Estimate camera distance
    max_scaled_dim = np.max([transformed_bounds[1]-transformed_bounds[0],
                           transformed_bounds[3]-transformed_bounds[2],
                           transformed_bounds[5]-transformed_bounds[4]])
    if max_scaled_dim <= 0: max_scaled_dim = args.target_size

    cam_distance = (max_scaled_dim / 2.0) / tan_half_fovy if tan_half_fovy > 1e-6 else max_scaled_dim * 2.0
    cam_distance *= args.distance_factor
    print(f"Calculated Camera Distance: {cam_distance} (Based on max scaled dim: {max_scaled_dim})")
    

    # --- 5. Generate Camera Poses ---
    camera_positions_unit = get_fibonacci_sphere_points(args.num_views)
    camera_positions = camera_positions_unit * cam_distance

    # Get object ID
    object_id = os.path.splitext(os.path.basename(stl_file_path))[0]

    # --- 6. Prepare Splits ---
    total_views = args.num_views
    num_train = int(total_views * args.train_split)
    num_val = int(total_views * args.val_split)
    num_test = total_views - num_train - num_val

    print(f"Total views: {total_views}, Train: {num_train}, Val: {num_val}, Test: {num_test}")

    indices = np.random.permutation(total_views)
    split_map = {}
    for i, original_idx in enumerate(indices):
        if i < num_train:
            split_map[original_idx] = "train"
        elif i < num_train + num_val:
            split_map[original_idx] = "val"
        else:
            split_map[original_idx] = "test"

    # The main output directory provided by the user
    main_output_dir = args.output_dir
    make_dirs(main_output_dir) # Ensure the main output directory exists

    # --- 7. Create Intrinsics File for Each Split ---
    #    (Intrinsics are placed inside the object_id folder under the split folder)
    for split_name in ["train", "val", "test"]:
        num_in_split = 0
        if split_name == "train": num_in_split = num_train
        elif split_name == "val": num_in_split = num_val
        else: num_in_split = num_test

        if num_in_split > 0:
            # MODIFIED PATH: No longer includes the extra 'pollen_{object_id}' level
            split_object_dir = os.path.join(main_output_dir, f"pollen_{split_name}", object_id)
            make_dirs(split_object_dir) # Create pollen_train/object_id etc.
            intrinsics_path = os.path.join(split_object_dir, "intrinsics.txt")

            try:
                with open(intrinsics_path, 'w') as f:
                    f.write(f"{focal_length_x:.8f} {cx:.8f} {cy:.8f} 0.\n")
                    f.write("0. 0. 0.\n")
                    f.write("1.\n")
                    f.write(f"{width} {height}\n")
                print(f"Saved intrinsics (custom format) to {intrinsics_path}")
            except IOError as e:
                print(f"Error writing intrinsics to {intrinsics_path}: {e}")

    # --- 8. Loop Through Views, Render, Save Pose and Image ---
    view_counters = {"train": 0, "val": 0, "test": 0}
    image_writer = vtk.vtkPNGWriter()
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.ReadFrontBufferOff()
    image_writer.SetInputConnection(window_to_image.GetOutputPort())

    for i in range(total_views):
        original_view_idx = i
        split = split_map[original_view_idx]
        view_idx_in_split = view_counters[split]

        pos = camera_positions[original_view_idx]

        camera.SetPosition(pos[0], pos[1], pos[2])
        camera.SetFocalPoint(0, 0, 0)
        view_up = [0.0, 1.0, 0.0]
        cam_direction = np.array([0,0,0]) - np.array(pos) # Direction from pos to origin
        cam_direction = cam_direction / np.linalg.norm(cam_direction) if np.linalg.norm(cam_direction) > 0 else np.array([0,0,-1])
        if abs(np.dot(cam_direction, view_up)) > 0.995: # Check if view direction is too close to view_up
            view_up = [0.0, 0.0, 1.0 if pos[1] > 0 else -1.0] # Use Z-up/down if looking along Y
        camera.SetViewUp(view_up[0], view_up[1], view_up[2])


        renderer.ResetCameraClippingRange()
        render_window.Render()
        
        # 1) grab VTK’s own choice
        near_vtk, far_vtk = camera.GetClippingRange()

        # 2) (optionally) override with your own sphere‐based choice
        r = max_scaled_dim / 2.0
        near_sph = max(cam_distance - r, 1e-3)
        far_sph  = cam_distance + r
        camera.SetClippingRange(near_sph, far_sph)

        # pick which you like better…
        near, far = near_sph, far_sph  



        vtk_world_to_cam_matrix = camera.GetViewTransformMatrix()
        vtk_cam_to_world_matrix = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(vtk_world_to_cam_matrix, vtk_cam_to_world_matrix)
        pose_matrix_np = vtk_matrix_to_numpy(vtk_cam_to_world_matrix)
        pose_matrix_flat = pose_matrix_np.flatten()

        # MODIFIED PATH: Base path for pose/rgb is now directly under the split folder
        current_output_base = os.path.join(main_output_dir, f"pollen_{split}", object_id)
                # Save them out:
        with open(os.path.join(current_output_base, "near_far.txt"), "w") as f:
            f.write(f"{near:.8f} {far:.8f}\n")
        pose_dir = os.path.join(current_output_base, "pose")
        rgb_dir = os.path.join(current_output_base, "rgb")
        make_dirs(pose_dir) # Ensure they exist (might be created by intrinsics, but make sure)
        make_dirs(rgb_dir)

        pose_filename = os.path.join(pose_dir, f"{view_idx_in_split:08d}.txt")
        rgb_filename = os.path.join(rgb_dir, f"{view_idx_in_split:08d}.png")

        try:
             np.savetxt(pose_filename, pose_matrix_flat[np.newaxis, :], fmt='%.17g', delimiter=' ')
        except IOError as e:
             print(f"Error writing pose to {pose_filename}: {e}")

        window_to_image.Modified()
        window_to_image.Update()
        image_writer.SetFileName(rgb_filename)
        try:
            image_writer.Write()
        except IOError as e:
             print(f"Error writing image to {rgb_filename}: {e}")

        if (i + 1) % 20 == 0 or i == total_views - 1 :
             # Slightly adjust print to be less verbose about path, just show relative file
            print(f"  Saved view {original_view_idx+1}/{total_views} (Split: {split}, Idx: {view_idx_in_split}) -> {os.path.join('rgb', os.path.basename(rgb_filename))}, {os.path.join('pose', os.path.basename(pose_filename))}")

        view_counters[split] += 1

    # MODIFIED PRINT: Reflect the new structure
    print(f"--- Finished Mesh: {os.path.basename(stl_file_path)} ---")
    print(f"--- Output saved under: {main_output_dir}/pollen_train|val|test/{object_id}/ ---")


# --- Main Execution Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PixelNeRF-style dataset (Split/Object ID structure) from all STL files in a directory using VTK.")
    parser.add_argument("--mesh_dir", type=str, required=True, help="Path to the directory containing input STL files.")
    parser.add_argument("--output_dir", type=str, default="pollen_dataset_structured", help="Base directory to save the structured datasets (e.g., <output_dir>/pollen_train/object_id/).")
    parser.add_argument("--num_views", type=int, default=100, help="Total number of views to generate per mesh.")
    parser.add_argument("--img_width", type=int, default=128, help="Width of the rendered images.")
    parser.add_argument("--img_height", type=int, default=128, help="Height of the rendered images.")
    parser.add_argument("--view_angle", type=float, default=51.93, help="Camera view angle (FOV) in degrees (~51.93 for fx=131.25, w=128).")
    parser.add_argument("--target_size", type=float, default=1.5, help="Target size for the longest dimension after scaling.")
    parser.add_argument("--distance_factor", type=float, default=1.8, help="Multiplier for camera distance ( > 1 means further away).")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of views for the training set.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of views for the validation set.")
    parser.add_argument("--bg_color", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Background color (R G B, values 0-1).")

    args = parser.parse_args()

    if not os.path.isdir(args.mesh_dir):
        print(f"Error: Provided mesh directory does not exist or is not a directory: {args.mesh_dir}")
        sys.exit(1)

    if args.train_split + args.val_split > 1.0:
        print("Error: Sum of train_split and val_split cannot exceed 1.0")
        sys.exit(1)


    files_processed = 0
    files_skipped = 0
    print(f"Searching for STL files in: {args.mesh_dir}")
    print(f"Output will be generated in: {args.output_dir}")
    stl_files = [f for f in os.listdir(args.mesh_dir) if f.lower().endswith(".stl")]

    if not stl_files:
        print(f"No STL files found in {args.mesh_dir}")
        sys.exit(0)

    print(f"Found {len(stl_files)} STL file(s). Starting processing...")

    for filename in stl_files:
        current_stl_path = os.path.join(args.mesh_dir, filename)
        try:
            generate_dataset_for_single_mesh(current_stl_path, args)
            files_processed += 1
        except Exception as e:
            print(f"!!! Critical Error processing file {filename}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            print("!!! Skipping this file and continuing with the next...")
            files_skipped += 1

    print(f"\n\n=== Processing Summary ===")
    print(f"Successfully processed: {files_processed} STL file(s)")
    print(f"Skipped due to errors: {files_skipped} STL file(s)")
    print(f"Dataset output generated in: {args.output_dir}")
    print(f"Structure: {args.output_dir}/pollen_train|val|test/<object_id>/[intrinsics.txt, pose/, rgb/]")
    print(f"==========================")