import vtk
import numpy as np
import os
import argparse
import math
import sys  # Import sys for sys.exit

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
        return

    polydata = reader.GetOutput()
    if not polydata or polydata.GetNumberOfPoints() == 0:
        print(f"Error: Could not read STL file or STL file is empty: {stl_file_path}")
        return

    bounds = polydata.GetBounds()
    center = np.array(polydata.GetCenter())
    print(f"Original Bounds: {bounds}")
    print(f"Original Center: {center}")

    # --- 2. Center & Scale ---
    translate = vtk.vtkTransform()
    translate.Translate(-center[0], -center[1], -center[2])

    dims = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    dims[dims <= 0] = 1e-6
    max_dim = np.max(dims)
    scale_factor = args.target_size / max_dim
    print(f"Scaling factor: {scale_factor}")

    scale = vtk.vtkTransform()
    scale.Scale(scale_factor, scale_factor, scale_factor)

    combined = vtk.vtkTransform()
    combined.Concatenate(scale)
    combined.Concatenate(translate)

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(polydata)
    tf.SetTransform(combined)
    tf.Update()

    transformed_bounds = tf.GetOutput().GetBounds()
    print(f"Transformed Bounds: {transformed_bounds}")

    # --- 3. Rendering Pipeline ---
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tf.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
     # matte orange
    prop = actor.GetProperty()
    prop.SetColor(0.9, 0.4, 0.1)  # dunkleres, weicheres Orange
    prop.SetDiffuse(0.7)
    prop.SetSpecular(0.0)
    prop.SetAmbient(0.2)


    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(*args.bg_color)
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(args.img_width, args.img_height)
    window.SetOffScreenRendering(1)

    # --- 4. Camera & Intrinsics ---
    cam = renderer.GetActiveCamera()
    cam.ParallelProjectionOff()
    cam.SetViewAngle(args.view_angle)
    fov = math.radians(args.view_angle)
    tan2 = math.tan(fov/2)
    fy = args.img_height/(2*tan2)
    fx = fy
    cx = args.img_width/2.0
    cy = args.img_height/2.0
    w = args.img_width
    h = args.img_height
    print(f"fx={fx:.4f}, fy={fy:.4f}, cx={cx:.2f}, cy={cy:.2f}")

    # Estimate camera distance
    max_t = np.max([transformed_bounds[1]-transformed_bounds[0],
                    transformed_bounds[3]-transformed_bounds[2],
                    transformed_bounds[5]-transformed_bounds[4]])
    if max_t <= 0:
        max_t = args.target_size
    cam_dist = (max_t/2)/tan2*args.distance_factor
    print(f"Camera distance: {cam_dist:.4f}")

    # --- 5. Views & Splits ---
    dirs = get_fibonacci_sphere_points(args.num_views)
    positions = dirs * cam_dist
    total = args.num_views
    n_tr = int(total * args.train_split)
    n_val = int(total * args.val_split)
    n_te = total - n_tr - n_val

    out = args.output_dir
    make_dirs(out)
    obj = os.path.splitext(os.path.basename(stl_file_path))[0]

    # Write intrinsics (10 values total: 4+3+1+2)
    for name, count in zip(["train", "val", "test"], [n_tr, n_val, n_te]):
        if count <= 0:
            continue
        sd = os.path.join(out, f"pollen_{name}", obj)
        make_dirs(sd)
        ip = os.path.join(sd, "intrinsics.txt")
        with open(ip, 'w') as f:
            # fx cx cy 0.
            f.write(f"{fx:.8f} {cx:.8f} {cy:.8f} 0.\n")
            # 0. 0. 0.
            f.write("0. 0. 0.\n")
            # 1.
            f.write("1.\n")
            # width height
            f.write(f"{w} {h}\n")
        print(f"Saved intrinsics ({ip})")

    # Writer setup
    writer = vtk.vtkPNGWriter()
    to_img = vtk.vtkWindowToImageFilter()
    to_img.SetInput(window)
    to_img.ReadFrontBufferOff()
    writer.SetInputConnection(to_img.GetOutputPort())

    counters = {"train": 0, "val": 0, "test": 0}
    for i in range(total):
        if i < n_tr:
            split = "train"
        elif i < n_tr + n_val:
            split = "val"
        else:
            split = "test"
        idx = counters[split]

        pos = positions[i]
        cam.SetPosition(*pos)
        cam.SetFocalPoint(0, 0, 0)
        up = np.array([0, 1, 0])
        dv = -pos / np.linalg.norm(pos)
        if abs(np.dot(dv, up)) > 0.995:
            up = np.array([0, 0, 1 if pos[1] > 0 else -1])
        cam.SetViewUp(*up)

        renderer.ResetCameraClippingRange()
        window.Render()
        near, far = cam.GetClippingRange()
        near, far = 0.8, 1.8
        base = os.path.join(out, f"pollen_{split}", obj)
        with open(os.path.join(base, "near_far.txt"), 'w') as f:
            f.write(f"{near:.8f} {far:.8f}\n")

        # Pose W2C
        m = vtk_matrix_to_numpy(cam.GetModelViewTransformMatrix())
        pf = m.flatten()
        pd = os.path.join(base, "pose")
        rd = os.path.join(base, "rgb")
        make_dirs(pd)
        make_dirs(rd)
        np.savetxt(os.path.join(pd, f"{idx:08d}.txt"), pf[np.newaxis, :], fmt='%.17g')
        to_img.Modified()
        to_img.Update()
        imgf = os.path.join(rd, f"{idx:08d}.png")
        writer.SetFileName(imgf)
        writer.Write()
        counters[split] += 1
        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"Saved view {i+1}/{total} [{split}#{idx}]")

    print(f"Done {obj} -> {out}/pollen_train|val|test/{obj}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_dir", required=True)
    p.add_argument("--output_dir", default="pollen_dataset_structured")
    p.add_argument("--num_views", type=int, default=100)
    p.add_argument("--img_width", type=int, default=128)
    p.add_argument("--img_height", type=int, default=128)
    p.add_argument("--view_angle", type=float, default=51.93)
    p.add_argument("--target_size", type=float, default=1.5)
    p.add_argument("--distance_factor", type=float, default=1.8)
    p.add_argument("--train_split", type=float, default=0.8)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--bg_color", nargs=3, type=float, default=[1, 1, 1])
    args = p.parse_args()
    if not os.path.isdir(args.mesh_dir): sys.exit("mesh_dir not found")
    if args.train_split + args.val_split > 1.0: sys.exit("train+val >1.0")
    files = [f for f in os.listdir(args.mesh_dir) if f.lower().endswith('.stl')]
    if not files: sys.exit("no STL")
    ok = sk = 0
    for fn in files:
        try:
            generate_dataset_for_single_mesh(os.path.join(args.mesh_dir, fn), args)
            ok += 1
        except Exception as e:
            print(f"Error {fn}: {e}")
            sk += 1
    print(f"Processed {ok}, Skipped {sk}")
