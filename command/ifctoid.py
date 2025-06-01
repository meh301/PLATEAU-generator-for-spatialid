import argparse
import logging
import os
import typing

import ifcopenshell
import ifcopenshell.geom
import pyvista as pv
import numpy as np
from pyproj import Transformer
from grids import get as get_grid
from outputs import export_csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial

Any = typing.Any
Dict = typing.Dict
List = typing.List
Tuple = typing.Tuple

logger = logging.getLogger(__name__)

def get_transformer():
    return Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def apply_global_transform(verts: np.ndarray, offset: Tuple[float, float, float], z_rot_deg: float):
    # Translate
    verts += np.array(offset)
    # Rotate around Z
    angle_rad = np.radians(z_rot_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                 0,                1]
    ])
    return verts @ rot_matrix.T

def extract_geometry(ifc_path: str, offset: Tuple[float, float, float], z_rot_deg: float) -> List[Tuple[List[List[List[float]]], Dict[str, Any]]]:
    model = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    out = []
    element_types = [
        "IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof",
        "IfcDoor", "IfcWindow", "IfcColumn", "IfcBeam", "IfcMember",
        "IfcPlate", "IfcCovering", "IfcRailing", "IfcStair", "IfcStairFlight",
        "IfcRamp", "IfcCurtainWall", "IfcBuildingElementProxy", "IfcFurnishingElement"
    ]
    for etype in element_types:
        for el in model.by_type(etype):
            try:
                shape = ifcopenshell.geom.create_shape(settings, el)
                verts = np.array(shape.geometry.verts).reshape(-1, 3)
                # Convert face indices to VTK format (triangle count prefixed)
                raw_faces = shape.geometry.faces
                faces = []
                for i in range(0, len(raw_faces), 3):
                    faces.append([3, raw_faces[i], raw_faces[i+1], raw_faces[i+2]])
                faces = np.array([i for f in faces for i in f], dtype=np.int64)

                verts = apply_global_transform(verts, offset, z_rot_deg)
                poly = pv.PolyData(verts, faces)
                poly.triangulate(inplace=True)
                raw_name = getattr(el, "Name", None) or getattr(el, "ObjectType", None) or "Unnamed"
                try:
                    name = raw_name.encode("utf-8").decode("utf-8").replace(" ", "_")
                except Exception:
                    name = "Unnamed"
                name = name.replace(" ", "_") if name else "Unnamed"
                element_label = f"{el.is_a()}-{name}-{el.GlobalId}"
                out.append(([[list(pt) for pt in poly.points]], {"gml_id": element_label, "geom_dim": 3}))
                # print(f"[DEBUG] Extracted {el.is_a()} {el.GlobalId} with {len(poly.points)} points")
            except Exception as e:
                print(f"[WARN] Skipping element {el.GlobalId} due to error: {e}")
    print(f"[INFO] Extracted {len(out)} elements from {ifc_path}")
    return out


def process_ifc(input_path: str, output_path: str, lod: int, grid_type: str, grid_level: int,
                grid_size: List[float], grid_crs: int, offset: Tuple[float, float, float],
                z_rot_deg: float, interpolate: bool, merge: bool, debug: bool):
    grid = get_grid(grid_type, level=grid_level, size=grid_size, crs=grid_crs)
    grid.clear()
    data = extract_geometry(input_path, offset, z_rot_deg)
    grid.load_geom_data(data, interpolate=interpolate, merge=merge)
    export_csv(grid, output_path, merge=merge)

def _safe_process_file(args):
    try:
        process_ifc(*args)
    except Exception as e:
        print(f"[ERROR] Failed on {args[0]} -> {args[1]}")
        import traceback
        traceback.print_exc()
        raise

def main(input_path: str, output_path: str, lod: int, grid_type: str, grid_level: int,
         grid_size: List[float], grid_crs: int, offset: Tuple[float, float, float],
         z_rot_deg: float, interpolate: bool, merge: bool, debug: bool):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".ifc")]
    output_files = [os.path.join(output_path, os.path.splitext(os.path.basename(f))[0] + ".csv") for f in input_files]
    task_args = [
        (inp, outp, lod, grid_type, grid_level, grid_size, grid_crs, offset, z_rot_deg, interpolate, merge, debug)
        for inp, outp in zip(input_files, output_files)
    ]
    with ProcessPoolExecutor() as executor:
        executor.map(_safe_process_file, task_args)
    # for args in task_args:
    #     _safe_process_file(args)
    print("[INFO] Merging outputs...")
    # TODO: call outputs.consolidate_output_parts if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--lod", type=int, default=3)
    parser.add_argument("--grid-type", default="zfxy")
    parser.add_argument("--grid-level", type=int, default=28)
    parser.add_argument("--grid-size", type=float, nargs="*", default=[1.0])
    parser.add_argument("--grid-crs", type=int, default=3857)
    parser.add_argument("--offset", type=float, nargs=3, help="x y z offset in meters")
    parser.add_argument("--z-rot-deg", type=float, default=0.0)
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.lod, args.grid_type, args.grid_level,
         args.grid_size, args.grid_crs, tuple(args.offset), args.z_rot_deg, args.interpolate, args.merge, args.debug)
