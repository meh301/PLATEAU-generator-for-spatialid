"""
IFC to Spatial ID CSV generator with global yaw rotation and transformation to lat/lon/alt.
Author: Alex Orsholits / SpatialID 2025
"""

import argparse
import logging
import os
import typing

import numpy as np
import pyvista as pv
import ifcopenshell
import ifcopenshell.geom
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R
from grids import get as get_grid
from outputs import export_csv
from concurrent.futures import ProcessPoolExecutor

Any = typing.Any
Dict = typing.Dict
List = typing.List
Tuple = typing.Tuple
logger = logging.getLogger(__name__)


# Earth radius for Δlat/Δlon conversions (still used if you want the “manual” approach).
R_EARTH = 6378137.0


def build_yaw_matrix(yaw_deg: float) -> np.ndarray:
    return R.from_euler('z', yaw_deg, degrees=True).as_matrix()


def local_to_latlonalt(
    raw_pts: np.ndarray,
    origin_llh: Tuple[float, float, float],
    yaw_deg: float
) -> np.ndarray:
    """
    1) Apply yaw about IFC‐origin. 2) Treat result (dx, dy, dz) in metres as (East, North, Up)
       from (lat₀, lon₀, alt₀). 3) Δlat = (North/R)·(180/π), Δlon = (East/(R·cos(lat₀)))·(180/π).
       newZ = alt₀ + dz. 
       Return N×3 array of (lat, lon, alt).
    """
    lat0, lon0, alt0 = origin_llh
    lat0_rad = np.radians(lat0)
    yaw_mat = build_yaw_matrix(yaw_deg)
    rotated = (yaw_mat @ raw_pts.T).T   # shape = (N,3)

    out = np.zeros_like(rotated)       # we'll fill with (lat, lon, alt)
    for i, (dx, dy, dz) in enumerate(rotated):
        dlat = (dy / R_EARTH) * (180.0 / np.pi)
        dlon = (dx / (R_EARTH * np.cos(lat0_rad))) * (180.0 / np.pi)
        lat_v = lat0 + dlat
        lon_v = lon0 + dlon
        alt_v = alt0 + dz
        out[i, 0] = lat_v
        out[i, 1] = lon_v
        out[i, 2] = alt_v
    return out  # (N, 3)


def extract_geometry(
    ifc_path: str,
    origin_llh: Tuple[float, float, float],
    yaw_deg: float
) -> List[Tuple[List[List[List[float]]], Dict[str, Any]]]:
    """
    1) Open IFC; for each element, get raw_verts in IFC local‐metres.
    2) Convert to (lat, lon, alt) via local_to_latlonalt().
    3) Reproject each (lat, lon) → (x, y) in WebMercator (EPSG:3857).
    4) Build PolyData with (x, y, alt) and faces, triangulate, label with “gml_id”.
    5) Return that list to be voxelized.
    """
    model = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    # We will reproject (lon,lat)→(x, y) in EPSG:3857 *here*:
    mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    out: List[Tuple[List[List[List[float]]], Dict[str, Any]]] = []
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
                raw_verts = np.array(shape.geometry.verts).reshape(-1, 3)  # (N,3)

                # → Step 2: (lat, lon, alt) via small‐offset formula
                latlonalt_pts = local_to_latlonalt(raw_verts, origin_llh, yaw_deg)
                # latlonalt_pts[i] = [lat_i, lon_i, alt_i]

                # → Step 3: Reproject each (lon, lat) → (x, y) in metres
                XY_alt = []
                for (lat_i, lon_i, alt_i) in latlonalt_pts:
                    x3857, y3857 = mercator.transform(lon_i, lat_i)
                    XY_alt.append((x3857, y3857, alt_i))
                verts_3857 = np.array(XY_alt)  # shape (N,3)

                # Build faces exactly as before
                raw_faces = shape.geometry.faces
                faces = np.hstack([
                    [3, raw_faces[i], raw_faces[i+1], raw_faces[i+2]]
                    for i in range(0, len(raw_faces), 3)
                ])

                # Step 4: Build PolyData in “real WebMercator‐metres”
                poly = pv.PolyData(verts_3857, faces)
                poly.triangulate(inplace=True)

                # Step 5: Label
                raw_name = getattr(el, "Name", None) or getattr(el, "ObjectType", None) or "Unnamed"
                nm = str(raw_name).replace(" ", "_")
                nm = nm.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
                gml_label = f"{el.is_a()}-{nm}-{el.GlobalId}"

                # “poly.points” is an M×3 array in (X_webmerc, Y_webmerc, Z=alt)
                pts_list = poly.points.tolist()
                out.append(( [pts_list], {"gml_id": gml_label, "geom_dim": 3} ))

            except Exception as e:
                print(f"[WARN] Skipping {el.is_a()} {getattr(el, 'GlobalId','?')} due to: {e}")

    print(f"[INFO] Extracted {len(out)} elements from {ifc_path}")
    return out


def process_ifc(
    input_path: str,
    output_path: str,
    lod: int,
    grid_type: str,
    grid_level: int,
    grid_size: List[float],
    grid_crs: int,
    origin_llh: Tuple[float, float, float],
    yaw_deg: float,
    interpolate: bool,
    merge: bool,
    debug: bool
):
    # Step A: We must force grid_crs=3857 if grid_type='zfxy'
    grid = get_grid(grid_type, level=grid_level, size=grid_size, crs=grid_crs)
    grid.clear()

    # Step B: Extract (and reproject) geometry → “WebMercator metres”
    data = extract_geometry(input_path, origin_llh, yaw_deg)

    # Step C: Voxelize
    grid.load_geom_data(data, interpolate=interpolate, merge=merge)

    # Step D: Write CSV
    export_csv(grid, output_path, merge=merge)


def _safe_process_file(args):
    try:
        process_ifc(*args)
    except Exception as e:
        print(f"[ERROR] Failed on {args[0]} -> {args[1]}")
        import traceback
        traceback.print_exc()
        raise


def main(
    input_path: str,
    output_path: str,
    lod: int,
    grid_type: str,
    grid_level: int,
    grid_size: List[float],
    grid_crs: int,
    origin_llh: Tuple[float, float, float],
    yaw_deg: float,
    interpolate: bool,
    merge: bool,
    debug: bool
):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    # 1) Find all .ifc files under input_path
    input_files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith(".ifc")
    ]
    # 2) Build .csv output names under output_path
    output_files = [
        os.path.join(output_path, os.path.splitext(os.path.basename(f))[0] + ".csv")
        for f in input_files
    ]

    task_args = [
        (
            inp, outp,
            lod,
            grid_type,
            grid_level,
            grid_size,
            grid_crs,
            origin_llh,
            yaw_deg,
            interpolate,
            merge,
            debug
        )
        for inp, outp in zip(input_files, output_files)
    ]

    print(f"[INFO] Launching {len(task_args)} parallel tasks…")
    with ProcessPoolExecutor() as executor:
        executor.map(_safe_process_file, task_args)

    print("[INFO] All tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--lod",       type=int,   default=3)
    parser.add_argument("--grid-type", default="zfxy")
    parser.add_argument("--grid-level", type=int,  default=28)
    parser.add_argument("--grid-size",  type=float, nargs="*", default=[1.0])
    parser.add_argument(
        "--grid-crs",
        type=int,
        default=3857,
        help="When you use grid-type='zfxy', you MUST pass --grid-crs 3857"
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        required=True,
        help="anchor lat lon alt (e.g. 35.7157352071 139.761010938 59.6643551096)"
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=0.0,
        help="global yaw (°) about IFC origin (0,0,0) before geospatial conversion"
    )
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--merge",       action="store_true")
    parser.add_argument("--debug",       action="store_true")
    args = parser.parse_args()

    # Sanity check:
    if args.grid_type.lower() == "zfxy" and args.grid_crs != 3857:
        parser.error("When grid-type='zfxy', you must set --grid-crs 3857 (WebMercator).")

    main(
        args.input_path,
        args.output_path,
        args.lod,
        args.grid_type,
        args.grid_level,
        args.grid_size,
        args.grid_crs,
        (args.origin[0], args.origin[1], args.origin[2]),
        args.yaw_deg,
        args.interpolate,
        args.merge,
        args.debug
    )


