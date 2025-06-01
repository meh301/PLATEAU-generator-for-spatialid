import argparse
import logging
import os
import typing

import grids
import inputs
import outputs

from concurrent.futures import ProcessPoolExecutor
from functools import partial

Any = typing.Any
Dict = typing.Dict
List = typing.List
Tuple = typing.Tuple
Iterator = typing.Iterator

logger = logging.getLogger(__name__)


def _process_file(input_file, output_file, lod, grid_type, grid_level, grid_size, grid_crs,
                  ids, extract, extrude, interpolate, merge, debug):
    grid = grids.get(
        grid_type,
        level=grid_level,
        size=grid_size,
        crs=grid_crs
    )
    grid.clear()
    if extract:
        xml2id(
            input_file, output_file, lod, grid, ids,
            extrude=extrude, interpolate=interpolate,
            merge=merge, debug=debug
        )
    else:
        geom2id(
            input_file, output_file, lod, grid, ids,
            interpolate=interpolate, merge=merge, debug=debug
        )

def _safe_process_file(args):
    input_file, output_file, kwargs = args
    try:
        _process_file(input_file, output_file, **kwargs)
    except Exception as e:
        print(f"[ERROR] Failed on {input_file} -> {output_file}")
        import traceback
        traceback.print_exc()


def main(input_file_or_dir: str, output_file_or_dir: str, lod: int,
         grid_type: str, grid_level: int, grid_size: List[float], grid_crs: int,
         ids: Tuple[str], extract: bool = False, extrude: List[float] = [],
         interpolate: bool = False, merge: bool = False, debug: bool = False
         ) -> None:
    extrude = extrude or []
    if extrude and len(extrude) != 2:
        raise ValueError(f'Invalid extrude: {extrude}')

    # Ensure ids is never None
    ids = ids or ()

    output_ext = os.path.splitext(output_file_or_dir)[-1]
    if os.path.isdir(input_file_or_dir) and output_ext == '':
        input_files = inputs.get_target_gml_files(input_file_or_dir)
        output_files = outputs.build_output_paths(
            grids.get(grid_type, level=grid_level, size=grid_size, crs=grid_crs),
            input_file_or_dir,
            input_files,
            output_file_or_dir,
            merge=merge
        )
    elif os.path.isfile(input_file_or_dir) and output_ext == '.csv':
        input_files = [input_file_or_dir]
        output_files = [output_file_or_dir]
    else:
        raise ValueError(f'Invalid path: {input_file_or_dir} {output_file_or_dir}')

    process_func = partial(
        _process_file,
        lod=lod,
        grid_type=grid_type,
        grid_level=grid_level,
        grid_size=grid_size,
        grid_crs=grid_crs,
        ids=ids,
        extract=extract,
        extrude=extrude,
        interpolate=interpolate,
        merge=merge,
        debug=debug
    )

    with ProcessPoolExecutor() as executor:
        kwargs = dict(
        lod=lod,
        grid_type=grid_type,
        grid_level=grid_level,
        grid_size=grid_size,
        grid_crs=grid_crs,
        ids=ids,
        extract=extract,
        extrude=extrude,
        interpolate=interpolate,
        merge=merge,
        debug=debug
    )

    task_args = [(inp, outp, kwargs) for inp, outp in zip(input_files, output_files)]

    print(f"[INFO] Launching {len(task_args)} parallel tasks...")
    with ProcessPoolExecutor() as executor:
        executor.map(_safe_process_file, task_args)

    # Merge CSV parts if they exist
    print(f"[INFO] Consolidating CSV parts...")
    outputs.consolidate_output_parts(output_file_or_dir)


def xml2id(input_file: str, output_file: str, lod: int, grid: grids.Grid,
           ids: Tuple[str], extrude: List[float] = [], interpolate: bool = False,
           merge: bool = False, debug: bool = False) -> None:
    xml = inputs.load_xml(input_file)
    grid.extract_ids(xml)
    grid.extrude(*extrude[:2])
    if merge:
        grid.merge()
    outputs.export_csv(grid, output_file, merge=merge)


def geom2id(input_file: str, output_file: str, lod: int, grid: grids.Grid,
            ids: Tuple[str], interpolate: bool = False, merge: bool = False,
            debug: bool = False) -> None:
    data = inputs.load_features(input_file, ids, lod, grid.crs, debug=debug)
    grid.load_geom_data(data, interpolate=interpolate, merge=merge)
    outputs.export_csv(grid, output_file, merge=merge)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file_or_dir', help='path to the CityGML file (*.gml) or directory')
    parser.add_argument('output_file_or_dir', help='path to the ID pair list file (*.csv) or directory')
    parser.add_argument('--lod', type=int, choices=(1, 2, 3), default=3, help='maximum LOD of target geometries')
    parser.add_argument('--grid-type', choices=('zfxy',), default='zfxy', help='type of the output voxel grid')
    parser.add_argument('--grid-level', type=int, help='zoom level of the output voxel grid')
    parser.add_argument('--grid-size', type=float, nargs='*', help='size of the output voxel grid')
    parser.add_argument('--grid-crs', type=int, help='coordinate reference system of the output voxel grid')
    parser.add_argument('--id', nargs='*', help='gml:ids which will be filtered')
    parser.add_argument('--extract', action='store_true', help='whether extract spatial ids from CityGML or not')
    parser.add_argument('--extrude', type=float, nargs='*', help='min extrude and max extrude (unit: m)')
    parser.add_argument('--interpolate', action='store_true', help='whether interpolate inner voxels of solids or not')
    parser.add_argument('--merge', action='store_true', help='whether merge 8 adjacent voxels into 1 large voxel or not')
    parser.add_argument('--debug', action='store_true', help='whether output debug messages and retain temporary files or not')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    main(
        input_file_or_dir=args.input_file_or_dir,
        output_file_or_dir=args.output_file_or_dir,
        lod=args.lod,
        grid_type=args.grid_type,
        grid_level=args.grid_level,
        grid_size=args.grid_size,
        grid_crs=args.grid_crs,
        ids=args.id,
        extract=args.extract,
        extrude=args.extrude,
        interpolate=args.interpolate,
        merge=args.merge,
        debug=args.debug
    )
