import os
from pathlib import Path
from lxml import etree

NAMESPACE = {
    'core': 'http://www.opengis.net/citygml/2.0',
}

def find_large_gml_files(base_dir: Path, size_threshold_mb: int = 20):
    """Return list of .gml files over given size (in MB)."""
    return [
        f for f in base_dir.rglob('*.gml')
        if f.stat().st_size > size_threshold_mb * 1024 * 1024
    ]


def split_citygml_file(input_file: Path, input_root: Path, output_root: Path, chunk_size: int = 1000):
    """Split a CityGML file by <core:cityObjectMember> elements, preserving folder structure."""
    rel_path = input_file.relative_to(input_root)
    out_subdir = output_root / rel_path.parent
    out_subdir.mkdir(parents=True, exist_ok=True)

    context = etree.iterparse(str(input_file), events=('end',), tag='{http://www.opengis.net/citygml/2.0}cityObjectMember')

    # Read the root and namespaces
    tree = etree.parse(str(input_file))
    root = tree.getroot()
    nsmap = root.nsmap
    header = b'<?xml version="1.0" encoding="UTF-8"?>\n' + etree.tostring(root, encoding='utf-8', pretty_print=True, xml_declaration=False).split(b'>', 1)[0] + b'>\n'

    members = []
    count = 0
    file_index = 0

    for event, elem in context:
        members.append(etree.tostring(elem, encoding='utf-8'))
        count += 1
        if count % chunk_size == 0:
            write_chunk(out_subdir, input_file.stem, file_index, header, members)
            members.clear()
            file_index += 1
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    # Write remaining
    if members:
        write_chunk(out_subdir, input_file.stem, file_index, header, members)

    print(f"✅ Split {input_file.relative_to(input_root)} into {file_index + 1} parts.")
    return file_index + 1


def write_chunk(output_dir: Path, base_name: str, index: int, header: bytes, members: list):
    """Write one chunk of cityObjectMembers to a GML file."""
    out_path = output_dir / f"{base_name}_part{index}.gml"
    with open(out_path, 'wb') as f:
        f.write(header)
        f.write(b'\n'.join(members))
        f.write(b'\n</core:CityModel>')
    print(f"  ↳ Wrote {out_path.relative_to(output_dir.parent)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split large CityGML files into smaller subfiles.")
    parser.add_argument("input_dir", type=str, help="Path to directory containing CityGML files.")
    parser.add_argument("output_dir", type=str, help="Directory where subfiles will be written.")
    parser.add_argument("--threshold-mb", type=int, default=20, help="Size in MB to consider a file 'large'.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of <cityObjectMember> per output file.")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    large_files = find_large_gml_files(input_dir, args.threshold_mb)
    if not large_files:
        print("No large .gml files found.")
    else:
        for gml_file in large_files:
            split_citygml_file(gml_file, input_dir, output_dir, args.chunk_size)
