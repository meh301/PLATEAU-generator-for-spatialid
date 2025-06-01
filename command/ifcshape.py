import ifcopenshell
import ifcopenshell.geom

model = ifcopenshell.open(r"C:\Users\kenchitaru-alex\Downloads\ifcconvert-0.8.2-win64\ifc\I-REF-ifc4.ifc")
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

types = ["IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof", "IfcDoor", "IfcWindow"]
for t in types:
    elements = model.by_type(t)
    print(f"{t}: {len(elements)}")
    for el in elements[:5]:  # test just a few
        try:
            shape = ifcopenshell.geom.create_shape(settings, el)
            print(f"✅ {el.is_a()} {el.GlobalId} → verts: {len(shape.geometry.verts)}")
        except Exception as e:
            print(f"❌ Failed on {el.GlobalId} ({el.is_a()}): {e}")
