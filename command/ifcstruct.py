import ifcopenshell
from collections import Counter

model = ifcopenshell.open(r"C:\Users\kenchitaru-alex\Downloads\ifcconvert-0.8.2-win64\ifc\I-REF-ifc4.ifc")
type_counts = Counter(el.is_a() for el in model)

for element_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"{element_type}: {count}")