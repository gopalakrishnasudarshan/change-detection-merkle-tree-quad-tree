Change detection using Merkle Trees on Satellite Imagery.
Detecting changes between two satellite images of the same location .
Uses Sentinel - 2 imagery -OSCD dataset
Images are divided into fixed-size tiles for localized analysis.
Each tile is hashed and organized into a Merkle tree.
Root hash comparison indicates global change; tree structure enables future change localization.

Work done so far:
Image loading and validation
Tiling implemented
Descriptor Extraction (intensity normalization, quantization, and edge/gradient-based descriptors)
Tile level hashing
Merkle tree construction

Next up:
Visualization and change localization
Introduce spatial tolerence to handle minor misregistration (±1–2 pixels) between Sentinel-2 image pairs
