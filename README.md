Change detection using Merkle Trees on Satellite Imagery.
Detecting changes between two satellite images of the same location .
Uses Sentinel - 2 imagery -OSCD dataset
Images are divided into fixed-size tiles for localized analysis.
Each tile is hashed and organized into a Merkle tree.
Root hash comparison indicates global change; tree structure enables future change localization.

Work done so far:
Image loading and validation
Tiling implemented
Tile hashing
Merkle tree construction

Next up:
Visualization and change localization
