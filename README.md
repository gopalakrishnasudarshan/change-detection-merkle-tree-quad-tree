# Merkle Quad-Tree Based Image Change Detection

Detects localized differences between two aligned RGB images using a Merkle Quad-Tree framework.

## Overview

The image is divided into level-based quadtree patches:

- L = 1 → 2×2
- L = 2 → 4×4
- L = 3 → 8×8 (64 patches)
- L = 4 → 16×16 (256 patches)

Each patch is hashed using perceptual hashing (pHash) and organized into a Merkle Tree.

- Root hash comparison → Global change detection
- Leaf comparison → Patch-level localization

## Current Features

- RGB image loading and validation
- Level-based quadtree partitioning
- Perceptual hashing (ImageHash)
- Merkle tree construction
- Change localization via leaf differences
- Visualization:
  - Changed patches remain in color
  - Unchanged patches converted to grayscale

  ## Recent Improvements

- Added alignment and pixel-difference sanity checks to validate input image consistency.
- Modularized visualization logic into viz.py for cleaner architecture.
- Improved hash handling by standardizing byte-level patch hashes for Merkle tree input.

## Future Enhancements

- Apply the pipeline to real multi-temporal Google Maps satellite screenshot pairs (same location, different timeframes).
- Increase quadtree level (L > 4) for finer spatial localization of changes.
- Experiment with different quadtree levels to analyze precision vs. patch size trade-offs.
- Extend visualization to replicate higher-version PPT outputs more precisely.

## Dependencies

- Python 3.x
- Pillow
- ImageHash
- NumPy
