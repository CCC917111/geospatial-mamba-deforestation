"""Shared ForTy metadata and model defaults, without framework imports."""

from __future__ import annotations

GCS_ROOT = "gs://forest_typology/forty_v1/1.0.0"
TOTAL_SHARDS = 1024

# Class order of the ForTy v1 segmentation labels.
CLASS_NAMES = (
    "Unknown",
    "Natural Forest",
    "Planted Forest",
    "Tree Crops",
    "Other Vegetation",
    "Water",
    "Ice",
    "Bare Ground",
    "Built Areas",
)
NUM_CLASSES = len(CLASS_NAMES)

# The three forest types aggregated into the "Forests" metric.
FOREST_CLASS_INDICES = (1, 2, 3)
NATURAL_FOREST_INDEX = 1
PLANTED_FOREST_INDEX = 2
TREE_CROPS_INDEX = 3

FEATURE_SHAPES = {
    "s2": (4, 128, 128, 10),
    "elevation": (128, 128, 3),
    "climate": (4, 14),
    "segmentation_labels": (128, 128),
}

MODEL_DEFAULTS = {
    "d_model": 32,
    "dropout_rate": 0.2,
    "d_state_temporal": 8,
    "d_state_spatial": 16,
    "bottleneck_channels": 512,
}
