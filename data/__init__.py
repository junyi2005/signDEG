"""Data pipeline: cached per-frame features (FeatureStore), the multi-temporal-crop sampler and its collate.

The dataset returns multi-temporal-crop samples for one stream at a time; the trainer runs one stream
per process.
"""

from .temporal_crop import TemporalMultiCropSampler, MultiCropConfig
from .collate import multi_crop_collate
from .feature_store import FeatureStore, MultiStore, GeoDatasetConfig, GeoStreamClipDataset

__all__ = [
    "TemporalMultiCropSampler",
    "MultiCropConfig",
    "multi_crop_collate",
    "FeatureStore",
    "MultiStore",
    "GeoDatasetConfig",
    "GeoStreamClipDataset",
]
