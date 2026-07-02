"""Dataset registry and base classes for UGVU.

Supports: Cityscapes, ADE20K, RefCOCO/RefCOCOg, NYU Depth V2, KITTI.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Lazy imports for heavy ML libraries
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    TorchDataset = object  # fallback


# ---------------------------------------------------------------------------
# Color maps for semantic segmentation datasets
# ---------------------------------------------------------------------------

CITYSCAPES_COLORMAP = np.array([
    [128,  64, 128],  # 0: road
    [244,  35, 232],  # 1: sidewalk
    [ 70,  70,  70],  # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170,  30],  # 6: traffic light
    [220, 220,   0],  # 7: traffic sign
    [107, 142,  35],  # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [ 70, 130, 180],  # 10: sky
    [220,  20,  60],  # 11: person
    [255,   0,   0],  # 12: rider
    [  0,   0, 142],  # 13: car
    [  0,   0,  70],  # 14: truck
    [  0,  60, 100],  # 15: bus
    [  0,  80, 100],  # 16: train
    [  0,   0, 230],  # 17: motorcycle
    [119,  11,  32],  # 18: bicycle
], dtype=np.uint8)

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth",
    "door", "table", "mountain", "plant", "curtain", "chair", "car",
    "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug",
    "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe",
    "lamp", "bathtub", "railing", "cushion", "base", "box", "column",
    "signboard", "chest of drawers", "counter", "sand", "sink",
    "skyscraper", "fireplace", "refrigerator", "grandstand", "path",
    "stairs", "runway", "case", "pool table", "pillow", "screen door",
    "stairway", "river", "bridge", "bookcase", "blind", "coffee table",
    "toilet", "flower", "book", "hill", "bench", "countertop", "stove",
    "palm", "kitchen island", "computer", "swivel chair", "boat", "bar",
    "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower",
    "chandelier", "awning", "streetlight", "booth", "television receiver",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister",
    "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van",
    "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall", "tent",
    "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank",
    "trade name", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen",
    "plate", "monitor", "bulletin board", "shower", "radiator", "glass",
    "clock", "flag",
]


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------

if HAS_TORCH:
    _Base = (TorchDataset, ABC)
else:
    _Base = (ABC,)

class BaseVisionDataset(*_Base):
    """Abstract base for all UGVU vision datasets."""

    def __init__(
        self,
        root: str,
        split: str = "val",
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        self.samples: List[dict] = []
        self._load_samples()

    @abstractmethod
    def _load_samples(self) -> None:
        """Populate self.samples with dicts containing at minimum {'image': path, 'label': path}."""
        ...

    def __len__(self) -> int:
        if self.max_samples is not None:
            return min(len(self.samples), self.max_samples)
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = np.array(Image.open(sample["image"]).convert("RGB"))
        label = self._load_label(sample)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image, label = transformed["image"], transformed["mask"]

        return {
            "image": image,
            "label": label,
            "image_path": str(sample["image"]),
            "label_path": sample.get("label", ""),
            "index": idx,
        }

    @abstractmethod
    def _load_label(self, sample: dict) -> np.ndarray:
        """Load and return the ground-truth label as a numpy array."""
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        ...

    @property
    @abstractmethod
    def ignore_index(self) -> int:
        ...

    @property
    def colormap(self) -> Optional[np.ndarray]:
        return None

    @property
    def class_names(self) -> Optional[List[str]]:
        return None


# ---------------------------------------------------------------------------
# Cityscapes
# ---------------------------------------------------------------------------

class CityscapesDataset(BaseVisionDataset):
    """Cityscapes semantic segmentation dataset.

    Expected structure:
        root/
          leftImg8bit/{split}/{city}/*.png
          gtFine/{split}/{city}/*_gtFine_labelIds.png
    """

    num_classes = 19
    ignore_index = 255
    colormap = CITYSCAPES_COLORMAP
    class_names = CITYSCAPES_CLASSES

    def _load_samples(self) -> None:
        img_dir = self.root / "leftImg8bit" / self.split
        if not img_dir.exists():
            raise FileNotFoundError(f"Cityscapes image directory not found: {img_dir}")

        for city in sorted(os.listdir(img_dir)):
            city_img_dir = img_dir / city
            if not city_img_dir.is_dir():
                continue
            for fname in sorted(os.listdir(city_img_dir)):
                if not fname.endswith(".png"):
                    continue
                img_path = city_img_dir / fname
                # gtFine/{split}/{city}/{base}_gtFine_labelIds.png
                base = fname.replace("_leftImg8bit.png", "")
                lbl_path = self.root / "gtFine" / self.split / city / f"{base}_gtFine_labelIds.png"
                if lbl_path.exists():
                    self.samples.append({"image": str(img_path), "label": str(lbl_path)})

    def _load_label(self, sample: dict) -> np.ndarray:
        label = np.array(Image.open(sample["label"]))
        # Map Cityscapes label IDs (0-33) to train IDs (0-18)
        label = self._map_to_train_ids(label)
        return label.astype(np.int64)

    @staticmethod
    def _map_to_train_ids(label: np.ndarray) -> np.ndarray:
        """Map Cityscapes 34-class labels to 19-class train IDs."""
        # Mapping from label IDs to train IDs
        mapping = np.full(256, 255, dtype=np.int64)  # default: ignore
        train_ids = {
             0: 255,  1: 255,  2: 255,  3: 255,  4: 255,  5: 255,  6: 255,
             7:   0,  8:   1,  9: 255, 10: 255, 11:   2, 12:   3, 13:   4,
            14: 255, 15: 255, 16: 255, 17:   5, 18: 255, 19:   6, 20:   7,
            21:   8, 22:   9, 23:  10, 24:  11, 25:  12, 26:  13, 27:  14,
            28:  15, 29: 255, 30: 255, 31:  16, 32:  17, 33:  18,
        }
        for k, v in train_ids.items():
            mapping[k] = v
        return mapping[label]


# ---------------------------------------------------------------------------
# ADE20K
# ---------------------------------------------------------------------------

class ADE20KDataset(BaseVisionDataset):
    """ADE20K semantic segmentation dataset.

    Expected structure:
        root/
          images/{split}/*.jpg
          annotations/{split}/*.png
    """

    num_classes = 150
    ignore_index = 255
    class_names = ADE20K_CLASSES

    def _load_samples(self) -> None:
        split = "validation" if self.split == "val" else self.split
        img_dir = self.root / "images" / split
        ann_dir = self.root / "annotations" / split
        if not img_dir.exists():
            raise FileNotFoundError(f"ADE20K image directory not found: {img_dir}")

        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = img_dir / fname
            base = os.path.splitext(fname)[0]
            lbl_path = ann_dir / f"{base}.png"
            if lbl_path.exists():
                self.samples.append({"image": str(img_path), "label": str(lbl_path)})

    def _load_label(self, sample: dict) -> np.ndarray:
        label = np.array(Image.open(sample["label"]))
        # ADE20K labels: 0 is unlabeled→255, 1-150 stay as 0-149
        label = label.astype(np.int64)
        label[label == 0] = 255         # unlabeled → ignore
        label[label != 255] -= 1        # 1..150 → 0..149
        return label


# ---------------------------------------------------------------------------
# RefCOCO / RefCOCOg (Referring Segmentation)
# ---------------------------------------------------------------------------

class RefCOCODataset(BaseVisionDataset):
    """RefCOCO / RefCOCOg referring segmentation dataset.

    Expected structure:
        root/
          images/          # MS COCO train2014 images
          annotations/     # refcoco.json / refcocog.json
          masks/           # ground-truth binary masks per annotation
    """

    num_classes = 2       # foreground (1) / background (0)
    ignore_index = 255

    def __init__(self, root: str, split: str = "val", dataset_name: str = "refcoco", **kwargs):
        self.dataset_name = dataset_name  # "refcoco" or "refcocog"
        super().__init__(root, split, **kwargs)

    def _load_samples(self) -> None:
        ann_file = self.root / "annotations" / f"{self.dataset_name}.json"
        if not ann_file.exists():
            # Fall back to synthetic structure for dev
            img_dir = self.root / "images"
            mask_dir = self.root / "masks"
            if img_dir.exists() and mask_dir.exists():
                for fname in sorted(os.listdir(img_dir)):
                    if fname.endswith((".jpg", ".png")):
                        base = os.path.splitext(fname)[0]
                        mpath = mask_dir / f"{base}.png"
                        if mpath.exists():
                            self.samples.append({
                                "image": str(img_dir / fname),
                                "label": str(mpath),
                                "expression": "",
                            })
                return
            raise FileNotFoundError(f"RefCOCO annotation file not found: {ann_file}")

        with open(ann_file, "r") as f:
            data = json.load(f)

        image_dir = self.root / "images"
        mask_dir = self.root / "masks"
        for ann in data.get("annotations", []):
            if ann.get("split") != self.split:
                continue
            img_path = image_dir / ann["image"]
            mask_path = mask_dir / f"{ann['id']}.png"
            if img_path.exists():
                self.samples.append({
                    "image": str(img_path),
                    "label": str(mask_path) if mask_path.exists() else "",
                    "expression": ann.get("expression", ""),
                    "ann_id": ann["id"],
                })

    def _load_label(self, sample: dict) -> np.ndarray:
        if sample.get("label"):
            label = np.array(Image.open(sample["label"]).convert("L"))
            return (label > 127).astype(np.int64)
        return np.zeros((1, 1), dtype=np.int64)


# ---------------------------------------------------------------------------
# NYU Depth V2
# ---------------------------------------------------------------------------

class NYUDepthV2Dataset(BaseVisionDataset):
    """NYU Depth V2 dataset for depth estimation.

    Expected structure:
        root/
          images/*.jpg
          depth/*.png   (16-bit depth maps in mm)
    """

    num_classes = 1
    ignore_index = -1

    def _load_samples(self) -> None:
        img_dir = self.root / "images"
        depth_dir = self.root / "depth"
        if not img_dir.exists():
            raise FileNotFoundError(f"NYU Depth V2 image directory not found: {img_dir}")

        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith((".jpg", ".png", ".jpeg")):
                continue
            base = os.path.splitext(fname)[0]
            dpath = depth_dir / f"{base}.png"
            if dpath.exists():
                self.samples.append({"image": str(img_dir / fname), "label": str(dpath)})

    def _load_label(self, sample: dict) -> np.ndarray:
        label = np.array(Image.open(sample["label"]))
        # Convert mm to meters
        depth_m = label.astype(np.float32) / 1000.0
        return depth_m


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, type] = {
    "cityscapes": CityscapesDataset,
    "ade20k": ADE20KDataset,
    "refcoco": RefCOCODataset,
    "refcocog": RefCOCODataset,
    "nyu_depth_v2": NYUDepthV2Dataset,
}


def build_dataset(name: str, root: str, split: str = "val", **kwargs) -> BaseVisionDataset:
    """Factory to build a dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    cls = DATASET_REGISTRY[name]
    if name in ("refcoco", "refcocog"):
        return cls(root, split=split, dataset_name=name, **kwargs)
    return cls(root, split=split, **kwargs)



