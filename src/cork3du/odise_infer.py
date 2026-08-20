"""Open-vocabulary panoptic inference via NVlabs/ODISE (RGB → labels + instance masks)."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/Panoptic/odise_label_coco_50e.py"
DEFAULT_INIT = "odise://Panoptic/odise_label_coco_50e"
DEFAULT_LABEL_SETS = ("COCO", "ADE", "LVIS")


@dataclass
class OdisePrediction:
    """Per-pixel panoptic id map + segment metadata for one RGB frame."""

    panoptic: np.ndarray  # HxW int32; 0 = void / unlabeled
    segments: list[dict[str, Any]] = field(default_factory=list)
    # contiguous category_id → display name / isthing
    category_names: dict[int, str] = field(default_factory=dict)
    category_isthing: dict[int, bool] = field(default_factory=dict)


class OdiseModel:
    """Lazy-loaded ODISE demo wrapper. Call once per process, reuse across frames."""

    def __init__(
        self,
        *,
        odise_root: Path,
        config_file: str | Path | None = None,
        init_from: str | None = None,
        label_sets: Sequence[str] = DEFAULT_LABEL_SETS,
        vocab: Sequence[Sequence[str]] | None = None,
        device: str | None = None,
    ) -> None:
        import torch
        from detectron2.config import LazyConfig, instantiate
        from detectron2.data import MetadataCatalog
        from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
        from detectron2.engine import create_ddp_model
        from detectron2.utils.visualizer import random_color
        from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
        from torch import nn
        from detectron2.evaluation import inference_context
        from contextlib import ExitStack

        from odise.checkpoint import ODISECheckpointer
        from odise.config import instantiate_odise
        from odise.data import get_openseg_labels
        from odise.engine.defaults import get_model_from_module

        root = Path(odise_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"ODISE clone missing at {root} — run bash scripts/setup.sh")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        cfg_path = Path(config_file) if config_file else root / DEFAULT_CONFIG
        if not cfg_path.is_file():
            # configs live under the clone; allow relative to root
            alt = root / str(config_file or DEFAULT_CONFIG)
            if alt.is_file():
                cfg_path = alt
            else:
                raise FileNotFoundError(cfg_path)
        init_from = init_from or DEFAULT_INIT
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        cfg = LazyConfig.load(str(cfg_path))
        cfg.model.overlap_threshold = 0
        cfg.model.clip_head.alpha = 0.35
        cfg.model.clip_head.beta = 0.65
        cfg.train.device = device

        coco_thing = [
            label
            for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
            if COCO_CATEGORIES[idx]["isthing"] == 1
        ]
        coco_thing_colors = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
        coco_stuff = [
            label
            for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
            if COCO_CATEGORIES[idx]["isthing"] == 0
        ]
        coco_stuff_colors = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]
        ade_thing = [
            label
            for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
            if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
        ]
        ade_thing_colors = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
        ade_stuff = [
            label
            for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
            if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
        ]
        ade_stuff_colors = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]
        lvis = get_openseg_labels("lvis_1203", True)
        import itertools

        lvis_colors = list(
            itertools.islice(itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(lvis))
        )

        thing_classes: list[list[str]] = [list(g) for g in (vocab or [])]
        stuff_classes: list[list[str]] = []
        thing_colors = [random_color(rgb=True, maximum=1) for _ in thing_classes]
        stuff_colors: list = []

        label_upper = {s.upper() for s in label_sets}
        if "COCO" in label_upper:
            thing_classes += coco_thing
            stuff_classes += coco_stuff
            thing_colors += coco_thing_colors
            stuff_colors = list(coco_stuff_colors)
        if "ADE" in label_upper:
            thing_classes += ade_thing
            stuff_classes += ade_stuff
            thing_colors += ade_thing_colors
            stuff_colors += ade_stuff_colors
        if "LVIS" in label_upper:
            thing_classes += lvis
            thing_colors += lvis_colors

        demo_metadata = MetadataCatalog.get("odise_demo_metadata")
        demo_metadata.thing_classes = [c[0] for c in thing_classes]
        demo_metadata.stuff_classes = [
            *demo_metadata.thing_classes,
            *[c[0] for c in stuff_classes],
        ]
        demo_metadata.thing_colors = thing_colors
        demo_metadata.stuff_colors = thing_colors + stuff_colors
        demo_metadata.stuff_dataset_id_to_contiguous_id = {
            idx: idx for idx in range(len(demo_metadata.stuff_classes))
        }
        demo_metadata.thing_dataset_id_to_contiguous_id = {
            idx: idx for idx in range(len(demo_metadata.thing_classes))
        }

        dataset_cfg = cfg.dataloader.test
        wrapper_cfg = cfg.dataloader.wrapper
        wrapper_cfg.labels = thing_classes + stuff_classes
        wrapper_cfg.metadata = demo_metadata

        from detectron2.data import transforms as T

        aug = instantiate(dataset_cfg.mapper).augmentations

        logger.info("Loading ODISE (%s) from %s", device, init_from)
        model = instantiate_odise(cfg.model)
        model.to(device)
        ODISECheckpointer(model).load(init_from)
        while "model" in wrapper_cfg:
            wrapper_cfg = wrapper_cfg.model
        wrapper_cfg.model = get_model_from_module(model)
        inference_model = create_ddp_model(instantiate(cfg.dataloader.wrapper))
        inference_model.eval()

        self._aug = aug
        self._model = inference_model
        self._metadata = demo_metadata
        self._device = device
        self._stack = ExitStack()
        if isinstance(inference_model, nn.Module):
            self._stack.enter_context(inference_context(inference_model))
        self._stack.enter_context(torch.no_grad())

        n_thing = len(demo_metadata.thing_classes)
        self.category_names: dict[int, str] = {
            i: name for i, name in enumerate(demo_metadata.stuff_classes)
        }
        self.category_isthing: dict[int, bool] = {
            i: (i < n_thing) for i in range(len(demo_metadata.stuff_classes))
        }
        self.label_sets = tuple(sorted(label_upper))
        self.init_from = init_from
        self.config_file = str(cfg_path)

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> OdiseModel:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def predict(self, image_rgb: np.ndarray) -> OdisePrediction:
        """Run ODISE on an HxWx3 RGB uint8 (or float) image."""
        import torch
        from detectron2.data import transforms as T

        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        height, width = image_rgb.shape[:2]
        # Detectron2 demo path expects RGB numpy; AugInput applies ResizeShortestEdge etc.
        aug_input = T.AugInput(image_rgb, sem_seg=None)
        self._aug(aug_input)
        image = torch.as_tensor(aug_input.image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        predictions = self._model([inputs])[0]

        if "panoptic_seg" not in predictions:
            raise RuntimeError("ODISE returned no panoptic_seg — check model/config")
        panoptic_t, segments_info = predictions["panoptic_seg"]
        panoptic = panoptic_t.detach().cpu().numpy().astype(np.int32)
        segments: list[dict[str, Any]] = []
        for s in segments_info:
            cat = int(s["category_id"])
            segments.append(
                {
                    "id": int(s["id"]),
                    "category_id": cat,
                    "isthing": bool(s.get("isthing", self.category_isthing.get(cat, False))),
                    "score": float(s["score"]) if "score" in s else None,
                    "name": self.category_names.get(cat, f"class_{cat}"),
                }
            )
        return OdisePrediction(
            panoptic=panoptic,
            segments=segments,
            category_names=dict(self.category_names),
            category_isthing=dict(self.category_isthing),
        )


def parse_vocab(spec: str | None) -> list[list[str]] | None:
    """'truck, pickup; sky' → [['truck','pickup'], ['sky']]."""
    if not spec or not spec.strip():
        return None
    groups: list[list[str]] = []
    for part in spec.split(";"):
        words = [w.strip() for w in part.split(",") if w.strip()]
        if words:
            groups.append(words)
    return groups or None
