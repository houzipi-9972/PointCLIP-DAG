from __future__ import annotations

from pathlib import Path
import inspect
import os
import os.path as osp
import pickle
import time
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T


class UniDSegDatasetAdapter(Dataset):
    """Wrap UniDSeg datasets and expose a stable PointCLIP-DAG sample schema."""

    DATASET_IMPORTS = {
        "VirtualKITTISCN": ("xmuda.data.virtual_kitti.virtual_kitti_dataloader", "VirtualKITTISCN"),
        "SemanticKITTISCN": ("xmuda.data.semantic_kitti.semantic_kitti_dataloader", "SemanticKITTISCN"),
        "NuScenesLidarSegSCN": ("xmuda.data.nuscenes_lidarseg.nuscenes_lidarseg_dataloader", "NuScenesLidarSegSCN"),
    }

    def __init__(self, dataset_type: str, split, dataset_kwargs: dict, mode: str = "train"):
        module_name, class_name = self.DATASET_IMPORTS[dataset_type]
        module = __import__(module_name, fromlist=[class_name])
        dataset_cls = getattr(module, class_name)
        kwargs = dict(dataset_kwargs)
        if dataset_type == "NuScenesLidarSegSCN":
            _require_nuscenes_devkit()
            kwargs.pop("nuscenes_devkit", None)
        if dataset_type in {"SemanticKITTISCN", "NuScenesLidarSegSCN"}:
            kwargs.setdefault("output_orig", mode != "train")
        kwargs = _filter_kwargs(dataset_cls, kwargs)
        kwargs = _normalize_unidseg_kwargs(kwargs)
        self.dataset = dataset_cls(split=tuple(split), **kwargs)
        self.dataset_type = dataset_type

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        coords = item["coords"]
        point_count = len(coords)
        feats = item["feats"]
        labels = item.get("seg_label", np.full(point_count, -100, dtype=np.int64))
        img_indices = item["img_indices"]
        point_count = min(point_count, len(feats), len(labels), len(img_indices))
        img_shape = item["img"].shape
        valid_mask = _valid_projected_mask(img_indices[:point_count], img_shape[1], img_shape[2])

        return {
            "coords": coords[:point_count],
            "features_3d": feats[:point_count],
            "labels_3d": labels[:point_count],
            "image": item["img"],
            "sparse_depth": item.get("depth", np.zeros((1, item["img"].shape[1], item["img"].shape[2]), dtype=np.float32)),
            "calib": item.get("calib", None),
            "point2img": img_indices[:point_count],
            "valid_mask": valid_mask,
            "scene_id": str(item.get("scene_id", "")),
            "frame_id": str(item.get("frame_id", index)),
            "dataset_name": self.dataset_type,
            "raw": item if item.get("calib", None) is not None else None,
        }


class VirtualKITTIRawSCN(Dataset):
    """VirtualKITTI pkl reader without UniDSeg six-class merging."""

    proj_matrix = np.array([[725, 0, 620.5], [0, 725, 187], [0, 0, 1]], dtype=np.float32)

    def __init__(
        self,
        split,
        preprocess_dir,
        virtual_kitti_dir="",
        scale=20,
        full_scale=4096,
        crop_size=tuple(),
        bottom_crop=False,
        rand_crop=tuple(),
        fliplr=0.0,
        color_jitter=None,
        random_weather=tuple(),
        noisy_rot=0.0,
        flip_y=0.0,
        rot_z=0.0,
        transl=False,
        downsample=(-1,),
        length=None,
        num_points=None,
        use_color=False,
        image_normalizer=None,
        **kwargs,
    ):
        self.split = tuple(split)
        self.preprocess_dir = preprocess_dir
        self.virtual_kitti_dir = virtual_kitti_dir
        self.scale = scale
        self.full_scale = full_scale
        self.crop_size = tuple(crop_size)
        self.bottom_crop = bottom_crop
        self.rand_crop = np.array(rand_crop)
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.random_weather = tuple(random_weather)
        self.noisy_rot = noisy_rot
        self.flip_y = flip_y
        self.rot_z = rot_z
        self.transl = transl
        if num_points is not None:
            downsample = (int(num_points),)
        self.downsample = tuple(downsample)[0] if len(tuple(downsample)) == 1 else tuple(downsample)
        self.use_color = use_color
        self.image_normalizer = image_normalizer

        self.data = []
        for curr_split in self.split:
            pkl_path = osp.join(self.preprocess_dir, curr_split + ".pkl")
            print(f"[data:VirtualKITTI] loading {pkl_path} ({_file_size_gb(pkl_path):.1f} GB)", flush=True)
            start = time.time()
            with open(pkl_path, "rb") as handle:
                self.data.extend(pickle.load(handle))
            if length is not None:
                self.data = self.data[: int(length)]
            print(
                f"[data:VirtualKITTI] loaded split={curr_split} total_samples={len(self.data)} "
                f"time={time.time() - start:.1f}s",
                flush=True,
            )
        self._debug_printed = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        from xmuda.data.utils.augmentation_3d import augment_and_scale_3d

        data_dict = self.data[index]
        points = data_dict["points"].copy()
        labels = data_dict["seg_labels"].astype(np.int64)
        labels[labels == 99] = -100

        num_points = self.downsample
        if isinstance(num_points, tuple):
            num_points = np.random.randint(low=num_points[0], high=num_points[1])
        if num_points > 0 and num_points < len(points):
            choice = np.random.choice(len(points), size=num_points, replace=False)
            points = points[choice]
            labels = labels[choice]

        points_cam_coords = np.array([-1, -1, 1]) * points[:, [1, 2, 0]]
        points_img = (self.proj_matrix @ points_cam_coords.T).T
        points_img = points_img[:, :2] / np.expand_dims(points_img[:, 2], axis=1)
        points_img = np.fliplr(points_img)

        weather = "clone"
        if self.random_weather:
            weather = self.random_weather[np.random.randint(len(self.random_weather))]
        img_path = osp.join(
            self.virtual_kitti_dir,
            "vkitti_1.3.1_rgb",
            data_dict["scene_id"],
            weather,
            data_dict["frame_id"] + ".png",
        )
        if self._debug_printed < 3:
            print(
                f"[data:VirtualKITTI] reading sample index={index} image={img_path} "
                f"points={len(points)}",
                flush=True,
            )
            self._debug_printed += 1
        image = Image.open(img_path)

        image_valid_mask = _valid_projected_mask(points_img, image.size[1], image.size[0], depth=points_cam_coords[:, 2])
        if self.crop_size:
            valid_crop = False
            for _ in range(10):
                if self.bottom_crop:
                    left = int(np.random.rand() * (image.size[0] + 1 - self.crop_size[0]))
                    right = left + self.crop_size[0]
                    top = image.size[1] - self.crop_size[1]
                    bottom = image.size[1]
                else:
                    crop_height, crop_width = self.rand_crop[0::2] + np.random.rand(2) * (
                        self.rand_crop[1::2] - self.rand_crop[0::2]
                    )
                    top = np.random.rand() * (1 - crop_height) * image.size[1]
                    left = np.random.rand() * (1 - crop_width) * image.size[0]
                    bottom = top + crop_height * image.size[1]
                    right = left + crop_width * image.size[0]
                    top, left, bottom, right = int(top), int(left), int(bottom), int(right)
                keep_idx = points_img[:, 0] >= top
                keep_idx = np.logical_and(keep_idx, points_img[:, 0] < bottom)
                keep_idx = np.logical_and(keep_idx, points_img[:, 1] >= left)
                keep_idx = np.logical_and(keep_idx, points_img[:, 1] < right)
                if np.sum(keep_idx) > 100:
                    valid_crop = True
                    break
            if valid_crop:
                image = image.crop((left, top, right, bottom))
                points_img[:, 0] -= top
                points_img[:, 1] -= left
                image_valid_mask = _valid_projected_mask(
                    points_img,
                    image.size[1],
                    image.size[0],
                    depth=points_cam_coords[:, 2],
                )

        img_indices = points_img.astype(np.int64)
        depth = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
        valid_depth = image_valid_mask & _valid_projected_mask(
            img_indices,
            image.size[1],
            image.size[0],
            depth=points_cam_coords[:, 2],
        )
        depth[img_indices[valid_depth, 0], img_indices[valid_depth, 1]] = points_cam_coords[valid_depth, 2] / 100.0

        if self.color_jitter is not None:
            image = self.color_jitter(image)
        image_np = np.array(image, dtype=np.float32, copy=False) / 255.0
        if np.random.rand() < self.fliplr:
            image_np = np.ascontiguousarray(np.fliplr(image_np))
            img_indices[:, 1] = image_np.shape[1] - 1 - img_indices[:, 1]
            depth = np.ascontiguousarray(np.fliplr(depth))
        if self.image_normalizer:
            mean, std = self.image_normalizer
            image_np = (image_np - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
        image_chw = np.moveaxis(image_np, -1, 0)

        coords = augment_and_scale_3d(
            points,
            self.scale,
            self.full_scale,
            noisy_rot=self.noisy_rot,
            flip_y=self.flip_y,
            rot_z=self.rot_z,
            transl=self.transl,
        )
        coords = coords.astype(np.int64)
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)
        if self.use_color:
            feats = _sample_point_colors(image_chw, img_indices[idxs], image_valid_mask[idxs])
        else:
            feats = np.ones([np.sum(idxs), 1], np.float32)

        return {
            "coords": coords[idxs],
            "features_3d": feats,
            "labels_3d": labels[idxs],
            "image": image_chw,
            "sparse_depth": depth[None].astype(np.float32),
            "calib": None,
            "point2img": img_indices[idxs],
            "valid_mask": image_valid_mask[idxs],
            "scene_id": str(data_dict.get("scene_id", "")),
            "frame_id": str(data_dict.get("frame_id", index)),
            "dataset_name": "virtual_kitti",
        }


class SemanticKITTIRawSCN(Dataset):
    """SemanticKITTI preprocessed pkl reader without UniDSeg class merging.

    Use this for open-vocabulary evaluation with raw SemanticKITTI label ids.
    The output schema matches UniDSegDatasetAdapter.
    """

    def __init__(
        self,
        split,
        preprocess_dir,
        semantic_kitti_dir="",
        scale=20,
        full_scale=4096,
        crop_size=tuple(),
        bottom_crop=False,
        rand_crop=tuple(),
        fliplr=0.0,
        color_jitter=None,
        image_normalizer=None,
        use_color=False,
        length=None,
        num_points=None,
        **kwargs,
    ):
        self.split = tuple(split)
        self.preprocess_dir = preprocess_dir
        self.semantic_kitti_dir = semantic_kitti_dir
        self.scale = scale
        self.full_scale = full_scale
        self.crop_size = tuple(crop_size)
        self.bottom_crop = bottom_crop
        self.rand_crop = np.array(rand_crop)
        self.fliplr = fliplr
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.image_normalizer = image_normalizer
        self.use_color = use_color
        self.num_points = None if num_points is None else int(num_points)

        self.data = []
        for curr_split in self.split:
            pkl_path = osp.join(self.preprocess_dir, curr_split + ".pkl")
            print(f"[data:SemanticKITTI] loading {pkl_path} ({_file_size_gb(pkl_path):.1f} GB)", flush=True)
            start = time.time()
            with open(pkl_path, "rb") as handle:
                self.data.extend(pickle.load(handle))
            if length is not None:
                self.data = self.data[: int(length)]
            print(
                f"[data:SemanticKITTI] loaded split={curr_split} total_samples={len(self.data)} "
                f"time={time.time() - start:.1f}s",
                flush=True,
            )
        self._debug_printed = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        from xmuda.data.utils.augmentation_3d import augment_and_scale_3d

        data_dict = self.data[index]
        points = data_dict["points"].copy()
        labels = data_dict["seg_labels"].astype(np.int64)
        pts_cam_coord = data_dict["pts_cam_coord"]
        points_img = data_dict["points_img"].copy()

        if self.num_points is not None and self.num_points > 0 and self.num_points < len(points):
            choice = np.random.choice(len(points), size=self.num_points, replace=False)
            points = points[choice]
            labels = labels[choice]
            pts_cam_coord = pts_cam_coord[choice]
            points_img = points_img[choice]

        img_path = osp.join(self.semantic_kitti_dir, data_dict["camera_path"])
        if self._debug_printed < 3:
            print(
                f"[data:SemanticKITTI] reading sample index={index} image={img_path} "
                f"points={len(points)}",
                flush=True,
            )
            self._debug_printed += 1
        image = Image.open(img_path)
        image_valid_mask = _valid_projected_mask(points_img, image.size[1], image.size[0], depth=pts_cam_coord[:, 2])

        if self.crop_size:
            valid_crop = False
            for _ in range(10):
                if self.bottom_crop:
                    left = int(np.random.rand() * (image.size[0] + 1 - self.crop_size[0]))
                    right = left + self.crop_size[0]
                    top = image.size[1] - self.crop_size[1]
                    bottom = image.size[1]
                else:
                    crop_height, crop_width = self.rand_crop[0::2] + np.random.rand(2) * (
                        self.rand_crop[1::2] - self.rand_crop[0::2]
                    )
                    top = np.random.rand() * (1 - crop_height) * image.size[1]
                    left = np.random.rand() * (1 - crop_width) * image.size[0]
                    bottom = top + crop_height * image.size[1]
                    right = left + crop_width * image.size[0]
                    top, left, bottom, right = int(top), int(left), int(bottom), int(right)

                crop_keep_idx = points_img[:, 0] >= top
                crop_keep_idx = np.logical_and(crop_keep_idx, points_img[:, 0] < bottom)
                crop_keep_idx = np.logical_and(crop_keep_idx, points_img[:, 1] >= left)
                crop_keep_idx = np.logical_and(crop_keep_idx, points_img[:, 1] < right)
                if np.sum(crop_keep_idx) > 100:
                    valid_crop = True
                    break

            if valid_crop:
                image = image.crop((left, top, right, bottom))
                points_img[:, 0] -= top
                points_img[:, 1] -= left
                image_valid_mask = _valid_projected_mask(
                    points_img,
                    image.size[1],
                    image.size[0],
                    depth=pts_cam_coord[:, 2],
                )

        img_indices = points_img.astype(np.int64)
        depth = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
        valid_depth = image_valid_mask & _valid_projected_mask(
            img_indices,
            image.size[1],
            image.size[0],
            depth=pts_cam_coord[:, 2],
        )
        depth[img_indices[valid_depth, 0], img_indices[valid_depth, 1]] = pts_cam_coord[valid_depth, 2] / 80.0

        if self.color_jitter is not None:
            image = self.color_jitter(image)
        image_np = np.array(image, dtype=np.float32, copy=False) / 255.0
        if np.random.rand() < self.fliplr:
            image_np = np.ascontiguousarray(np.fliplr(image_np))
            img_indices[:, 1] = image_np.shape[1] - 1 - img_indices[:, 1]
            depth = np.ascontiguousarray(np.fliplr(depth))
        if self.image_normalizer:
            mean, std = self.image_normalizer
            image_np = (image_np - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)

        coords = augment_and_scale_3d(points, self.scale, self.full_scale)
        coords = coords.astype(np.int64)
        idxs = (coords.min(1) >= 0) * (coords.max(1) < self.full_scale)
        image_chw = np.moveaxis(image_np, -1, 0)

        if self.use_color:
            feats = _sample_point_colors(image_chw, img_indices[idxs], image_valid_mask[idxs])
        else:
            feats = np.ones([np.sum(idxs), 1], np.float32)

        return {
            "coords": coords[idxs],
            "features_3d": feats,
            "labels_3d": labels[idxs],
            "image": image_chw,
            "sparse_depth": depth[None].astype(np.float32),
            "calib": None,
            "point2img": img_indices[idxs],
            "valid_mask": image_valid_mask[idxs],
            "scene_id": str(data_dict.get("scene_id", "")),
            "frame_id": str(data_dict.get("frame_id", index)),
            "dataset_name": "semantic_kitti",
        }


def _require_nuscenes_devkit() -> None:
    try:
        import nuscenes  # noqa: F401
        from nuscenes.nuscenes import NuScenes  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "NuScenes loading requires nuscenes-devkit. Install it with "
            "`pip install nuscenes-devkit` in the active training environment."
        ) from exc


def _valid_projected_mask(point_rc, height, width, depth=None):
    coords = np.asarray(point_rc)
    if coords.size == 0:
        return np.zeros((0,), dtype=bool)
    rows = coords[:, 0]
    cols = coords[:, 1]
    mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if depth is not None:
        mask = np.logical_and(mask, np.asarray(depth) > 0)
    return mask.astype(bool)


def _sample_point_colors(image_chw, img_indices, valid_mask):
    feats = np.zeros((len(img_indices), image_chw.shape[0]), dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.any():
        feats[valid] = image_chw[:, img_indices[valid, 0], img_indices[valid, 1]].T
    return feats


def _filter_kwargs(cls, kwargs: dict) -> dict:
    signature = inspect.signature(cls.__init__)
    allowed = set(signature.parameters.keys()) - {"self", "split"}
    return {key: value for key, value in kwargs.items() if key in allowed}


def _normalize_unidseg_kwargs(kwargs: dict) -> dict:
    tuple_keys = {
        "downsample",
        "crop_size",
        "rand_crop",
        "random_weather",
        "resize",
        "color_jitter",
        "image_normalizer",
        "pselab_paths",
    }
    out = dict(kwargs)
    for key in tuple_keys:
        value = out.get(key)
        if isinstance(value, list):
            out[key] = tuple(value)
    return out


class SyntheticDataset(Dataset):
    """Tiny deterministic dataset for smoke tests and CI without external data."""

    def __init__(self, length=4, num_points=256, image_size=(96, 160), num_classes=6, in_channels=1):
        self.length = int(length)
        self.num_points = int(num_points)
        self.image_size = tuple(image_size)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(index)
        height, width = self.image_size
        coords = rng.integers(0, 128, size=(self.num_points, 3), dtype=np.int64)
        feats = rng.normal(size=(self.num_points, self.in_channels)).astype(np.float32)
        labels = rng.integers(0, self.num_classes, size=(self.num_points,), dtype=np.int64)
        image = rng.random((3, height, width), dtype=np.float32)
        depth = np.zeros((1, height, width), dtype=np.float32)
        rows = rng.integers(0, height, size=(self.num_points,), dtype=np.int64)
        cols = rng.integers(0, width, size=(self.num_points,), dtype=np.int64)
        depth[0, rows, cols] = rng.random(self.num_points)
        return {
            "coords": coords,
            "features_3d": feats,
            "labels_3d": labels,
            "image": image,
            "sparse_depth": depth,
            "calib": None,
            "point2img": np.stack([rows, cols], axis=1),
            "valid_mask": np.ones(self.num_points, dtype=bool),
            "scene_id": "synthetic",
            "frame_id": f"{index:06d}",
            "dataset_name": "synthetic",
        }


def _file_size_gb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 ** 3)
    except OSError:
        return 0.0
