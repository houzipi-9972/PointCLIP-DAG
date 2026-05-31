from __future__ import annotations


RAW_LABEL_SPACES = {
    "virtual_kitti": {
        0: "terrain",
        1: "tree",
        2: "vegetation",
        3: "building",
        4: "road",
        5: "guardrail",
        6: "traffic sign",
        7: "traffic light",
        8: "pole",
        9: "misc",
        10: "truck",
        11: "car",
        12: "van",
        13: "dont care",
        99: "dont care",
    },
    "semantic_kitti": {
        0: "unlabeled",
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on rails",
        18: "truck",
        20: "other vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other ground",
        50: "building",
        51: "fence",
        52: "other structure",
        60: "lane marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic sign",
        99: "other object",
        252: "moving car",
        253: "moving bicyclist",
        254: "moving person",
        255: "moving motorcyclist",
        256: "moving on rails",
        257: "moving bus",
        258: "moving truck",
        259: "moving other vehicle",
    },
    "nuscenes": {
        0: "ignore",
        1: "barrier",
        2: "bicycle",
        3: "bus",
        4: "car",
        5: "construction vehicle",
        6: "motorcycle",
        7: "pedestrian",
        8: "traffic cone",
        9: "trailer",
        10: "truck",
        11: "driveable surface",
        12: "other flat",
        13: "sidewalk",
        14: "terrain",
        15: "manmade",
        16: "vegetation",
    },
}


ALIASES = {
    "SemanticKITTIRawSCN": "semantic_kitti",
    "SemanticKITTISCN": "semantic_kitti",
    "VirtualKITTIRawSCN": "virtual_kitti",
    "VirtualKITTISCN": "virtual_kitti",
    "NuScenesLidarSegSCN": "nuscenes",
    "SyntheticDataset": "synthetic",
}


def normalize_dataset_name(dataset_name: str) -> str:
    return ALIASES.get(dataset_name, dataset_name)


def build_raw_to_name(dataset_name: str) -> dict[int, str]:
    dataset_name = normalize_dataset_name(dataset_name)
    if dataset_name == "synthetic":
        return {}
    if dataset_name not in RAW_LABEL_SPACES:
        raise KeyError(f"Unknown raw label space: {dataset_name}")
    return dict(RAW_LABEL_SPACES[dataset_name])
