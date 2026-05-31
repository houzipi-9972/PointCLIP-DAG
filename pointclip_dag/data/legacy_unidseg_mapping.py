"""Legacy UniDSeg merged label spaces.

PointCLIP-DAG does not use these mappings by default. They are kept only for
reproducing old UniDSeg-style six-class baselines.
"""

VKITTI_SKITTI_LEGACY_6 = {
    0: "vegetation terrain",
    1: "building",
    2: "road",
    3: "object",
    4: "truck",
    5: "car",
}

NUSCENES_LEGACY_6 = {
    0: "vehicle",
    1: "driveable surface",
    2: "sidewalk",
    3: "terrain",
    4: "manmade",
    5: "vegetation",
}
