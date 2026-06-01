from .vocabulary import Vocabulary, build_vocabulary, build_vocabulary_from_names

__all__ = [
    "build_dataloader",
    "Vocabulary",
    "build_vocabulary",
    "build_vocabulary_from_names",
]


def __getattr__(name):
    if name == "build_dataloader":
        from .build import build_dataloader

        return build_dataloader
    raise AttributeError(name)
