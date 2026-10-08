__all__ = ["VMambaMultiStageExtractor"]


def __getattr__(name):
    if name == "VMambaMultiStageExtractor":
        from .vmamba_multistage_extractor import VMambaMultiStageExtractor

        return VMambaMultiStageExtractor
    raise AttributeError("module '{}' has no attribute '{}'".format(__name__, name))
