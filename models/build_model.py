from .stage9_model import Stage9Model


def build_model(args, num_classes):
    return Stage9Model(args=args, num_classes=num_classes)
