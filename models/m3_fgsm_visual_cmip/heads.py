import torch.nn as nn


def weights_init_kaiming(module):
    classname = module.__class__.__name__
    if "Linear" in classname:
        nn.init.kaiming_normal_(module.weight.data, a=0, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif "BatchNorm1d" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.01)
        nn.init.zeros_(module.bias.data)


def weights_init_classifier(module):
    classname = module.__class__.__name__
    if "Linear" in classname:
        nn.init.normal_(module.weight.data, 0, 0.001)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        return x.div(norm)


class RetrievalHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(RetrievalHead, self).__init__()
        self.x_pool_proj = nn.Linear(input_dim, 2048)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.l2norm = Normalize(2)
        self.x_pool_proj.apply(weights_init_kaiming)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def project_x_pool(self, video_feat):
        return self.x_pool_proj(video_feat)

    def forward(self, video_feat=None, override_x_pool=None):
        if override_x_pool is None:
            if video_feat is None:
                raise ValueError("RetrievalHead requires video_feat when override_x_pool is not provided.")
            x_pool = self.project_x_pool(video_feat)
        else:
            x_pool = override_x_pool
        bn_feat = self.bottleneck(x_pool)
        logits = self.classifier(bn_feat)
        test_feat = self.l2norm(bn_feat)
        return {
            "x_pool": x_pool,
            "bn_feat": bn_feat,
            "logits": logits,
            "test_feat": test_feat,
            "debug": {
                "video_feat_shape": None if video_feat is None else tuple(video_feat.shape),
                "video_feat_norm": 0.0 if video_feat is None else float(video_feat.norm(dim=1).mean().item()),
                "x_pool_shape": tuple(x_pool.shape),
                "x_pool_norm": float(x_pool.norm(dim=1).mean().item()),
                "bn_feat_shape": tuple(bn_feat.shape),
                "bn_feat_norm": float(bn_feat.norm(dim=1).mean().item()),
                "logits_shape": tuple(logits.shape),
                "test_feat_shape": tuple(test_feat.shape),
                "test_feat_norm_mean": float(test_feat.norm(dim=1).mean().item()),
            },
        }
