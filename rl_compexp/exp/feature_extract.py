import torch

from model import QNet


class HookLayer:
    def __init__(self):
        self.hook_modules = []
        self.features_blobs = []

    def hook_layer(self, layer: torch.nn.Module):
        self.hook_modules.append(layer.register_forward_hook(self.hook_feature))

    def hook_feature(self, module, inp, output):
        self.features_blobs.append(output.data.cpu().numpy())


def test_hook():
    model = QNet()
    hook = HookLayer()
    hook.hook_layer(layer=model.act2)
    inp = torch.randn(8)
    model(inp)
    print(hook.features_blobs[0].shape)


if __name__ == "__main__":
    test_hook()
