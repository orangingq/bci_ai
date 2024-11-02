import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, pretrained=False):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights='IMAGENET1K_V1' if pretrained else None, norm_layer=nn.InstanceNorm2d if not pretrained else None),
                            'resnet34': models.resnet34(weights='IMAGENET1K_V1' if pretrained else None, norm_layer=nn.InstanceNorm2d if not pretrained else None),
                            "resnet50": models.resnet50(weights='IMAGENET1K_V2' if pretrained else None, norm_layer=nn.InstanceNorm2d if not pretrained else None),}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        # remove the classification head of the base resnet model
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # add additional classification head
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x

