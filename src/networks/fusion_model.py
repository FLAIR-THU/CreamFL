import torch

import torch.nn as nn

def freeze_model(m):
    for param in m.parameters():
        param.requires_grad = False

class LinearFusionModel(nn.Module):
    def __init__(self, image_model, text_model, num_classes):
        super(LinearFusionModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.fc = nn.Linear(image_model.output_size + text_model.output_size, num_classes)

    def forward(self, image_input, text_input):
        image_features = self.image_model(image_input)
        text_features = self.text_model(text_input)
        fused_features = torch.cat((image_features, text_features), dim=1)
        output = self.fc(fused_features)
        return output