import torch
import torch.nn as nn

from src.networks.models.pcme import PCME

def freeze_model(m):
    for param in m.parameters():
        param.requires_grad = False

# class LinearFusionModel(nn.Module):
#     def __init__(self, image_model, text_model, num_classes):
#         super(LinearFusionModel, self).__init__()
#         self.image_model = image_model
#         self.text_model = text_model
#         self.fc = nn.Linear(image_model.output_size + text_model.output_size, num_classes)

#     def forward(self, image_input, text_input):
#         image_features = self.image_model(image_input)
#         text_features = self.text_model(text_input)
#         fused_features = torch.cat((image_features, text_features), dim=1)
#         output = self.fc(fused_features)
#         return output

class LinearFusionModel(nn.Module):
    def __init__(self, base_model:PCME):
        super(LinearFusionModel, self).__init__()
        self.base_model = base_model
        device = next(self.base_model.parameters()).device 
        self.fc = nn.Linear(base_model.embed_dim *2 , base_model.embed_dim)
        self.to(device)
        # print(f"LinearFusionModel device {device}")
    
    def forward(self, images, sentences, captions_word, lengths):
        outputs = self.base_model.forward(images, sentences, captions_word, lengths)
        image_features = outputs['image_features']
        caption_features = outputs['caption_features']
        fused_features = torch.cat((image_features, caption_features), dim=1)
        output = self.fc(fused_features)
        return output
        