import torch
import torch.nn as nn
from enum import Enum

from src.networks.models.pcme import PCME

class InputType(Enum):
    A_B = 'A_B'
    AxB = 'AxB'

def freeze_model(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False
        
def unfreeze_model(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True

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

class LinearFusionModelEmbedded(nn.Module):
    def __init__(self, base_model:PCME):
        super(LinearFusionModelEmbedded, self).__init__()
        self.base_model = base_model
        device = next(self.base_model.parameters()).device 
        self.fc = nn.Linear(base_model.embed_dim *2 , base_model.embed_dim)
        self.to(device)
    
    def forward(self, images, sentences, captions_word, lengths):
        outputs = self.base_model.forward(images, sentences, captions_word, lengths)
        image_features = outputs['image_features']
        caption_features = outputs['caption_features']
        fused_features = torch.cat((image_features, caption_features), dim=1)
        output = self.fc(fused_features)
        return output



class LinearFusionModelCategorical(nn.Module):
    def __init__(self, base_model: PCME, num_classes: int, hidden_sizes: list, input_type: InputType,
                 dropout_rate=0.0):
        super(LinearFusionModelCategorical, self).__init__()
        self.base_model = base_model    
        input_type = InputType(input_type)
        self.input_type = input_type
        self.frozen_base_model = True
        freeze_model(base_model)
        device = next(self.base_model.parameters()).device

        layers = []
        input_size = base_model.embed_dim * 2  # Input size to the first hidden layer
        if input_type == InputType.AxB:
            input_size = base_model.embed_dim
        #print(input_size)
        for hidden_size in hidden_sizes:
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size  # Update input size for the next layer

        # Add the final layer
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(input_size, num_classes))
        self.classifier = nn.Sequential(*layers)
        self.to(device)
    
    def forward(self, images, sentences, captions_word, lengths):
        outputs = self.base_model.forward(images, sentences, captions_word, lengths)
        image_features = outputs['image_features']
        caption_features = outputs['caption_features']
        return self.forward_fusion(image_features, caption_features)
    
    def forward_fusion(self, image_features, caption_features):
        #print(image_features.shape, caption_features.shape)
        fused_features = None 
        if self.input_type == InputType.A_B: # Concatenation
            fused_features = torch.cat((image_features, caption_features), dim=1)
        if self.input_type == InputType.AxB: # Element-wise multiplication
            fused_features = image_features * caption_features
        if fused_features is None:
            raise ValueError(f"input_type {self.input_type} is not supported in forward_fusion")
        #print(fused_features.shape)
        return self.classifier(fused_features)
    
    def unfreeze_base_model(self):
        self.frozen_base_model = False
        unfreeze_model(self.base_model)