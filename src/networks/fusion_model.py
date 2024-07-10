import torch
import torch.nn as nn
from enum import Enum

from src.networks.models.pcme import PCME

class InputType(Enum):
    A_B = 'A_B'
    AxB = 'AxB'

def freeze_model(m):
    for param in m.parameters():
        param.requires_grad = False
        
def unfreeze_model(m):
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
    
class VQAFusionModel(nn.Module):
    def __init__(self, device, base_model: PCME, img_features:int, txt_features:int, num_classes: int, hidden_sizes: list,
                 dropout_rate=0.0):
        super(VQAFusionModel, self).__init__()
        self.base_model = base_model
        self.device = device
        
        print(f'VQA Fusion Model device: {self.device}')
        
        cross_size = hidden_sizes[0]
        
        self.image_in = nn.Linear(img_features * base_model.embed_dim, cross_size)
        self.text_in = nn.Linear(txt_features * base_model.embed_dim, cross_size)
        
        layers = []
        for hidden_size in hidden_sizes[1:]:
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Tanh()),
            layers.append(nn.Linear(cross_size, hidden_size))
            cross_size = hidden_size
        self.features_extractor = nn.Sequential(*layers)
        
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Tanh(),
            nn.Linear(cross_size, num_classes),  # Final classification layer
        )
        
        self.to(self.device)
        
    def forward(self, batch):
        questions = batch['question']
        outputs = None
        images = batch['image'].to(self.device)
        #sub_images = batch['sub_images'].to(self.device)
        #print(f'types images: {type(images)}, sub_images: {type(sub_images)}')
        #print(f'shapes images: {images.shape}, sub_images: {sub_images.shape}')
        outputs = self.base_model.forward(images, [], questions, 0)
        image_features = outputs['image_features']
        caption_features = outputs['caption_features']
        #sub_images_features = self.base_model.image_forward(sub_images.view(-1, 3, 224, 224))['embedding']
        #sub_images_features = sub_images_features.view(-1, 4, self.base_model.embed_dim).transpose(0, 1)
        #question_type_features = self.base_model.text_forward([], batch['question_type'], 0)['embedding']
        #question_rest_features = self.base_model.text_forward([], batch['question_rest'], 0)['embedding']
        #print(f'image_features: {image_features.shape} sub_images_features[0]: {sub_images_features[0].shape}')
        #return self.forward_fusion([image_features]+[f for f in sub_images_features], [caption_features, question_type_features, question_rest_features])
        return self.forward_fusion([image_features], [caption_features])
    
    def forward_fusion(self, image_features, text_features):
        image_features = self.image_in(torch.cat(image_features, dim=1))
        text_features = self.text_in(torch.cat(text_features, dim=1))
        fused_features = image_features * text_features
        last_features = self.features_extractor(fused_features)
        return self.classifier_head(last_features), last_features
    
    def unfreeze_base_model(self):
        self.frozen_base_model = False
        unfreeze_model(self.base_model)
        
    def freeze_base_model(self):
        self.frozen_base_model = True
        freeze_model(self.base_model)
        
    def unfreeze_base_image_model(self):
        self.frozen_base_model = False
        unfreeze_model(self.base_model.img_enc)
        
    def freeze_base_image_model(self):
        self.frozen_base_model = True
        freeze_model(self.base_model.img_enc)
        
class LinearFusionModelCategorical(nn.Module):
    def __init__(self, base_model: PCME, num_features:int, num_classes: int, hidden_sizes: list, input_type: InputType,
                 dropout_rate=0.0):
        super(LinearFusionModelCategorical, self).__init__()
        self.base_model = base_model    
        input_type = InputType(input_type)
        self.input_type = input_type
        self.frozen_base_model = True
        freeze_model(base_model)
        self.device = next(self.base_model.parameters()).device

        layers = []
        input_size = base_model.embed_dim * num_features  # Input size to the first hidden layer
        if input_type == InputType.AxB:
            input_size = base_model.embed_dim
        #print(input_size)
        for hidden_size in hidden_sizes:
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size  # Update input size for the next layer
        self.features_extractor = nn.Sequential(*layers)

        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, num_classes)  # Final classification layer
        )
        
        self.to(self.device)
    
    def forward(self, batch):
        questions = batch['question']
        outputs = None
        if 'image_features' in batch: # use precalculated features if available
            outputs = self.forward_fusion(
                [batch['image_features'],
                batch['caption_features']]+batch['sub_images'])
        else:
            images = batch['image'].to(self.device)
            sub_images = batch['sub_images'].to(self.device)
        #print(f'types images: {type(images)}, sub_images: {type(sub_images)}')
        #print(f'shapes images: {images.shape}, sub_images: {sub_images.shape}')
        outputs = self.base_model.forward(images, [], questions, 0)
        image_features = outputs['image_features']
        caption_features = outputs['caption_features']
        sub_images_features = self.base_model.image_forward(sub_images.view(-1, 3, 224, 224))['embedding']
        sub_images_features = sub_images_features.view(-1, 4, self.base_model.embed_dim).transpose(0, 1)
        question_type_features = self.base_model.text_forward([], batch['question_type'], 0)['embedding']
        question_rest_features = self.base_model.text_forward([], batch['question_rest'], 0)['embedding']
        #print(f'image_features: {image_features.shape} sub_images_features[0]: {sub_images_features[0].shape}')
        return self.forward_fusion([image_features, caption_features, question_type_features, question_rest_features]+[f for f in sub_images_features])
    
    def forward_fusion(self, features_list):
        #print(image_features.shape, caption_features.shape)
        #for i, tensor in enumerate(features_list):
        #    print(f"Tensor {i} shape: {tensor.shape}")
        fused_features = None 
        if self.input_type == InputType.A_B: # Concatenation
            fused_features = torch.cat(features_list, dim=1)
        if self.input_type == InputType.AxB: # Element-wise multiplication
            fused_features = features_list[0]
            for i in range(1, len(features_list)):
                fused_features = fused_features * features_list[i]
        if fused_features is None:
            raise ValueError(f"input_type {self.input_type} is not supported in forward_fusion")
        #print(fused_features.shape)
        last_features = self.features_extractor(fused_features)
        return self.classifier_head(last_features), last_features
    
    def unfreeze_base_model(self):
        self.frozen_base_model = False
        unfreeze_model(self.base_model)
        
