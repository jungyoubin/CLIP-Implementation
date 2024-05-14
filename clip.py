import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import BertModel, BertConfig

class ImageEncoder(nn.Module):
    def __init__(self, pretrained=None, trainable=True):
        super().__init__()
        self.model = resnet50(weights=pretrained)
        self.model.fc = nn.Identity()

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = BertModel.from_pretrained(model_name)
        else:
            config = BertConfig.from_pretrained(model_name)
            self.model = BertModel(config)

    def forward(self, input_ids, attention_mask=True):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

class ClipModel(nn.Module):
    def __init__(self, image_embedding_dim, text_embedding_dim, output_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder('bert-base-uncased')

        self.image_projection = nn.Linear(image_embedding_dim, output_dim)
        self.text_projection = nn.Linear(text_embedding_dim, output_dim)

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        image_embeddings = F.normalize(self.image_projection(image_features))

        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_embeddings = F.normalize(self.text_projection(text_features))
        logits = torch.matmul(image_embeddings, text_embeddings.T) * self.temperature

        return logits
