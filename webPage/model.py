import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

# EfficientNetGRU model
class EfficientNetGRU(nn.Module):
    def __init__(self, sequence_length=10, num_classes=2):
        super(EfficientNetGRU, self).__init__()
        self.sequence_length = sequence_length
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet_features = nn.Sequential(*list(self.efficientnet.children())[:-2])
        feature_size = self.efficientnet.classifier[1].in_features
        self.gru = nn.GRU(feature_size, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        assert seq_len == self.sequence_length, "Sequence length mismatch!"
        features = []
        for i in range(seq_len):
            frame_features = self.efficientnet_features(x[:, i])
            frame_features = frame_features.mean([2, 3])
            features.append(frame_features)
        features = torch.stack(features, dim=1)
        _, hidden = self.gru(features)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        output = self.fc(hidden)
        return output

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_from_cropped_images(image_dir, model, device, sequence_length=10):
    # Preprocess the frames
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frames = sorted(os.listdir(image_dir))[:sequence_length]
    sequence = []

    for frame in frames:
        frame_path = os.path.join(image_dir, frame)
        image = Image.open(frame_path).convert("RGB")
        image = transform(image)
        sequence.append(image)

    sequence = torch.stack(sequence).unsqueeze(0)  # Add batch dimension

    # Make prediction
    sequence = sequence.to(device)
    output = model(sequence)
    prediction = output.argmax(dim=1).cpu().numpy()[0]

    return "Original" if prediction == 0 else "Manipulated"
