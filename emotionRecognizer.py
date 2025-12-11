import torch
import torch.nn as nn
from torchvision import models, transforms
import time
from collections import Counter
import numpy as np
import cv2
class EmotionRecognizer:
    def __init__(self, model_path = r"data\best_model2.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"cuda available: {torch.cuda.is_available()}")

        class_names = ["angry", "default", "happy", "sad"]
        m_path = model_path
        model = models.resnet101(weights=None)
        
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load(m_path, map_location=device))
        model.to(device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((300, 300)),
            #transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            ),
        ])

        self.model = model
        self.class_names = class_names
        self.device = device
    def recognize_emotion(self, frame: np.ndarray):
        with torch.no_grad():
            pil = transforms.ToPILImage()(frame)

            # 전처리 및 추론
            input_tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]

            pred_idx = torch.argmax(probs).item()
            pred_label = self.class_names[pred_idx]

            return pred_label, probs.cpu()      
            
    def getClassNames(self):
        return self.class_names