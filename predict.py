# Imports here
import numpy as np
import torch

from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
import json


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type = str, default='flowers/test/1/image_06743.jpg', help = 'image path') 
    parser.add_argument('checkpoint', type = str, default = 'save_checkpoints/checkpoint.pth', help = 'checkpoint path') 
    parser.add_argument('--top_k', type = int, default=5, help = 'top K most likely classes')
    parser.add_argument('--category_names', type = str, default='cat_to_name.json', 
                        help = 'Mapping of category to real name')
    parser.add_argument('--gpu', action='store_true',help='Use gpu') 
    
    in_arg = parser.parse_args()
    
    

    with open(in_arg.category_names, 'r') as f:
        class_to_idx = json.load(f)
    
    if in_arg.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    checkpoint = torch.load(in_arg.checkpoint)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer_state']
    model.load_state_dict(checkpoint['state_dict'])
    train_datasets = checkpoint['train_datasets']
    
    img_pil = Image.open(in_arg.input)
    
    # preprocess img
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    topk = in_arg.top_k
    
    img_tensor = preprocess(img_pil)
    # Flatten image
    img_tensor = np.expand_dims(img_tensor, 0)
    img_tensor = torch.from_numpy(img_tensor)
    model.eval()
    input_img = img_tensor.requires_grad_(False).to(device)
    logps = model.forward(input_img)
    prediction = F.softmax(logps, dim = 1)
    topk = prediction.cpu().topk(topk)
    top_predictions = (k.data.numpy().squeeze().tolist() for k in topk)
    probs, classes = top_predictions
    
    num = 1
    class_names = train_datasets.classes
    categories = [class_to_idx[class_names[name]] for name in classes]
    for category, prob in zip(categories, probs):
        print(f"{num}.Class Name: {category} with probability {prob}.")
        num+=1
    
if __name__ == "__main__":
    main()