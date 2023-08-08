# Imports here
import numpy as np
import torch

from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import sys

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type = str) 
    
    parser.add_argument('--save_dir', type = str, default = 'save_checkpoints/', 
                        help = 'Save model') 

    parser.add_argument('--arch', type = str, default = 'vgg', 
                        help = 'Model Type (vgg or densenet)')
    
    parser.add_argument('--learning_rate', type = int, default = 0.003, 
                        help = 'Learning Rate')
    
    parser.add_argument('--hidden_units', type = int, default = 256, 
                        help = 'Hidden layers')  
    
    parser.add_argument('--epochs', type = int, default = 2, 
                        help = 'Num. of epochs')
    
    parser.add_argument('--gpu', action='store_true', 
                        help = 'Use gpu')    
    
    in_arg = parser.parse_args()
    
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_transforms = transforms.Compose([transforms.Resize(224), transforms.RandomResizedCrop(224),                         transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)                    
    
   # TODO: Build and train your network
    if in_arg.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if in_arg.arch == 'vgg':
        model = models.vgg16(pretrained=True)
        input_nodes = 25088
    elif in_arg.arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_nodes = 2014
    

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(input_nodes, in_arg.hidden_units),
                                     nn.ReLU(),
                                     nn.Linear(in_arg.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)

    model.to(device);

    epochs = in_arg.epochs
    running_loss = 0
    steps = 0
    print_every=5
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()  
            
            running_loss += loss.item()
            if steps % print_every == 0:            
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()

    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'input_size': input_nodes,
                  'output_size': 102,
                  'class_to_idx': model.class_to_idx,
                  'epochs': epochs,
                  'optimizer_state': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'model' : model,
                  'train_datasets': train_datasets}

    torch.save(checkpoint, in_arg.save_dir + 'checkpoint.pth')    
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()