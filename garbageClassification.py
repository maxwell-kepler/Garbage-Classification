import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer 
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os


class GarbageClassifierFusion(nn.Module):
    """
    A multimodal neural network that fuses image and text features for garbage classification.
    Uses EfficientNet-B0 for image processing and DistilBERT for text processing.
    """
    def __init__(self, num_classes=4):
        super(GarbageClassifierFusion, self).__init__()
        
        # Image encoder (EfficientNet)
        # EfficientNet is chosen for its efficiency and performance on image classification tasks
        self.image_encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Extract all layers except the final classifier
        self.image_features = nn.Sequential(*list(self.image_encoder.children())[:-2])
        # Global average pooling to reduce spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        img_embedding_dim = 1280  # EfficientNet-B0's feature dimension after pooling
        
        # Text encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        text_embedding_dim = 768  # DistilBERT's hidden size for each token
        
        # Freeze first 6 layers of image encoder to preserve low-level feature extraction
        for param in self.image_features[:6].parameters():
            param.requires_grad = False
            
        # Freeze entire text encoder since garbage descriptions use standard language
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Image projection layer
        # Projects high-dimensional image features (1280) to common space (512)
        # This reduces dimensionality and helps create a shared representation space
        self.image_projector = nn.Sequential(
            nn.Linear(img_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3) #0.3 chosen empirically
        )
        
        # Text projection layer
        # Projects BERT features (768) to same common space (512) for fusion
        # Having the same dimension (512) for both modalities helps balance fusion
        self.text_projector = nn.Sequential(
            nn.Linear(text_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3) #0.3 chosen empirically
        )

        fusion_dim = 512 + 512 
        # Fusion layers for combining and processing the multimodal features
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512), # Normalizes activations for faster and more stable training
            nn.Dropout(0.4), # Higher dropout rate (0.4) in fusion layers for stronger regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, images, text_inputs):
        # Process images through EfficientNet and projection
        img_features = self.image_features(images)
        img_features = self.avg_pool(img_features) # Global average pooling
        img_features = torch.flatten(img_features, 1)  # Flatten to [batch_size, features]
        img_embeddings = self.image_projector(img_features) # Project to common space
        
        # Process text through DistilBERT and projection
        text_outputs = self.text_encoder(**text_inputs).last_hidden_state # The last_hidden_state contains contextualized word embeddings
        text_embeddings = torch.mean(text_outputs, dim=1) # Mean pooling over token dimension to get sentence embedding
        text_embeddings = self.text_projector(text_embeddings)
        
        # Concatenate both embeddings for multimodal fusion
        fused_features = torch.cat([img_embeddings, text_embeddings], dim=1)
        output = self.fusion_layers(fused_features)
        return output
    
class MultimodalGarbageDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for handling multimodal (image + text) garbage classification data.
    This dataset extracts text descriptions from filenames and prepares both image and text inputs for the model.
    """
    def __init__(self, image_dataset, tokenizer, max_length=32):
        self.image_dataset = image_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.image_dataset)
        
    def __getitem__(self, idx):
        # Get image and label from the base image dataset
        image, label = self.image_dataset[idx]

        # Extract text description from the filename
        file_path = self.image_dataset.samples[idx][0]
        description = self._get_description_from_filename(file_path)

        # Tokenize the text description for input to the text encoder
        text_encoding = self.tokenizer(
            description,
            max_length=self.max_length,
            padding='max_length', # Pad all sequences to max_length
            truncation=True, # Truncate sequences longer than max_length
            return_tensors='pt'
        )
        
        # Return all necessary components for the multimodal model
        return {
            'image': image,
            'text_inputs': {
                'input_ids': text_encoding['input_ids'].squeeze(0),
                'attention_mask': text_encoding['attention_mask'].squeeze(0)
            },
            'label': label
        }
    
    def _get_description_from_filename(self, file_path):
        filename = file_path.split('/')[-1].split('.')[0] # Get filename without extension
        description = '_'.join(filename.split('_')[:-1]) # Remove the ID
        description = description.replace('_', ' ') # Replace underscores with spaces
        return description.lower() # Convert to lowercase for consistency

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, patience=5):
    best_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for batch in dataloaders[phase]:
                images = batch['image'].to(device)
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }
                labels = batch['label'].to(device)
                
                optimizer.zero_grad() # Zero the parameter gradients
                
                 # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, text_inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Get statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # If in validation phase, check for early stopping and model saving
            if phase == 'val':
                scheduler.step(epoch_loss) # Step the learning rate scheduler based on validation loss
                
                # If validation loss improved, save model and reset counter
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_fusion_model.pth')
                    early_stopping_counter = 0
                    print("Saved model, reset early stopping counter")
                else:
                    # If no improvement, increment counter
                    early_stopping_counter += 1
                    print(f"EarlyStopping counter: {early_stopping_counter} out of {patience}")
                    
                # If no improvement for 'patience' epochs, stop training
                if early_stopping_counter >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    return model
    
    return model

data_dir = "/work/TALC/enel645_2025w/garbage_data/"
train_dir = os.path.join(data_dir, "CVPR_2024_dataset_Train")
val_dir = os.path.join(data_dir, "CVPR_2024_dataset_Val")
test_dir = os.path.join(data_dir, "CVPR_2024_dataset_Test")

# Standard evaluation transform, which resizes and normalizes with the ImageNet statistics
evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)), # EfficientNet-B0 expects 224x224 inputs
    transforms.ToTensor(), # Convert PIL image to tensor (0-1 range)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
])

# Data augmentation and preprocessing for each split
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)), # EfficientNet-B0 expects 224x224 inputs
        transforms.RandomHorizontalFlip(p=0.5), # Flip image horizontally with 50% probability
        transforms.RandomRotation(15), # Rotate image up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Randomly adjust brightness and contrast
        transforms.ToTensor(), # Convert PIL image to tensor (0-1 range)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
    ]),
    "val": evaluation_transform,
    "test": evaluation_transform
}

# Create dataset objects for each split
datasets = {
    "train": datasets.ImageFolder(train_dir, transform=transform["train"]),
    "val": datasets.ImageFolder(val_dir, transform=transform["val"]),
    "test": datasets.ImageFolder(test_dir, transform=transform["test"]),
}

# Initialize the multimodal fusion model
model = GarbageClassifierFusion(num_classes=4)

# Set up device, using the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Initialize tokenizer for text processing
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Create multimodal datasets for each split
train_dataset = MultimodalGarbageDataset(datasets["train"], tokenizer)
val_dataset = MultimodalGarbageDataset(datasets["val"], tokenizer)
test_dataset = MultimodalGarbageDataset(datasets["test"], tokenizer)

# Create dataloaders
batch_size = 16 
dataloaders = {
    "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2), # Shuffle training data
    "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),  # No need to shuffle 
    "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)  # No need to shuffle 
}

# Print dataset information
print(f"Using device: {device}")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Define loss function
criterion = nn.CrossEntropyLoss()

# Use AdamW as an optimizer
optimizer = optim.AdamW([
    {'params': model.image_features[6:].parameters(), 'lr': 1e-4},
    {'params': model.image_projector.parameters(), 'lr': 1e-3},
    {'params': model.text_projector.parameters(), 'lr': 1e-3},
    {'params': model.fusion_layers.parameters(), 'lr': 1e-3}
], weight_decay=0.01)

# Setup the learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

# Train the model
model = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler
)

def test_fusion_model(model, test_loader):
    # Get the device where the model is located
    device = next(model.parameters()).device

    # Set model to evaluation mode ensuring no dropout
    model.eval()

    running_corrects = 0
    all_preds = []
    all_labels = []

    print("\nEvaluating on test set")
    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            labels = batch['label'].to(device)
            
             # Get model predictions
            outputs = model(images, text_inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')

# Evaluate the model on the test set
model.load_state_dict(torch.load('best_fusion_model.pth', weights_only=True))
test_fusion_model(model, dataloaders["test"])