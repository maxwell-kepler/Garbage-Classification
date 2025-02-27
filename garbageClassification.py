import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer 
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os


class GarbageClassifierFusion(nn.Module):
    def __init__(self, num_classes=4):
        super(GarbageClassifierFusion, self).__init__()
        
        # Image encoder (EfficientNet)
        self.image_encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.image_features = nn.Sequential(*list(self.image_encoder.children())[:-2])
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
        self.image_projector = nn.Sequential(
            nn.Linear(img_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3) #0.3 chosen empirically
        )
        
        # Text projection layer
        # Projects BERT features (768) to same common space (512) for fusion
        self.text_projector = nn.Sequential(
            nn.Linear(text_embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3) #0.3 chosen empirically
        )

        fusion_dim = 512 + 512 
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, images, text_inputs):
        # Process images through EfficientNet and projection
        img_features = self.image_features(images)
        img_features = self.avg_pool(img_features)
        img_features = torch.flatten(img_features, 1) 
        img_embeddings = self.image_projector(img_features)
        
        # Process text through DistilBERT and projection
        text_outputs = self.text_encoder(**text_inputs).last_hidden_state
        text_embeddings = torch.mean(text_outputs, dim=1)
        text_embeddings = self.text_projector(text_embeddings)
        
        fused_features = torch.cat([img_embeddings, text_embeddings], dim=1)
        output = self.fusion_layers(fused_features)
        return output
    
class MultimodalGarbageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dataset, tokenizer, max_length=32):
        self.image_dataset = image_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.image_dataset)
        
    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        file_path = self.image_dataset.samples[idx][0]
        description = self._get_description_from_filename(file_path)
        text_encoding = self.tokenizer(
            description,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'text_inputs': {
                'input_ids': text_encoding['input_ids'].squeeze(0),
                'attention_mask': text_encoding['attention_mask'].squeeze(0)
            },
            'label': label
        }
    
    def _get_description_from_filename(self, file_path):
        filename = file_path.split('/')[-1].split('.')[0]
        description = '_'.join(filename.split('_')[:-1])
        description = description.replace('_', ' ')
        materials = ['plastic', 'metal', 'paper', 'glass', 'aluminum']
        if not any(material in description for material in materials):
            class_name = self.image_dataset.classes[self.image_dataset.class_to_idx[file_path.split('/')[-2]]]
            description = f"{class_name} {description}"
        
        return description.lower()

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
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, text_inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val':
                scheduler.step(epoch_acc)
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'best_fusion_model.pth')
                    early_stopping_counter = 0
                    print("Saved model, reset early stopping counter")
                else:
                    early_stopping_counter += 1
                    print(f"EarlyStopping counter: {early_stopping_counter} out of {patience}")
                    
                if early_stopping_counter >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    return model
    
    return model

data_dir = "/work/TALC/enel645_2025w/garbage_data/"
train_dir = os.path.join(data_dir, "CVPR_2024_dataset_Train")
val_dir = os.path.join(data_dir, "CVPR_2024_dataset_Val")
test_dir = os.path.join(data_dir, "CVPR_2024_dataset_Test")


evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "val": evaluation_transform,
    "test": evaluation_transform
}

datasets = {
    "train": datasets.ImageFolder(train_dir, transform=transform["train"]),
    "val": datasets.ImageFolder(val_dir, transform=transform["val"]),
    "test": datasets.ImageFolder(test_dir, transform=transform["test"]),
}

model = GarbageClassifierFusion(num_classes=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_dataset = MultimodalGarbageDataset(datasets["train"], tokenizer)
val_dataset = MultimodalGarbageDataset(datasets["val"], tokenizer)
test_dataset = MultimodalGarbageDataset(datasets["test"], tokenizer)

# Create dataloaders
batch_size = 16 
dataloaders = {
    "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2),
    "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2),
    "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
}

print(f"Using device: {device}")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW([
    {'params': model.image_features[6:].parameters(), 'lr': 1e-4},
    {'params': model.image_projector.parameters(), 'lr': 1e-3},
    {'params': model.text_projector.parameters(), 'lr': 1e-3},
    {'params': model.fusion_layers.parameters(), 'lr': 1e-3}
], weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

# Train the model
model = train_model(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=10,
    patience=3
)

def test_fusion_model(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    print("\nEvaluating on test set")
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            labels = batch['label'].to(device)
            
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