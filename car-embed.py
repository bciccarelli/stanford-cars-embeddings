import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
import scipy.io
import kagglehub

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import faiss

import wandb

# Initialize a new run
wandb.init(project="stanford-cars", name="training-run-cosine")
config = wandb.config
config.learning_rate = 0.01
config.batch_size = 32
config.epochs = 1000
config.similarity = "cosine"  # Added to track the similarity metric used

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Data Loading and Preprocessing
class StanfordCarsDataset(Dataset):
    def __init__(self, root_dir, path_to_meta, split='train', transform=None):
        self.root_dir = root_dir
        self.path_to_meta = path_to_meta
        self.transform = transform
        self.split = split
        
        # Load annotations
        cars_annos_test = os.path.join(self.path_to_meta, "cars_test_annos_withlabels (1).mat") 
        cars_annos_train = os.path.join(self.path_to_meta, "devkit", "cars_train_annos.mat")
        cars_annos_meta = os.path.join(self.path_to_meta, "devkit", "cars_meta.mat")
        self.cars_meta_mat = scipy.io.loadmat(cars_annos_meta)
        self.annotations_test = scipy.io.loadmat(cars_annos_test)
        self.annotations_train = scipy.io.loadmat(cars_annos_train)

        class_names = [name[0] for name in self.cars_meta_mat['class_names'][0]]
        # Process annotations
        self.samples = []
        if self.split == 'test':
                    
            for anno in self.annotations_test['annotations'][0]:
                img_name = anno[5][0][-9:]
                file_path = os.path.join(root_dir, 'cars_' + split, 'cars_' + split, img_name)
                # Check if this is train or test image based on path
                is_test = 'test' in split
                if (self.split == 'test' and is_test) or (self.split == 'train' and not is_test):
                    class_id = int(anno[4][0][0]) - 1  # Convert to 0-indexed
                    self.samples.append((file_path, class_id))
        
        elif self.split == 'train':
            for anno in self.annotations_train['annotations'][0]:
                img_name = anno[5][0][-9:]
                file_path = os.path.join(root_dir, 'cars_' + split, 'cars_' + split, img_name)
                # Check if this is train or test image based on path
                is_test = 'test' in split
                if (self.split == 'test' and is_test) or (self.split == 'train' and not is_test):
                    class_id = int(anno[4][0][0]) - 1  # Convert to 0-indexed
                    self.samples.append((file_path, class_id))

        self.class_names = class_names
        print(f"Loaded {len(self.samples)} {split} samples with {len(class_names)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

class SiamesePairDataset(Dataset):
    def __init__(self, dataset, num_pairs=10000):
        self.dataset = dataset
        self.num_pairs = num_pairs
        
        # Create a dictionary mapping class IDs to sample indices
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(dataset.samples):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
            
        self.pairs = self._generate_pairs()
        
    def _generate_pairs(self):
        pairs = []
        # Generate positive pairs (same class)
        for _ in range(self.num_pairs // 2):
            label = random.choice(list(self.label_to_indices.keys()))
            if len(self.label_to_indices[label]) >= 2:
                idx1, idx2 = random.sample(self.label_to_indices[label], 2)
                pairs.append((idx1, idx2, 1))  # 1 indicates same class
                
        # Generate negative pairs (different classes)
        for _ in range(self.num_pairs // 2):
            label1, label2 = random.sample(list(self.label_to_indices.keys()), 2)
            idx1 = random.choice(self.label_to_indices[label1])
            idx2 = random.choice(self.label_to_indices[label2])
            pairs.append((idx1, idx2, 0))  # 0 indicates different classes
            
        random.shuffle(pairs)
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]
        img1, _ = self.dataset[idx1]
        img2, _ = self.dataset[idx2]
        
        return img1, img2, torch.FloatTensor([label])

# Define transformations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

# 2. Model Definition
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        
        # Load pretrained EfficientNet-B0
        base_model = models.efficientnet_b0(pretrained=True)
        
        # Remove the classifier
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Get the output dimension of the last conv layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension of EfficientNet-B0
        feature_dim = 1280
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward_one(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        # L2 normalize the embedding
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

# 3. Loss Functions - Modified to use cosine similarity
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        # Calculate cosine similarity (dot product of normalized vectors)
        cosine_similarity = torch.sum(output1 * output2, dim=1)
        
        # Convert similarity to distance (1 - similarity)
        # This maps: 1 (identical) -> 0 (no distance), -1 (opposite) -> 2 (max distance)
        cosine_distance = 1 - cosine_similarity
        
        # Contrastive loss
        # For similar pairs: minimize distance
        # For dissimilar pairs: push distance to be at least margin
        loss_contrastive = torch.mean(
            label * cosine_distance + 
            (1 - label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2)
        )
        
        return loss_contrastive

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # Calculate cosine similarities
        pos_sim = torch.sum(anchor * positive, dim=1)
        neg_sim = torch.sum(anchor * negative, dim=1)
        
        # Convert to distances
        pos_dist = 1 - pos_sim
        neg_dist = 1 - neg_sim
        
        # Triplet loss using cosine distance
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss

# 4. Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (img1, img2, label) in enumerate(progress_bar):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output1, output2 = model(img1, img2)
            
            # Calculate loss
            loss = criterion(output1, output2, label)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        wandb.log({"loss": epoch_loss, "epoch": epoch})
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
    
    return train_losses

# 5. Generate Embeddings for All Images
def generate_embeddings(model, data_loader):
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Generating embeddings"):
            images = images.to(device)
            embeddings = model.forward_one(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.concatenate(all_labels)
    
    return all_embeddings, all_labels

# 6. Nearest Neighbor Search using FAISS with Cosine Similarity
def build_index(embeddings):
    dim = embeddings.shape[1]
    # Use IndexFlatIP for Inner Product, which is equivalent to cosine similarity for normalized vectors
    index = faiss.IndexFlatIP(dim)
    # Since our vectors are already normalized in the model, we can use inner product directly
    index.add(embeddings.astype(np.float32))
    return index

def find_similar_cars(index, query_embedding, all_embeddings, all_labels, k=5):
    # For cosine similarity, higher values (closer to 1) are better
    similarities, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
    similar_embeddings = [all_embeddings[idx] for idx in indices[0]]
    similar_labels = [all_labels[idx] for idx in indices[0]]
    return similarities[0], indices[0], similar_embeddings, similar_labels

# Function to display some sample images
def show_samples(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        idx = random.randint(0, len(dataset)-1)
        img, label = dataset[idx]
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Class: {label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


# Main execution
def main():
    # Download and set path to dataset
    path_to_stanford_cars = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")
    print(f"Dataset path: {path_to_stanford_cars}")
        
    path_to_meta = kagglehub.dataset_download("abdelrahmant11/standford-cars-dataset-meta")

    # Create transforms
    train_transform, test_transform = get_transforms()
    
    # Create datasets
    train_dataset = StanfordCarsDataset(path_to_stanford_cars, path_to_meta, split='train', transform=train_transform)
    test_dataset = StanfordCarsDataset(path_to_stanford_cars, path_to_meta, split='test', transform=test_transform)
    
    # Show some sample images
    show_samples(train_dataset)
    
    # Create Siamese pairs
    train_siamese_dataset = SiamesePairDataset(train_dataset, num_pairs=10000)

    # Create data loaders
    train_loader = DataLoader(train_siamese_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model
    model = SiameseNetwork().to(device)
    
    # Define loss function and optimizer
    criterion = ContrastiveLoss(margin=0.5)  # Different margin for cosine similarity
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs=1000)
    
    wandb.finish()

    # Save the model
    torch.save(model.state_dict(), 'siamese_car_model_cosine.pth')

    # Test the model by generating embeddings and creating an index
    print("Generating embeddings for test dataset...")
    test_embeddings, test_labels = generate_embeddings(model, test_loader)
    
    # Build the index for fast similarity search
    print("Building FAISS index...")
    index = build_index(test_embeddings)
    
    # Example of finding similar cars for the first test image
    if len(test_embeddings) > 0:
        query_embedding = test_embeddings[0]
        similarities, indices, similar_embeddings, similar_labels = find_similar_cars(
            index, query_embedding, test_embeddings, test_labels
        )
        print("Query label:", test_labels[0])
        print("Similar labels:", similar_labels)
        print("Similarities (cosine):", similarities)


if __name__ == "__main__":
    main()