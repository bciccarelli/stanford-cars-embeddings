import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import faiss
import scipy.io
import kagglehub

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the SiameseNetwork architecture (same as in training script)
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

# Function to load and preprocess image
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor, img

# Find similar cars using FAISS index
def find_similar_cars(index, query_embedding, all_embeddings, all_labels, k=5):
    distances, indices = index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
    similar_embeddings = [all_embeddings[idx] for idx in indices[0]]
    similar_labels = [all_labels[idx] for idx in indices[0]]
    return distances[0], indices[0], similar_embeddings, similar_labels

# Class to load test dataset (simplified from the original StanfordCarsDataset)
class StanfordCarsDataset:
    def __init__(self, root_dir, path_to_meta, split='test', transform=None):
        self.root_dir = root_dir
        self.path_to_meta = path_to_meta
        self.transform = transform
        self.split = split
        
        # Load annotations
        cars_annos_test = os.path.join(self.path_to_meta, "cars_test_annos_withlabels (1).mat") 
        cars_annos_meta = os.path.join(self.path_to_meta, "devkit", "cars_meta.mat")
        self.cars_meta_mat = scipy.io.loadmat(cars_annos_meta)
        self.annotations_test = scipy.io.loadmat(cars_annos_test)

        class_names = [name[0] for name in self.cars_meta_mat['class_names'][0]]
        
        # Process annotations
        self.samples = []
        for anno in self.annotations_test['annotations'][0]:
            img_name = anno[5][0][-9:]
            file_path = os.path.join(root_dir, 'cars_' + split, 'cars_' + split, img_name)
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

# Function to display query image and similar cars
def plot_similar_cars(query_img, test_dataset, indices, distances, class_names):
    plt.figure(figsize=(15, 8))
    
    # Display query image
    plt.subplot(1, 6, 1)
    plt.imshow(query_img)
    plt.title("Query Image", fontsize=10)
    plt.axis('off')
    
    # Display top 5 similar cars
    for i in range(5):
        plt.subplot(1, 6, i+2)
        
        # Get the image from test dataset
        img_tensor, label = test_dataset[indices[i]]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        
        # Display car model name and similarity score
        car_name = class_names[test_dataset.samples[indices[i]][1]]
        title = f"#{i+1}: {car_name}\nScore: {1 - distances[i]:.2f}"
        plt.title(title, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('similar_cars.png', dpi=300)
    plt.show()

def main():
    # Fixed query image path
    query_image_path = "car.png"
    
    # Check if query image exists
    if not os.path.exists(query_image_path):
        print(f"Error: Query image {query_image_path} does not exist")
        return
    
    # Use same paths as in the original script
    path_to_stanford_cars = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")
    print(f"Dataset path: {path_to_stanford_cars}")
    
    path_to_meta = kagglehub.dataset_download("abdelrahmant11/standford-cars-dataset-meta")
    
    # Model and data paths
    model_path = 'siamese_car_model.pth'
    embeddings_path = 'car_embeddings.pkl'
    index_path = 'car_index.faiss'
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist")
        return
        
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings path {embeddings_path} does not exist")
        return
        
    if not os.path.exists(index_path):
        print(f"Error: Index path {index_path} does not exist")
        return
    
    # Load the trained model
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load car embeddings and labels
    with open(embeddings_path, 'rb') as f:
        test_embeddings, test_labels = pickle.load(f)
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    
    # Load and preprocess the query image
    query_tensor, query_img = load_image(query_image_path)
    query_tensor = query_tensor.to(device)
    
    # Generate embedding for the query image
    with torch.no_grad():
        query_embedding = model.forward_one(query_tensor).cpu().numpy()
    
    # Find similar cars
    distances, indices, similar_embeddings, similar_labels = find_similar_cars(
        index, query_embedding, test_embeddings, test_labels, k=5
    )
    
    print("Similar car indices:", indices)
    print("Similar car labels:", similar_labels)
    print("Similarity scores:", [1 - dist for dist in distances])  # Convert distance to similarity score
    
    # Create transforms for test dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load the test dataset
    test_dataset = StanfordCarsDataset(
        root_dir=path_to_stanford_cars,
        path_to_meta=path_to_meta,
        split='test',
        transform=test_transform
    )
    
    # Get class names from the test dataset
    class_names = test_dataset.class_names
    
    # Plot similar cars
    plot_similar_cars(query_img, test_dataset, indices, distances, class_names)

if __name__ == "__main__":
    main()