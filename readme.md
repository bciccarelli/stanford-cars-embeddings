# stanford-car-embeddings

A repository implementing a Siamese network with EfficientNet-B0 to learn embeddings for car images and recommend visually similar vehicles.

## Overview

This project trains a Siamese network on the Stanford Cars Dataset to learn a compact embedding space where visually similar cars are clustered together. After training, the model generates embeddings for any input image, and a nearest-neighbor search retrieves similar cars.

## Example Usage

### BMW Query  
![BMW Query Image](assets/bmw.png)

### Recommended Similar BMWs  
![Similar BMWs](assets/similar_bmw.png)

### SUV Query  
![SUV Query Image](assets/suv.png)

### Recommended Similar SUVs  
![Similar SUVs](assets/similar_suv.png)

## Installation

- Clone the repository  
- Create a virtual environment: python -m venv venv  
- Activate the environment (on MacOS or Linux): source venv/bin/activate  
- Activate the environment (on Windows): venv\\Scripts\\activate  
- Install dependencies: pip install -r requirements.txt

## Usage

1. Prepare and preprocess the Stanford Cars Dataset.  
2. Train the Siamese network: python train.py --data_path path/to/cars_dataset  
3. Generate embeddings for images: python embed.py --image path/to/image --output embedding.npy  
4. Build the FAISS index: python build_index.py --embeddings_dir embeddings/ --output index.faiss  
5. Search for similar images: python search.py --index index.faiss --query_embedding embedding.npy --top_k 5

## Repository Structure

- data/               Dataset files and example images  
- src/                Source code for model, training, and inference  
- notebooks/          Jupyter notebooks for experiments  
- models/             Saved model checkpoints  
- train.py            Script to train the Siamese network  
- embed.py            Script to compute embeddings for images  
- build_index.py      Script to build the FAISS index  
- search.py           Script to perform nearest-neighbor search  
- requirements.txt    Python dependencies  

## Contributing

Contributions are welcome. Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.
