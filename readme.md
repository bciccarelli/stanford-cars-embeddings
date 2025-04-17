# stanford-car-embeddings

A repository implementing a Siamese network with EfficientNet-B0 to learn embeddings for car images and recommend visually similar vehicles.

## Overview

This project trains a Siamese network on the Stanford Cars Dataset using triplet loss to learn a compact embedding space where visually similar cars are clustered together. After training, the model generates embeddings for any input image, and a cosine similarity search retrieves similar cars.

## Example Usage

### BMW Query  
![Similar BMWs](assets/similar_bmw.png)

### SUV Query 
![Similar SUVs](assets/similar_suv.png)

## Usage

1. Train the Siamese network: `python car-embed.py`
2. Predict on an image: `python predict.py`

## Contributing

Contributions are welcome. Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.
