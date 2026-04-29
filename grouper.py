import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import cv2
from sklearn.cluster import DBSCAN

class ProductGrouper:
    def __init__(self):
        # Force PyTorch to use exactly 1 CPU thread to prevent OpenMP crashes
        torch.set_num_threads(1)
        
        # Use CPU by default to satisfy constraints
        self.device = torch.device("cpu")
        
        # Load pre-trained MobileNetV2 (headless feature extractor)
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        self.model.eval()
        self.model.to(self.device)
        
        # Standard ImageNet transforms for MobileNetV2
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        # Convert BGR (OpenCV) to RGB (Torchvision expects RGB)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract features and perform Global Average Pooling
            features = self.model(tensor)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            embedding = pooled.view(pooled.size(0), -1).squeeze().cpu().numpy()
            
        return embedding

    def group_products(self, image: np.ndarray, bboxes: list, eps: float = 0.4, min_samples: int = 1) -> list:
        """
        Groups detected products using MobileNetV2 embeddings and DBSCAN clustering.
        Args:
            image: Full image as NumPy array (BGR).
            bboxes: List of bounding boxes `[x1, y1, x2, y2]`.
            eps: DBSCAN maximum distance between two samples.
            min_samples: DBSCAN minimum samples in a neighborhood.
        Returns:
            List of group_ids corresponding to the order of bboxes.
        """
        if not bboxes:
            return []

        embeddings = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within image boundaries
            h, w = image.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            
            crop = image[y1:y2, x1:x2]
            
            # Handle invalid crops
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                # Add a dummy zero embedding if crop fails
                embeddings.append(np.zeros((1280,)))
                continue
                
            emb = self._extract_embedding(crop)
            embeddings.append(emb)

        embeddings_array = np.array(embeddings)
        
        # Normalize embeddings to use cosine distance effectively with euclidean metric
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1 # Avoid division by zero
        embeddings_normalized = embeddings_array / norms
        
        # Cluster embeddings using DBSCAN
        # eps=0.4 on L2-normalized vectors represents a cosine similarity threshold
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(embeddings_normalized)
        
        # Convert numpy integer types to standard python int for JSON serialization
        return [int(label) for label in clustering.labels_]
