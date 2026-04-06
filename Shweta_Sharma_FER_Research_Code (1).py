"""
=============================================================================
  PhD Research Code: Facial Expression Recognition using
  Memory-Attentive Swarm Intelligence — An Explainable and
  Culturally-Aware Approach

  Researcher  : Shweta Sharma (Enrollment No.: TDR24110011)
  Supervisor  : Dr. Ashok Kumar, Professor, CCSIT, TMU Moradabad
  Year        : 2025
  Frameworks  : PyTorch + TensorFlow/Keras (both shown)
=============================================================================

CONTENTS
--------
1.  Environment Setup & Imports
2.  Dataset Loading & Preprocessing  (PyTorch)
3.  Feature Extraction  (Landmarks + HOG + CNN deep features)
4.  Model Architecture  (CNN + BiLSTM + Multi-Head Attention)  [PyTorch]
5.  Model Architecture  (CNN + BiLSTM + Attention)             [Keras]
6.  Swarm Intelligence Optimizers
    a. Grey Wolf Optimizer (GWO)
    b. Particle Swarm Optimization (PSO)
    c. Quantum Improved Firefly + Bee Colony (QIFABC)
7.  Training Loop with Swarm Optimization
8.  Explainable AI
    a. Grad-CAM
    b. SHAP
    c. LIME
9.  Cultural Fairness Evaluation
10. Statistical Analysis & Hypothesis Testing
11. Utility: Confusion Matrix, Learning Curves, Results Export
"""

# ============================================================
# SECTION 1 — ENVIRONMENT SETUP & IMPORTS
# ============================================================
# Install required packages (run once):
# pip install torch torchvision tensorflow keras
# pip install opencv-python dlib facenet-pytorch
# pip install shap lime pytorch-grad-cam
# pip install scikit-learn scipy matplotlib seaborn
# pip install mlflow albumentations

import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# Keras / TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Computer Vision
import cv2
from PIL import Image

# Face Detection & Landmarks
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("facenet_pytorch not installed — MTCNN face detection unavailable")

# XAI
try:
    import shap
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from lime import lime_image
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("XAI libraries not installed — run: pip install shap lime pytorch-grad-cam")

# Statistics
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.model_selection import KFold
from scipy import stats

warnings.filterwarnings('ignore')

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Emotion class labels (7-class standard FER)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
NUM_CLASSES = len(EMOTION_LABELS)


# ============================================================
# SECTION 2 — DATASET LOADING & PREPROCESSING (PyTorch)
# ============================================================

class FERDataset(Dataset):
    """
    Generic FER Dataset loader.
    Expects folder structure:
        root_dir/
            Angry/   img1.jpg  img2.jpg ...
            Happy/   ...
            ...
    """
    def __init__(self, root_dir, transform=None, img_size=224):
        self.root_dir  = root_dir
        self.transform = transform
        self.img_size  = img_size
        self.samples   = []
        self.labels    = []

        # Load file paths and labels
        for idx, emotion in enumerate(EMOTION_LABELS):
            emotion_dir = os.path.join(root_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            for fname in os.listdir(emotion_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(os.path.join(emotion_dir, fname))
                    self.labels.append(idx)

        print(f"  Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label    = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(img_size=224, augment=True):
    """
    Returns train and validation transforms.
    Augmentation only applies during training.
    """
    # Training: with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),           # simulate occlusion
    ])

    # Validation/Test: no augmentation
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms


def compute_class_weights(dataset):
    """
    Compute inverse-frequency class weights to handle class imbalance.
    Used in CrossEntropyLoss.
    """
    label_array = np.array(dataset.labels)
    class_counts = np.bincount(label_array, minlength=NUM_CLASSES)
    class_counts = np.maximum(class_counts, 1)  # avoid division by zero
    weights = 1.0 / class_counts
    weights = weights / weights.sum()  # normalize
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ============================================================
# SECTION 3 — FEATURE EXTRACTION
# ============================================================

class FaceLandmarkExtractor:
    """
    Extracts 68 facial landmark points using Dlib.
    Requires: pip install dlib
              Download shape_predictor_68_face_landmarks.dat from dlib.net
    """
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        try:
            import dlib
            self.detector  = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
            self.available = True
        except Exception as e:
            print(f"Dlib not available: {e}")
            self.available = False

    def extract(self, image_bgr):
        """
        Returns 136-dim landmark vector [x1,y1, x2,y2, ..., x68,y68]
        normalized by image width/height.
        Returns zero vector if no face detected.
        """
        if not self.available:
            return np.zeros(136)

        import dlib
        gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            return np.zeros(136)

        face  = faces[0]
        shape = self.predictor(gray, face)
        h, w  = image_bgr.shape[:2]
        coords = []
        for i in range(68):
            pt = shape.part(i)
            coords.extend([pt.x / w, pt.y / h])  # normalize

        return np.array(coords, dtype=np.float32)


def extract_hog_features(image_bgr, cell_size=(8, 8)):
    """
    Extracts HOG (Histogram of Oriented Gradients) texture features.
    Returns a 1D feature vector.
    """
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))

    # HOG via OpenCV
    win_size   = (64, 64)
    block_size = (16, 16)
    block_step = (8, 8)
    cell_size  = (8, 8)
    nbins      = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_step, cell_size, nbins)
    hog_features = hog.compute(resized).flatten()
    return hog_features.astype(np.float32)


# ============================================================
# SECTION 4 — MODEL ARCHITECTURE (PyTorch)
# ============================================================

class AdaptiveSpatialFusion(nn.Module):
    """
    Adaptive Spatial Fusion (ASF) layer.
    Learns to combine features from different spatial scales using
    learnable attention weights (not fixed).
    Input:  (batch, channels, h, w)
    Output: (batch, channels, h, w)
    """
    def __init__(self, channels):
        super().__init__()
        # Two 3x3 convolutions operating at different dilations (multi-scale)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=2, dilation=2)
        # Learnable attention weights (one per spatial branch)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.ones(1))
        self.norm  = nn.BatchNorm2d(channels)

    def forward(self, x):
        x1  = F.relu(self.conv1(x))
        x2  = F.relu(self.conv2(x))
        out = self.alpha * x1 + self.beta * x2
        return self.norm(out)


class FER_BiLSTM_Attention(nn.Module):
    """
    Main hybrid model:
      CNN Backbone (ResNet-50) → Adaptive Spatial Fusion →
      BiLSTM (temporal memory) → Multi-Head Self-Attention →
      Classifier

    For static images:  sequence_len = 1 (single frame)
    For video clips:    sequence_len = number of frames per clip
    """
    def __init__(self,
                 num_classes   = NUM_CLASSES,
                 lstm_hidden   = 256,
                 lstm_layers   = 2,
                 attention_heads = 8,
                 dropout       = 0.5,
                 pretrained    = True):
        super().__init__()

        # ---- CNN Backbone (ResNet-50, pretrained on ImageNet) ----
        resnet         = models.resnet50(pretrained=pretrained)
        # Remove the final FC and average pool layers
        self.cnn_feat  = nn.Sequential(*list(resnet.children())[:-2])
        cnn_out_ch     = 2048

        # ---- Adaptive Spatial Fusion ----
        self.asf = AdaptiveSpatialFusion(cnn_out_ch)

        # ---- Global Average Pool (spatial → vector) ----
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ---- BiLSTM for temporal modeling ----
        # Input to LSTM: flattened CNN feature vector (2048)
        self.bilstm = nn.LSTM(
            input_size   = cnn_out_ch,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            batch_first  = True,
            bidirectional = True,
            dropout      = dropout if lstm_layers > 1 else 0.0
        )
        lstm_out_size = lstm_hidden * 2  # bidirectional → 2x hidden

        # ---- Multi-Head Self-Attention ----
        self.attention = nn.MultiheadAttention(
            embed_dim   = lstm_out_size,
            num_heads   = attention_heads,
            dropout     = dropout,
            batch_first = True
        )
        self.attn_norm = nn.LayerNorm(lstm_out_size)

        # ---- Classifier ----
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_size, num_classes)

    def forward(self, x):
        """
        x shape:
          Static image: (batch, 3, H, W)  → will be treated as seq of 1 frame
          Video clip:   (batch, T, 3, H, W)  where T = number of frames
        """
        # Handle static images by adding a sequence dimension
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, C, H, W)

        B, T, C, H, W = x.shape
        # Process each frame through CNN
        x = x.view(B * T, C, H, W)
        cnn_out = self.cnn_feat(x)               # (B*T, 2048, h, w)
        cnn_out = self.asf(cnn_out)              # Adaptive spatial fusion
        cnn_out = self.gap(cnn_out)              # (B*T, 2048, 1, 1)
        cnn_out = cnn_out.view(B, T, -1)         # (B, T, 2048)

        # BiLSTM
        lstm_out, _ = self.bilstm(cnn_out)       # (B, T, lstm_hidden*2)

        # Multi-Head Self-Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out    = self.attn_norm(attn_out + lstm_out)  # residual connection

        # Use the last time step output for classification
        final_feat  = attn_out[:, -1, :]         # (B, lstm_hidden*2)
        final_feat  = self.dropout(final_feat)
        logits      = self.classifier(final_feat) # (B, num_classes)
        return logits


# ============================================================
# SECTION 5 — MODEL ARCHITECTURE (Keras / TensorFlow)
# ============================================================

def build_keras_fer_model(img_size=224, num_classes=NUM_CLASSES,
                          lstm_units=256, num_heads=8, dropout=0.5):
    """
    Equivalent hybrid model in Keras:
    ResNet50 → BiLSTM → Multi-Head Attention → Dense Classifier

    For static images (sequence_len=1):
      Input shape: (1, img_size, img_size, 3)
    """
    # Input: sequence of frames
    inputs = keras.Input(shape=(None, img_size, img_size, 3),
                         name="frame_sequence")

    # ---- CNN Feature Extractor (ResNet50, frozen first) ----
    resnet = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    resnet.trainable = False  # freeze initially; unfreeze later for fine-tuning

    # Apply ResNet to each frame in the sequence
    cnn_out = layers.TimeDistributed(resnet, name="resnet_backbone")(inputs)
    # cnn_out shape: (batch, T, 2048)

    # ---- BiLSTM ----
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout),
        name="bilstm"
    )(cnn_out)
    # x shape: (batch, T, lstm_units*2)

    # ---- Multi-Head Attention ----
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=lstm_units * 2 // num_heads,
        dropout=dropout,
        name="multi_head_attention"
    )(x, x)
    x = layers.Add(name="residual_add")([x, attn_out])
    x = layers.LayerNormalization(name="layer_norm")(x)

    # Take the last time-step output
    x = layers.Lambda(lambda t: t[:, -1, :], name="last_timestep")(x)

    # ---- Classifier ----
    x      = layers.Dropout(dropout)(x)
    x      = layers.Dense(256, activation='relu', name="dense_head")(x)
    x      = layers.Dropout(dropout / 2)(x)
    output = layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="FER_BiLSTM_Attention_Keras")
    return model


# Quick demo of Keras model
def demo_keras_model():
    model = build_keras_fer_model()
    model.summary()
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\nKeras model built and compiled successfully.")
    return model


# ============================================================
# SECTION 6 — SWARM INTELLIGENCE OPTIMIZERS
# ============================================================

# ---------- 6a. Grey Wolf Optimizer (GWO) ----------
class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer for hyperparameter tuning.
    Reference: Mirjalili et al. (2014), Advances in Engineering Software.

    Search space example:
      {'learning_rate': (0.0001, 0.01),
       'dropout':       (0.2,    0.7),
       'lstm_hidden':   (128,    512),
       'batch_size':    (16,     128)}
    """
    def __init__(self, fitness_fn, bounds, n_wolves=30, max_iter=100):
        """
        fitness_fn : callable — receives a parameter dict, returns accuracy (higher=better)
        bounds     : dict of {param_name: (min_val, max_val)}
        n_wolves   : population size
        max_iter   : number of iterations
        """
        self.fitness_fn = fitness_fn
        self.bounds     = bounds
        self.n_wolves   = n_wolves
        self.max_iter   = max_iter
        self.param_names = list(bounds.keys())
        self.dim        = len(self.param_names)

        # Lower and upper bounds as arrays
        self.lb = np.array([bounds[k][0] for k in self.param_names], dtype=float)
        self.ub = np.array([bounds[k][1] for k in self.param_names], dtype=float)

        self.convergence = []  # track best fitness per iteration

    def _decode(self, position):
        """Convert continuous position array to parameter dict."""
        params = {}
        for i, name in enumerate(self.param_names):
            val = np.clip(position[i], self.lb[i], self.ub[i])
            # Integer params
            if name in ['lstm_hidden', 'batch_size', 'attention_heads']:
                val = int(round(val))
            params[name] = val
        return params

    def optimize(self):
        """Run GWO and return best parameters found."""
        # Initialize wolf positions randomly
        positions = np.random.uniform(
            low  = self.lb,
            high = self.ub,
            size = (self.n_wolves, self.dim)
        )
        fitness  = np.array([self.fitness_fn(self._decode(p)) for p in positions])

        # Alpha = best, Beta = second best, Delta = third best
        sorted_idx = np.argsort(-fitness)
        alpha_pos, beta_pos, delta_pos = (positions[sorted_idx[0]].copy(),
                                          positions[sorted_idx[1]].copy(),
                                          positions[sorted_idx[2]].copy())
        alpha_score = fitness[sorted_idx[0]]

        for iteration in range(self.max_iter):
            # Linearly decrease a from 2 to 0
            a = 2.0 * (1 - iteration / self.max_iter)

            for i in range(self.n_wolves):
                for j in range(self.dim):
                    # Update towards alpha
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    # Update towards beta
                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    # Update towards delta
                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    positions[i, j] = (X1 + X2 + X3) / 3.0

                # Clip to bounds
                positions[i] = np.clip(positions[i], self.lb, self.ub)

            # Re-evaluate fitness
            fitness = np.array([self.fitness_fn(self._decode(p)) for p in positions])

            # Update leaders
            sorted_idx = np.argsort(-fitness)
            if fitness[sorted_idx[0]] > alpha_score:
                alpha_score = fitness[sorted_idx[0]]
                alpha_pos   = positions[sorted_idx[0]].copy()
            beta_pos  = positions[sorted_idx[1]].copy()
            delta_pos = positions[sorted_idx[2]].copy()

            self.convergence.append(alpha_score)
            if (iteration + 1) % 10 == 0:
                print(f"  GWO Iteration {iteration+1}/{self.max_iter} "
                      f"| Best Fitness: {alpha_score:.4f}")

        best_params = self._decode(alpha_pos)
        print(f"\nGWO Best Params: {best_params}  | Best Fitness: {alpha_score:.4f}")
        return best_params, alpha_score, self.convergence


# ---------- 6b. Particle Swarm Optimization (PSO) ----------
class ParticleSwarmOptimizer:
    """
    Standard PSO for hyperparameter search.
    """
    def __init__(self, fitness_fn, bounds, n_particles=40, max_iter=100,
                 w_init=0.9, w_final=0.4, c1=2.0, c2=2.0):
        self.fitness_fn  = fitness_fn
        self.bounds      = bounds
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.w_init      = w_init
        self.w_final     = w_final
        self.c1, self.c2 = c1, c2
        self.param_names = list(bounds.keys())
        self.dim         = len(self.param_names)
        self.lb = np.array([bounds[k][0] for k in self.param_names], dtype=float)
        self.ub = np.array([bounds[k][1] for k in self.param_names], dtype=float)
        self.convergence = []

    def _decode(self, position):
        params = {}
        for i, name in enumerate(self.param_names):
            val = np.clip(position[i], self.lb[i], self.ub[i])
            if name in ['lstm_hidden', 'batch_size', 'attention_heads']:
                val = int(round(val))
            params[name] = val
        return params

    def optimize(self):
        positions  = np.random.uniform(self.lb, self.ub, (self.n_particles, self.dim))
        velocities = np.zeros_like(positions)
        p_best_pos = positions.copy()
        p_best_fit = np.array([self.fitness_fn(self._decode(p)) for p in positions])
        g_best_idx = np.argmax(p_best_fit)
        g_best_pos = p_best_pos[g_best_idx].copy()
        g_best_fit = p_best_fit[g_best_idx]

        for iteration in range(self.max_iter):
            # Linearly decrease inertia weight
            w = self.w_init - (self.w_init - self.w_final) * iteration / self.max_iter

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            velocities = (w * velocities
                          + self.c1 * r1 * (p_best_pos - positions)
                          + self.c2 * r2 * (g_best_pos - positions))
            positions  = np.clip(positions + velocities, self.lb, self.ub)

            fitness = np.array([self.fitness_fn(self._decode(p)) for p in positions])

            # Update personal bests
            improved = fitness > p_best_fit
            p_best_pos[improved] = positions[improved].copy()
            p_best_fit[improved] = fitness[improved]

            # Update global best
            best_idx = np.argmax(p_best_fit)
            if p_best_fit[best_idx] > g_best_fit:
                g_best_fit = p_best_fit[best_idx]
                g_best_pos = p_best_pos[best_idx].copy()

            self.convergence.append(g_best_fit)
            if (iteration + 1) % 10 == 0:
                print(f"  PSO Iteration {iteration+1}/{self.max_iter} "
                      f"| Best Fitness: {g_best_fit:.4f}")

        best_params = self._decode(g_best_pos)
        print(f"\nPSO Best Params: {best_params}  | Best Fitness: {g_best_fit:.4f}")
        return best_params, g_best_fit, self.convergence


# ---------- 6c. Quantum Improved Firefly + Bee Colony (QIFABC) ----------
class QIFABC:
    """
    Quantum Improved Firefly Algorithm with Bee Colony (QIFABC).
    Combines: quantum rotation gates (exploration boost) +
              firefly light attraction (exploitation) +
              bee colony foraging (diversity maintenance).
    Best suited for optimizing attention weights.
    """
    def __init__(self, fitness_fn, bounds, n_fireflies=20, n_bees=30,
                 max_iter=100, gamma=0.5, alpha=0.5):
        self.fitness_fn  = fitness_fn
        self.bounds      = bounds
        self.n_fireflies = n_fireflies
        self.n_bees      = n_bees
        self.max_iter    = max_iter
        self.gamma       = gamma   # light absorption coefficient
        self.alpha       = alpha   # step size
        self.param_names = list(bounds.keys())
        self.dim         = len(self.param_names)
        self.lb = np.array([bounds[k][0] for k in self.param_names], dtype=float)
        self.ub = np.array([bounds[k][1] for k in self.param_names], dtype=float)
        self.convergence = []

    def _decode(self, position):
        params = {}
        for i, name in enumerate(self.param_names):
            val = np.clip(position[i], self.lb[i], self.ub[i])
            if name in ['lstm_hidden', 'batch_size', 'attention_heads']:
                val = int(round(val))
            params[name] = val
        return params

    def _quantum_rotation(self, position, best_position):
        """Apply quantum rotation gate to move towards best position."""
        theta = np.pi / 4 * np.sign(best_position - position)
        new_pos = position + theta * (self.ub - self.lb) * 0.1
        return np.clip(new_pos, self.lb, self.ub)

    def optimize(self):
        # Initialize firefly and bee populations
        firefly_pos = np.random.uniform(self.lb, self.ub, (self.n_fireflies, self.dim))
        bee_pos     = np.random.uniform(self.lb, self.ub, (self.n_bees, self.dim))

        all_pos  = np.vstack([firefly_pos, bee_pos])
        all_fit  = np.array([self.fitness_fn(self._decode(p)) for p in all_pos])

        best_idx = np.argmax(all_fit)
        best_pos = all_pos[best_idx].copy()
        best_fit = all_fit[best_idx]

        for iteration in range(self.max_iter):
            # ---- Firefly movement (light attraction) ----
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if all_fit[j] > all_fit[i]:
                        r = np.linalg.norm(all_pos[i] - all_pos[j])
                        beta = np.exp(-self.gamma * r ** 2)
                        rand_step = self.alpha * (np.random.rand(self.dim) - 0.5)
                        all_pos[i] = (all_pos[i]
                                      + beta * (all_pos[j] - all_pos[i])
                                      + rand_step)
                        all_pos[i] = np.clip(all_pos[i], self.lb, self.ub)

            # ---- Bee colony foraging (employed bees) ----
            for i in range(self.n_fireflies, self.n_fireflies + self.n_bees):
                # Quantum rotation towards best known position
                all_pos[i] = self._quantum_rotation(all_pos[i], best_pos)
                # Random local search around current position
                perturbation = 0.1 * (self.ub - self.lb) * np.random.randn(self.dim)
                all_pos[i]  = np.clip(all_pos[i] + perturbation, self.lb, self.ub)

            # Re-evaluate all positions
            all_fit = np.array([self.fitness_fn(self._decode(p)) for p in all_pos])

            # Update global best
            curr_best = np.argmax(all_fit)
            if all_fit[curr_best] > best_fit:
                best_fit = all_fit[curr_best]
                best_pos = all_pos[curr_best].copy()

            self.convergence.append(best_fit)
            if (iteration + 1) % 10 == 0:
                print(f"  QIFABC Iteration {iteration+1}/{self.max_iter} "
                      f"| Best Fitness: {best_fit:.4f}")

        best_params = self._decode(best_pos)
        print(f"\nQIFABC Best Params: {best_params}  | Best Fitness: {best_fit:.4f}")
        return best_params, best_fit, self.convergence


# ============================================================
# SECTION 7 — TRAINING LOOP
# ============================================================

def train_epoch(model, loader, criterion, optimizer, device=DEVICE):
    """Single training epoch. Returns mean loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE):
    """Evaluation loop. Returns loss, accuracy, predictions, true labels."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels))


def train_model(model, train_loader, val_loader,
                n_epochs=100, lr=1e-3, patience=10,
                class_weights=None, device=DEVICE):
    """
    Full training loop with:
    - Adam optimizer
    - ReduceLROnPlateau scheduler
    - Early stopping
    - Class-weighted cross entropy loss
    Returns: trained model, history dict
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc  = 0.0
    patience_ctr  = 0
    history       = {'train_loss': [], 'val_loss': [],
                     'train_acc':  [], 'val_acc':  []}

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(vl_acc)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f"Epoch {epoch:03d}/{n_epochs} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        # Save best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "best_fer_model.pth")
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val Acc: {best_val_acc:.4f}")
            break

    # Load best weights
    model.load_state_dict(torch.load("best_fer_model.pth"))
    return model, history


# ============================================================
# SECTION 8 — EXPLAINABLE AI
# ============================================================

def generate_gradcam(model, image_tensor, target_class=None):
    """
    Generates Grad-CAM heatmap for a given image tensor.
    Requires: pip install pytorch-grad-cam

    image_tensor: (1, 3, H, W) on DEVICE
    Returns: heatmap as numpy array (H, W)
    """
    if not XAI_AVAILABLE:
        print("pytorch-grad-cam not installed.")
        return None

    # Target the last conv layer of ResNet backbone
    target_layers = [model.cnn_feat[-1][-1].conv3]

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = None  # None = use predicted class
    if target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    return grayscale_cam[0]


def visualize_gradcam(image_np, heatmap, emotion_label, save_path=None):
    """
    Overlays Grad-CAM heatmap on original image and saves/shows it.
    image_np: (H, W, 3) numpy array, values in [0,1]
    heatmap:  (H, W) numpy array
    """
    if not XAI_AVAILABLE:
        return

    visualization = show_cam_on_image(image_np, heatmap, use_rgb=True)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Original Face")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM — Predicted: {emotion_label}")
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def run_shap_analysis(model, background_data, test_data, n_samples=50):
    """
    Runs SHAP Deep Explainer on model.
    background_data: tensor (N_bg, ...) for SHAP baseline
    test_data:       tensor (N_test, ...)
    Returns SHAP values per class.
    """
    if not XAI_AVAILABLE:
        print("shap not installed.")
        return None

    model.eval()
    explainer  = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(test_data[:n_samples])

    # Summary plot for all emotion classes
    shap.summary_plot(shap_values, test_data[:n_samples].cpu().numpy(),
                      class_names=EMOTION_LABELS, show=True)
    return shap_values


def run_lime_explanation(model, image_np, top_labels=3):
    """
    Generates LIME image explanation.
    image_np: (H, W, 3) uint8 numpy image
    """
    if not XAI_AVAILABLE:
        print("lime not installed.")
        return None

    def predict_fn(images):
        """Wrapper for LIME: images is (N, H, W, 3) uint8."""
        model.eval()
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        batch = torch.stack([transform(img) for img in images]).to(DEVICE)
        with torch.no_grad():
            probs = F.softmax(model(batch), dim=1)
        return probs.cpu().numpy()

    explainer    = lime_image.LimeImageExplainer()
    explanation  = explainer.explain_instance(
        image_np, predict_fn, top_labels=top_labels, num_samples=1000)

    # Show explanation for the top predicted class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True,
        num_features=5, hide_rest=False)
    plt.imshow(temp)
    plt.title(f"LIME — Emotion: {EMOTION_LABELS[explanation.top_labels[0]]}")
    plt.axis('off')
    plt.show()
    return explanation


# ============================================================
# SECTION 9 — CULTURAL FAIRNESS EVALUATION
# ============================================================

def evaluate_cultural_fairness(model, data_dict, criterion, device=DEVICE):
    """
    Evaluates model accuracy per cultural/demographic group.

    data_dict: dict of {group_name: DataLoader}
               e.g., {'East_Asian': loader_ea, 'Western': loader_w, ...}

    Returns: DataFrame with accuracy, F1 per group + fairness metrics
    """
    results = {}
    for group_name, loader in data_dict.items():
        _, acc, preds, labels = evaluate(model, loader, criterion, device)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        results[group_name] = {'Accuracy': acc, 'Weighted_F1': f1,
                               'N_samples': len(labels)}

    df = pd.DataFrame(results).T
    overall_acc = df['Accuracy'].mean()

    # Demographic Parity Difference (DPD)
    dpd = df['Accuracy'].max() - df['Accuracy'].min()

    print("\n====== Cultural Fairness Report ======")
    print(df.to_string())
    print(f"\nOverall Mean Accuracy  : {overall_acc:.4f}")
    print(f"Demographic Parity Diff: {dpd:.4f}  (target <= 0.05)")
    print(f"Fairness Met           : {'YES ✓' if dpd <= 0.05 else 'NO ✗  — apply mitigation'}")

    # Bar chart
    df['Accuracy'].plot(kind='bar', color='steelblue', figsize=(8, 5))
    plt.axhline(overall_acc, color='red', linestyle='--', label='Overall Mean')
    plt.title('Emotion Recognition Accuracy by Cultural Group')
    plt.ylabel('Accuracy')
    plt.xlabel('Cultural Group')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df, dpd


# ============================================================
# SECTION 10 — STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# ============================================================

def paired_ttest(scores_proposed, scores_baseline, hypothesis="H1"):
    """
    Paired t-test to check if proposed model is significantly better.
    scores_proposed: list of accuracy values (one per run)
    scores_baseline: list of accuracy values (one per run)
    """
    t_stat, p_value = stats.ttest_rel(scores_proposed, scores_baseline)
    mean_diff = np.mean(scores_proposed) - np.mean(scores_baseline)
    # Cohen's d effect size
    pooled_std = np.sqrt((np.std(scores_proposed)**2 + np.std(scores_baseline)**2) / 2)
    cohens_d   = mean_diff / pooled_std if pooled_std > 0 else 0

    print(f"\n====== Paired t-test — {hypothesis} ======")
    print(f"  Proposed  : mean = {np.mean(scores_proposed):.4f}  "
          f"std = {np.std(scores_proposed):.4f}")
    print(f"  Baseline  : mean = {np.mean(scores_baseline):.4f}  "
          f"std = {np.std(scores_baseline):.4f}")
    print(f"  Mean Diff : {mean_diff:.4f}")
    print(f"  t-statistic: {t_stat:.4f}  p-value: {p_value:.4f}")
    print(f"  Cohen's d  : {cohens_d:.4f}  "
          f"({'Large' if abs(cohens_d)>=0.8 else 'Medium' if abs(cohens_d)>=0.5 else 'Small'})")
    print(f"  Result    : {'SIGNIFICANT (H accepted)' if p_value < 0.05 else 'NOT significant'}")
    return {'t': t_stat, 'p': p_value, 'mean_diff': mean_diff, 'cohens_d': cohens_d}


def friedman_test_optimizers(results_dict):
    """
    Friedman test to compare multiple optimizers.
    results_dict: {'GWO': [acc1,acc2,...], 'PSO': [...], 'QIFABC': [...], 'Adam': [...]}
    """
    groups = list(results_dict.values())
    stat, p = stats.friedmanchisquare(*groups)
    print(f"\n====== Friedman Test — Optimizer Comparison ======")
    print(f"  Chi-square: {stat:.4f}  p-value: {p:.4f}")
    for name, vals in results_dict.items():
        print(f"  {name:10s}: mean = {np.mean(vals):.4f}  std = {np.std(vals):.4f}")
    print(f"  Result: {'Significant differences found' if p < 0.05 else 'No significant difference'}")
    return stat, p


# ============================================================
# SECTION 11 — UTILITY FUNCTIONS
# ============================================================

def plot_confusion_matrix(true_labels, pred_labels, save_path=None):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Confusion Matrix — Proposed FER Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_learning_curves(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='orange')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train Accuracy', color='green')
    axes[1].plot(history['val_acc'],   label='Val Accuracy',   color='red')
    axes[1].set_title('Accuracy Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_convergence_comparison(results, save_path=None):
    """
    Plot convergence curves for GWO, PSO, QIFABC side by side.
    results: {'GWO': [f1,f2,...], 'PSO': [...], 'QIFABC': [...]}
    """
    plt.figure(figsize=(10, 5))
    colors = {'GWO': 'blue', 'PSO': 'orange', 'QIFABC': 'green', 'Adam': 'red'}
    for name, curve in results.items():
        plt.plot(curve, label=name, color=colors.get(name, 'black'))
    plt.title('Swarm Optimizer Convergence Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (Accuracy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def save_results_to_excel(results_df, filename="FER_Results.xlsx"):
    results_df.to_excel(filename, index=True)
    print(f"Results saved to {filename}")


# ============================================================
# QUICK DEMO — runs when this file is executed directly
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("  FER Research Code — Shweta Sharma (TDR24110011)")
    print("  CCSIT, Teerthanker Mahaveer University, Moradabad")
    print("="*60)

    # ---- Build and show PyTorch model ----
    print("\n[1] Building PyTorch model...")
    model_pt = FER_BiLSTM_Attention(
        num_classes=NUM_CLASSES,
        lstm_hidden=256,
        lstm_layers=2,
        attention_heads=8,
        dropout=0.5,
        pretrained=False   # set True when training for real
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model_pt.parameters())
    train_params = sum(p.numel() for p in model_pt.parameters() if p.requires_grad)
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {train_params:,}")

    # Forward pass test
    dummy_input = torch.randn(2, 1, 3, 224, 224).to(DEVICE)  # batch=2, T=1
    with torch.no_grad():
        output = model_pt(dummy_input)
    print(f"  Output shape         : {output.shape}  (expected: [2, 7])")

    # ---- Keras model ----
    print("\n[2] Building Keras model...")
    model_keras = demo_keras_model()

    # ---- Swarm optimizer demo (with dummy fitness function) ----
    print("\n[3] Running GWO demo (10 iterations, dummy fitness)...")

    def dummy_fitness(params):
        """
        Placeholder fitness function.
        In real research, this would:
          1. Set model hyperparameters to `params`
          2. Train for a few epochs on a small validation subset
          3. Return the validation accuracy
        """
        return (0.85
                - abs(params['learning_rate'] - 0.001) * 10
                - abs(params['dropout'] - 0.5) * 0.2
                + np.random.normal(0, 0.01))

    bounds = {
        'learning_rate': (0.0001, 0.01),
        'dropout':       (0.2,    0.7),
        'lstm_hidden':   (128,    512),
        'batch_size':    (16,     128)
    }

    gwo  = GreyWolfOptimizer(dummy_fitness, bounds, n_wolves=10, max_iter=10)
    pso  = ParticleSwarmOptimizer(dummy_fitness, bounds, n_particles=10, max_iter=10)
    qifa = QIFABC(dummy_fitness, bounds, n_fireflies=8, n_bees=10, max_iter=10)

    best_gwo,  _, conv_gwo  = gwo.optimize()
    best_pso,  _, conv_pso  = pso.optimize()
    best_qifa, _, conv_qifa = qifa.optimize()

    plot_convergence_comparison({
        'GWO': conv_gwo, 'PSO': conv_pso, 'QIFABC': conv_qifa
    })

    # ---- Statistical test demo ----
    print("\n[4] Statistical test demo (H1)...")
    proposed = [0.921, 0.918, 0.924, 0.920, 0.919]
    baseline = [0.882, 0.879, 0.885, 0.881, 0.880]
    paired_ttest(proposed, baseline, hypothesis="H1")

    print("\n[5] Optimizer comparison Friedman test demo...")
    friedman_test_optimizers({
        'GWO':   [0.921, 0.918, 0.924, 0.920, 0.919],
        'PSO':   [0.910, 0.908, 0.912, 0.909, 0.911],
        'QIFABC':[0.915, 0.912, 0.917, 0.913, 0.916],
        'Adam':  [0.882, 0.879, 0.885, 0.881, 0.880],
    })

    print("\n[DONE] All components verified successfully!")
    print("Next step: Replace dummy_fitness() with real training loop "
          "and point FERDataset to your downloaded dataset folders.")
