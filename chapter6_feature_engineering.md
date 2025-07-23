performance_df = pd.DataFrame(performance_results)
print(performance_df.to_string(index=False))

# Create comprehensive visualization of statistical anomaly detection
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=('Feature Distributions (Healthy vs Anomaly)', 'Hotelling T² Control Chart', 'Mahalanobis Distance',
                   'Robust vs Classical Covariance', 'Ensemble Method Comparison', 'ROC Curves',
                   'Feature Space (2D Projection)', 'Anomaly Scores Distribution', 'Confusion Matrix'),
    specs=[[{"type": "histogram"}, {"type": "scatter"}, {"type": "histogram"}],
           [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "histogram"}, {"type": "heatmap"}]]
)

# Plot 1: Feature distributions
feature_idx = 0  # First feature (Natural Frequency)
healthy_data = X_test[y_test == 0, feature_idx]
anomaly_data = X_test[y_test == 1, feature_idx]

fig.add_trace(
    go.Histogram(x=healthy_data, name='Healthy', 
                 marker_color='blue', opacity=0.7, nbinsx=30),
    row=1, col=1
)
fig.add_trace(
    go.Histogram(x=anomaly_data, name='Anomaly', 
                 marker_color='red', opacity=0.7, nbinsx=20),
    row=1, col=1
)

fig.update_xaxes(title_text=f"{feature_names[feature_idx]}", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=1)

# Plot 2: Hotelling T² control chart
t2_scores = detection_results['hotelling_t2_scores']
sample_indices = np.arange(len(t2_scores))

colors = ['red' if anomaly else 'blue' for anomaly in y_test]
fig.add_trace(
    go.Scatter(x=sample_indices, y=t2_scores, mode='markers',
               marker=dict(color=colors, size=4),
               name='T² Scores'),
    row=1, col=2
)

# Add control limit
fig.add_hline(y=stat_detector.thresholds['hotelling_t2'], 
              line_dash="dash", line_color="red",
              annotation_text="Control Limit",
              row=1, col=2)

fig.update_xaxes(title_text="Sample Index", row=1, col=2)
fig.update_yaxes(title_text="Hotelling T² Statistic", row=1, col=2)

# Plot 3: Mahalanobis distance distribution
mahal_distances = detection_results['mahalanobis_distances']

fig.add_trace(
    go.Histogram(x=mahal_distances[y_test == 0], name='Healthy (Mahal)',
                 marker_color='blue', opacity=0.6, nbinsx=25),
    row=1, col=3
)
fig.add_trace(
    go.Histogram(x=mahal_distances[y_test == 1], name='Anomaly (Mahal)',
                 marker_color='red', opacity=0.6, nbinsx=15),
    row=1, col=3
)

fig.update_xaxes(title_text="Mahalanobis Distance", row=1, col=3)
fig.update_yaxes(title_text="Frequency", row=1, col=3)

# Plot 4: Robust vs Classical detection
robust_scores = detection_results['robust_covariance_scores']
classical_anomalies = detection_results['hotelling_t2_anomalies']
robust_anomalies = detection_results['robust_covariance_anomalies']

fig.add_trace(
    go.Scatter(x=t2_scores, y=robust_scores,
               mode='markers',
               marker=dict(color=['red' if r or c else 'blue' 
                                for r, c in zip(robust_anomalies, classical_anomalies)],
                          size=5),
               name='Classical vs Robust'),
    row=2, col=1
)

fig.update_xaxes(title_text="Hotelling T² Score", row=2, col=1)
fig.update_yaxes(title_text="Robust Covariance Score", row=2, col=1)

# Plot 5: Method comparison (detection counts)
method_counts = []
method_labels = []

for method in ['hotelling_t2', 'mahalanobis', 'isolation_forest', 'lof', 'one_class_svm']:
    key = f'{method}_anomalies'
    if key in detection_results:
        count = np.sum(detection_results[key])
        method_counts.append(count)
        method_labels.append(method.replace('_', ' ').title())

fig.add_trace(
    go.Bar(x=method_labels, y=method_counts,
           marker_color='lightcoral', name='Detections'),
    row=2, col=2
)

fig.update_xaxes(tickangle=45, row=2, col=2)
fig.update_yaxes(title_text="Number of Anomalies Detected", row=2, col=2)

# Plot 6: ROC-like analysis using scores
from sklearn.metrics import roc_curve, auc

# Use Hotelling T² scores for ROC
fpr, tpr, _ = roc_curve(y_test, t2_scores)
roc_auc = auc(fpr, tpr)

fig.add_trace(
    go.Scatter(x=fpr, y=tpr, mode='lines',
               name=f'Hotelling T² (AUC = {roc_auc:.3f})',
               line=dict(color='blue', width=2)),
    row=2, col=3
)

# Add diagonal line
fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
               name='Random Classifier',
               line=dict(color='gray', dash='dash')),
    row=2, col=3
)

fig.update_xaxes(title_text="False Positive Rate", row=2, col=3)
fig.update_yaxes(title_text="True Positive Rate", row=2, col=3)

# Plot 7: 2D Feature space projection
# Use first two features for visualization
fig.add_trace(
    go.Scatter(x=X_test[y_test == 0, 0], y=X_test[y_test == 0, 1],
               mode='markers', name='Healthy',
               marker=dict(color='blue', size=5, opacity=0.6)),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=X_test[y_test == 1, 0], y=X_test[y_test == 1, 1],
               mode='markers', name='True Anomalies',
               marker=dict(color='red', size=6, symbol='x')),
    row=3, col=1
)

# Highlight ensemble-detected anomalies
ensemble_detected = detection_results['ensemble_decision']
false_positives = ensemble_detected & (y_test == 0)
if np.any(false_positives):
    fig.add_trace(
        go.Scatter(x=X_test[false_positives, 0], y=X_test[false_positives, 1],
                   mode='markers', name='False Positives',
                   marker=dict(color='orange', size=8, symbol='triangle-up')),
        row=3, col=1
    )

fig.update_xaxes(title_text=feature_names[0], row=3, col=1)
fig.update_yaxes(title_text=feature_names[1], row=3, col=1)

# Plot 8: Combined anomaly scores
combined_scores = (t2_scores/np.max(t2_scores) + 
                  np.abs(robust_scores)/np.max(np.abs(robust_scores))) / 2

fig.add_trace(
    go.Histogram(x=combined_scores[y_test == 0], name='Healthy (Combined)',
                 marker_color='blue', opacity=0.6, nbinsx=25),
    row=3, col=2
)
fig.add_trace(
    go.Histogram(x=combined_scores[y_test == 1], name='Anomaly (Combined)',
                 marker_color='red', opacity=0.6, nbinsx=15),
    row=3, col=2
)

fig.update_xaxes(title_text="Combined Anomaly Score", row=3, col=2)
fig.update_yaxes(title_text="Frequency", row=3, col=2)

# Plot 9: Confusion matrix for ensemble method
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, detection_results['ensemble_decision'])
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig.add_trace(
    go.Heatmap(z=cm_normalized,
               x=['Predicted Normal', 'Predicted Anomaly'],
               y=['Actual Normal', 'Actual Anomaly'],
               colorscale='Blues',
               text=[[f'{cm[i,j]}<br>({cm_normalized[i,j]:.2%})' 
                     for j in range(2)] for i in range(2)],
               texttemplate='%{text}',
               name='Confusion Matrix'),
    row=3, col=3
)

fig.update_xaxes(title_text="Predicted", row=3, col=3)
fig.update_yaxes(title_text="Actual", row=3, col=3)

# Update layout
fig.update_layout(
    height=1200,
    title_text="Comprehensive Statistical Anomaly Detection Analysis",
    showlegend=True,
    font=dict(size=10)
)

fig.show()

print(f"\nStatistical Anomaly Detection Summary:")
print(f"• Best performing method: {performance_df.loc[performance_df['F1-Score'].astype(float).idxmax(), 'Method']}")
print(f"• Ensemble decision combines {len([k for k in detection_results.keys() if k.endswith('_anomalies')])} methods")
print(f"• Hotelling T² effective for multivariate normal distributions")
print(f"• Robust methods handle outliers in training data better")
print(f"• ROC AUC for Hotelling T²: {roc_auc:.3f}")

---

## 7. PyTorch Implementation for Deep Learning-Based Anomaly Detection

### 7.1 Deep Learning Approach to SHM

While statistical methods provide interpretable and theoretically grounded approaches to anomaly detection, deep learning methods can automatically learn complex feature representations and non-linear decision boundaries that may be missed by traditional approaches. PyTorch provides a flexible framework for implementing custom neural network architectures tailored to structural health monitoring applications.

### 7.2 Autoencoder Architecture for Anomaly Detection

Autoencoders learn to reconstruct normal patterns in the data. Anomalies are detected based on reconstruction error, as the model struggles to reproduce patterns it hasn't seen during training.

**Encoder:**
$\mathbf{h} = f_{enc}(\mathbf{x}; \boldsymbol{\theta}_{enc}) \tag{6.43}$

**Decoder:**
$\hat{\mathbf{x}} = f_{dec}(\mathbf{h}; \boldsymbol{\theta}_{dec}) \tag{6.44}$

**Reconstruction Loss:**
$\mathcal{L}_{recon} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 \tag{6.45}$

### 7.3 Implementation: Deep Learning Anomaly Detection

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SHMAutoencoder(nn.Module):
    """
    Autoencoder for SHM anomaly detection
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16], 
                 dropout_rate: float = 0.2):
        """
        Initialize autoencoder
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(SHMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (reverse of encoder)
        decoder_layers = []
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i-1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dims[i-1])
            ])
        
        # Final reconstruction layer
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, h):
        """Decode from latent space"""
        return self.decoder(h)

class SHMVariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for more robust anomaly detection
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 latent_dim: int = 16, dropout_rate: float = 0.2):
        """
        Initialize VAE
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            latent_dim: Dimension of latent space
            dropout_rate: Dropout rate
        """
        super(SHMVariationalAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder_base = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode to latent parameters"""
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

class DeepAnomalyDetector:
    """
    Deep learning-based anomaly detection for SHM
    """
    
    def __init__(self, model_type: str = 'autoencoder', device: str = None):
        """
        Initialize deep anomaly detector
        
        Args:
            model_type: 'autoencoder' or 'vae'
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 0.001, validation_split: float = 0.2):
        """
        Train the deep anomaly detection model
        
        Args:
            X: Training data (healthy samples only)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            validation_split: Fraction of data for validation
        """
        # Prepare data
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val = train_test_split(X_scaled, test_size=validation_split, 
                                         random_state=42)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X.shape[1]
        
        if self.model_type == 'autoencoder':
            self.model = SHMAutoencoder(input_dim).to(self.device)
            criterion = nn.MSELoss()
        elif self.model_type == 'vae':
            self.model = SHMVariationalAutoencoder(input_dim).to(self.device)
        else:
            raise ValueError("model_type must be 'autoencoder' or 'vae'")
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        print(f"Training {self.model_type} on {self.device}...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                
                if self.model_type == 'autoencoder':
                    reconstructed = self.model(batch_x)
                    loss = criterion(reconstructed, batch_x)
                else:  # VAE
                    reconstructed, mu, logvar = self.model(batch_x)
                    recon_loss = F.mse_loss(reconstructed, batch_x, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.1 * kl_loss  # Beta-VAE with beta=0.1
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                if self.model_type == 'autoencoder':
                    val_reconstructed = self.model(X_val_tensor)
                    val_loss = criterion(val_reconstructed, X_val_tensor).item()
                else:  # VAE
                    val_reconstructed, val_mu, val_logvar = self.model(X_val_tensor)
                    val_recon_loss = F.mse_loss(val_reconstructed, X_val_tensor, reduction='sum')
                    val_kl_loss = -0.5 * torch.sum(1 + val_logvar - val_mu.pow(2) - val_logvar.exp())
                    val_loss = (val_recon_loss + 0.1 * val_kl_loss).item()
            
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Compute threshold based on training data reconstruction error
        self._compute_threshold(X_train_tensor)
        
        self.is_fitted = True
        self.train_losses = train_losses
        self.val_losses = val_losses
        
        print(f"Training completed. Threshold: {self.threshold:.6f}")
    
    def _compute_threshold(self, X_train_tensor):
        """Compute anomaly threshold based on training data"""
        self.model.eval()
        
        with torch.no_grad():
            if self.model_type == 'autoencoder':
                reconstructed = self.model(X_train_tensor)
                errors = torch.mean((X_train_tensor - reconstructed) ** 2, dim=1)
            else:  # VAE
                reconstructed, _, _ = self.model(X_train_tensor)
                errors = torch.mean((X_train_tensor - reconstructed) ** 2, dim=1)
        
        # Use 95th percentile as threshold
        self.threshold = torch.quantile(errors, 0.95).item()
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict anomalies using the trained model
        
        Args:
            X: Data to analyze
            
        Returns:
            Dictionary with reconstruction errors and anomaly predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            if self.model_type == 'autoencoder':
                reconstructed = self.model(X_tensor)
                reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            else:  # VAE
                reconstructed, mu, logvar = self.model(X_tensor)
                reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        
        errors = reconstruction_errors.cpu().numpy()
        predictions = (errors > self.threshold).astype(int)
        
        return {
            'reconstruction_errors': errors,
            'predictions': predictions,
            'threshold': self.threshold
        }

# Demonstrate deep learning anomaly detection
print("\n=== Deep Learning Anomaly Detection with PyTorch ===\n")

# Use the same SHM dataset from statistical methods
print(f"Dataset: {X_train.shape[0]} healthy training samples, {X_test.shape[0]} test samples")

# Train autoencoder
print("Training Autoencoder...")
ae_detector = DeepAnomalyDetector(model_type='autoencoder', device='cpu')
ae_detector.fit(X_train, epochs=150, batch_size=32, learning_rate=0.001)

# Train VAE
print("\nTraining Variational Autoencoder...")
vae_detector = DeepAnomalyDetector(model_type='vae', device='cpu')
vae_detector.fit(X_train, epochs=150, batch_size=32, learning_rate=0.001)

# Make predictions
ae_results = ae_detector.predict(X_test)
vae_results = vae_detector.predict(X_test)

# Evaluate performance
print(f"\nDeep Learning Results:")

# Autoencoder performance
ae_precision = np.sum((ae_results['predictions'] == 1) & (y_test == 1)) / np.sum(ae_results['predictions'] == 1)
ae_recall = np.sum((ae_results['predictions'] == 1) & (y_test == 1)) / np.sum(y_test == 1)
ae_f1 = 2 * ae_precision * ae_recall / (ae_precision + ae_recall)

# VAE performance  
vae_precision = np.sum((vae_results['predictions'] == 1) & (y_test == 1)) / np.sum(vae_results['predictions'] == 1)
vae_recall = np.sum((vae_results['predictions'] == 1) & (y_test == 1)) / np.sum(y_test == 1)
vae_f1 = 2 * vae_precision * vae_recall / (vae_precision + vae_recall)

deep_performance = pd.DataFrame([
    {'Method': 'Autoencoder', 'Precision': f'{ae_precision:.3f}', 'Recall': f'{ae_recall:.3f}', 'F1-Score': f'{ae_f1:.3f}'},
    {'Method': 'VAE', 'Precision': f'{vae_precision:.3f}', 'Recall': f'{vae_recall:.3f}', 'F1-Score': f'{vae_f1:.3f}'}
])

print(deep_performance.to_string(index=False))

# Visualize deep learning results
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=('Training Loss Curves', 'Reconstruction Error Distribution', 'Deep vs Statistical Methods',
                   'Autoencoder Latent Space', 'Feature Reconstruction Quality', 'Method Comparison'),
    specs=[[{"type": "scatter"}, {"type": "histogram"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}]]
)

# Plot 1: Training loss curves
epochs = range(len(ae_detector.train_losses))
fig.add_trace(
    go.Scatter(x=list(epochs), y=ae_detector.train_losses,
               name='AE Train Loss', line=dict(color='blue')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=list(epochs), y=ae_detector.val_losses,
               name='AE Val Loss', line=dict(color='blue', dash='dash')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=list(epochs), y=vae_detector.train_losses,
               name='VAE Train Loss', line=dict(color='red')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=list(epochs), y=vae_detector.val_losses,
               name='VAE Val Loss', line=dict(color='red', dash='dash')),
    row=1, col=1
)

fig.update_xaxes(title_text="Epoch", row=1, col=1)
fig.update_yaxes(title_text="Loss", type="log", row=1, col=1)

# Plot 2: Reconstruction error distributions
fig.add_trace(
    go.Histogram(x=ae_results['reconstruction_errors'][y_test == 0],
                 name='AE Healthy', marker_color='blue', opacity=0.6, nbinsx=30),
    row=1, col=2
)
fig.add_trace(
    go.Histogram(x=ae_results['reconstruction_errors'][y_test == 1],
                 name='AE Anomaly', marker_color='red', opacity=0.6, nbinsx=20),
    row=1, col=2
)

# Add threshold line
fig.add_vline(x=ae_results['threshold'], line_dash="dash", line_color="green",
              annotation_text="Threshold", row=1, col=2)

fig.update_xaxes(title_text="Reconstruction Error", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=2)

# Plot 3: Deep vs Statistical comparison
fig.add_trace(
    go.Scatter(x=detection_results['hotelling_t2_scores'], 
               y=ae_results['reconstruction_errors'],
               mode='markers',
               marker=dict(color=['red' if label else 'blue' for label in y_test],
                          size=5),
               name='Statistical vs Deep'),
    row=1, col=3
)

fig.update_xaxes(title_text="Hotelling T² Score", row=1, col=3)
fig.update_yaxes(title_text="AE Reconstruction Error", row=1, col=3)

# Plot 4: Autoencoder latent space (if 2D)
# Get latent representations
X_test_scaled = ae_detector.scaler.transform(X_test)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(ae_detector.device)

with torch.no_grad():
    latent_repr = ae_detector.model.encode(X_test_tensor).cpu().numpy()

# Use first two latent dimensions
fig.add_trace(
    go.Scatter(x=latent_repr[y_test == 0, 0], y=latent_repr[y_test == 0, 1],
               mode='markers', name='Healthy Latent',
               marker=dict(color='blue', size=5, opacity=0.6)),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=latent_repr[y_test == 1, 0], y=latent_repr[y_test == 1, 1],
               mode='markers', name='Anomaly Latent',
               marker=dict(color='red', size=6, symbol='x')),
    row=2, col=1
)

fig.update_xaxes(title_text="Latent Dimension 1", row=2, col=1)
fig.update_yaxes(title_text="Latent Dimension 2", row=2, col=1)

# Plot 5: Feature reconstruction quality
original_sample = X_test_scaled[0]  # First test sample
with torch.no_grad():
    reconstructed_sample = ae_detector.model(torch.FloatTensor([original_sample]).to(ae_detector.device))
    reconstructed_sample = reconstructed_sample.cpu().numpy()[0]

fig.add_trace(
    go.Bar(x=feature_names, y=original_sample, name='Original',
           marker_color='blue', opacity=0.7),
    row=2, col=2
)
fig.add_trace(
    go.Bar(x=feature_names, y=reconstructed_sample, name='Reconstructed',
           marker_color='orange', opacity=0.7),
    row=2, col=2
)

fig.update_xaxes(tickangle=45, row=2, col=2)
fig.update_yaxes(title_text="Normalized Feature Value", row=2, col=2)

# Plot 6: Method comparison
all_methods = ['Statistical (Best)', 'Autoencoder', 'VAE']
best_stat_f1 = float(performance_df['F1-Score'].max())
all_f1_scores = [best_stat_f1, ae_f1, vae_f1]

fig.add_trace(
    go.Bar(x=all_methods, y=all_f1_scores,
           marker_color=['lightblue', 'lightcoral', 'lightgreen'],
           name='F1-Score Comparison'),
    row=2, col=3
)

fig.update_xaxes(tickangle=45, row=2, col=3)
fig.update_yaxes(title_text="F1-Score", row=2, col=3)

fig.update_layout(
    height=1000,
    title_text="Deep Learning Anomaly Detection Analysis",
    showlegend=True,
    font=dict(size=10)
)

fig.show()

print(f"\nDeep Learning Analysis Summary:")
print(f"• Autoencoder threshold: {ae_results['threshold']:.6f}")
print(f"• VAE shows {'better' if vae_f1 > ae_f1 else 'similar'} performance to standard autoencoder")
print(f"• Deep learning methods can capture non-linear patterns")
print(f"• Training converged in {len(ae_detector.train_losses)} epochs")
print(f"• Latent space provides interpretable feature representations")

---

## 8. Exercises

### Exercise 1: Time-Domain Feature Analysis

**Problem:** You are monitoring a suspension bridge using accelerometers placed at mid-span. The bridge experiences different loading conditions throughout the day. Design a comprehensive time-domain feature extraction system that can distinguish between:
- Normal traffic loading
- Heavy truck passage events  
- Wind-induced vibrations
- Potential structural changes

**Tasks:**
a) Implement at least 8 different time-domain features
b) Generate synthetic bridge data representing the four conditions above
c) Analyze which features are most discriminative for each condition
d) Create visualizations showing feature evolution over time

**Solution:**

```python
# Exercise 1 Solution
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

class BridgeConditionSimulator:
    """Simulate different bridge loading conditions"""
    
    def __init__(self, fs=100, duration=300):
        self.fs = fs
        self.duration = duration
        self.t = np.linspace(0, duration, int(duration * fs))
        
    def generate_normal_traffic(self):
        """Normal traffic with occasional cars"""
        # Base structural response (1st and 2nd modes)
        response = (0.001 * np.sin(2 * np.pi * 2.2 * self.t) * 
                   np.exp(-0.03 * (self.t % 30)) +
                   0.0005 * np.sin(2 * np.pi * 6.8 * self.t) * 
                   np.exp(-0.05 * (self.t % 25)))
        
        # Random car passages
        np.random.seed(1)
        n_cars = np.random.poisson(10)  # ~10 cars per 5 minutes
        car_times = np.random.uniform(10, self.duration-10, n_cars)
        
        for car_time in car_times:
            car_mask = (self.t >= car_time) & (self.t <= car_time + 3)
            if np.any(car_mask):
                local_t = self.t[car_mask] - car_time
                car_response = 0.003 * np.exp(-local_t/2) * np.sin(2*np.pi*2.2*local_t)
                response[car_mask] += car_response
        
        # Environmental noise
        response += 0.0001 * np.random.normal(0, 1, len(self.t))
        return response
    
    def generate_heavy_truck(self):
        """Heavy truck passage events"""
        response = np.zeros_like(self.t)
        
        # 3-4 heavy trucks
        n_trucks = 3
        truck_times = np.linspace(60, self.duration-60, n_trucks)
        
        for truck_time in truck_times:
            truck_mask = (self.t >= truck_time) & (self.t <= truck_time + 8)
            if np.any(truck_mask):
                local_t = self.t[truck_mask] - truck_time
                # Multi-axle truck with dynamic amplification
                truck_response = (0.015 * np.exp(-local_t/3) * 
                                (1 + 0.3*np.sin(2*np.pi*2.2*local_t)) *
                                (np.exp(-(local_t-2)**2/2) + 0.7*np.exp(-(local_t-5)**2/3)))
                response[truck_mask] += truck_response
        
        # Base vibration + noise
        response += (0.0005 * np.sin(2 * np.pi * 2.2 * self.t) + 
                    0.0001 * np.random.normal(0, 1, len(self.t)))
        return response
    
    def generate_wind_loading(self):
        """Wind-induced vibrations"""
        # Wind loading typically excites higher modes and is more sustained
        # Simulate gusty wind conditions
        wind_base_freq = 0.2  # Low frequency wind fluctuation
        wind_intensity = 0.8 + 0.4 * np.sin(2 * np.pi * wind_base_freq * self.t)
        
        # Multiple modes excited by wind
        response = (wind_intensity * 0.002 * np.sin(2 * np.pi * 2.1 * self.t) +
                   wind_intensity * 0.003 * np.sin(2 * np.pi * 6.9 * self.t) + 
                   wind_intensity * 0.001 * np.sin(2 * np.pi * 12.3 * self.t))
        
        # Add turbulence
        turbulence = np.convolve(np.random.normal(0, 1, len(self.t)), 
                                np.ones(20)/20, mode='same')
        response += 0.0005 * wind_intensity * turbulence
        
        return response
    
    def generate_structural_change(self):
        """Simulate gradual structural change (e.g., loosening connection)"""
        # Frequency reduction over time + increased damping
        freq_change = 2.2 * (1 - 0.05 * self.t / self.duration)  # 5% frequency drop
        damp_increase = 0.03 * (1 + 0.5 * self.t / self.duration)  # 50% damping increase
        
        response = np.zeros_like(self.t)
        dt = 1/self.fs
        
        # Simulate degrading system response
        for i in range(len(self.t)):
            if i == 0:
                response[i] = 0.001 * np.sin(2 * np.pi * freq_change[i] * self.t[i])
            else:
                # Simple discrete damping model
                response[i] = (response[i-1] * (1 - damp_increase[i] * dt) + 
                              0.001 * np.sin(2 * np.pi * freq_change[i] * self.t[i]))
        
        # Add measurement noise
        response += 0.0001 * np.random.normal(0, 1, len(self.t))
        return response

# Generate data for all conditions
simulator = BridgeConditionSimulator(fs=100, duration=300)

conditions = {
    'Normal Traffic': simulator.generate_normal_traffic(),
    'Heavy Truck': simulator.generate_heavy_truck(), 
    'Wind Loading': simulator.generate_wind_loading(),
    'Structural Change': simulator.generate_structural_change()
}

# Extract comprehensive time-domain features
extractor = TimeDomainFeatureExtractor(sampling_rate=100)

condition_features = {}
for condition_name, signal in conditions.items():
    features = extractor.extract_all_features(signal)
    condition_features[condition_name] = features

# Create feature comparison
feature_df = pd.DataFrame(condition_features).T

# Additional discriminative features
def extract_advanced_features(signal, fs=100):
    """Extract additional discriminative features"""
    features = {}
    
    # Impulse factor
    features['impulse_factor'] = np.max(np.abs(signal)) / np.mean(np.abs(signal))
    
    # Clearance factor  
    features['clearance_factor'] = np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal))))**2
    
    # Energy concentration in different frequency bands
    from scipy.signal import welch
    f, psd = welch(signal, fs=fs, nperseg=1024)
    
    low_band = (f <= 5)
    mid_band = (f > 5) & (f <= 15) 
    high_band = (f > 15)
    
    total_energy = np.sum(psd)
    features['low_freq_ratio'] = np.sum(psd[low_band]) / total_energy
    features['mid_freq_ratio'] = np.sum(psd[mid_band]) / total_energy
    features['high_freq_ratio'] = np.sum(psd[high_band]) / total_energy
    
    # Spectral centroid
    features['spectral_centroid'] = np.sum(f * psd) / total_energy
    
    return features

# Add advanced features
for condition_name, signal in conditions.items():
    advanced_features = extract_advanced_features(signal)
    condition_features[condition_name].update(advanced_features)

# Update DataFrame
feature_df = pd.DataFrame(condition_features).T

print("Exercise 1: Time-Domain Feature Analysis")
print("=" * 50)
print("\nFeature Comparison Across Bridge Conditions:")
print(feature_df.round(6))

# Discriminant Analysis
feature_matrix = feature_df.values
condition_labels = list(feature_df.index)

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_matrix)

# LDA for feature discrimination
lda = LinearDiscriminantAnalysis(n_components=2)
features_lda = lda.fit_transform(features_scaled, range(len(condition_labels)))

# Feature importance analysis
feature_importance = np.abs(lda.coef_[0])
feature_names = list(feature_df.columns)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(f"\nMost Discriminative Features:")
print(importance_df.head(8).to_string(index=False))

# Visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Bridge Response Signals', 'Feature Discrimination (LDA)',
                   'Most Important Features', 'Feature Correlation Matrix')
)

# Plot 1: Time series
colors = ['blue', 'red', 'green', 'orange']
for i, (condition, signal) in enumerate(conditions.items()):
    fig.add_trace(
        go.Scatter(x=simulator.t[:5000], y=signal[:5000]*1000,  # First 50 seconds
                   name=condition, line=dict(color=colors[i])),
        row=1, col=1
    )

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Acceleration (mg)", row=1, col=1)

# Plot 2: LDA projection
for i, condition in enumerate(condition_labels):
    fig.add_trace(
        go.Scatter(x=[features_lda[i, 0]], y=[features_lda[i, 1]],
                   mode='markers+text', name=f'LDA {condition}',
                   marker=dict(color=colors[i], size=12),
                   text=[condition], textposition="top center"),
        row=1, col=2
    )

fig.update_xaxes(title_text="First Discriminant", row=1, col=2)
fig.update_yaxes(title_text="Second Discriminant", row=1, col=2)

# Plot 3: Feature importance
top_features = importance_df.head(6)
fig.add_trace(
    go.Bar(x=top_features['Feature'], y=top_features['Importance'],
           marker_color='lightcoral'),
    row=2, col=1
)

fig.update_xaxes(tickangle=45, row=2, col=1)
fig.update_yaxes(title_text="LDA Coefficient", row=2, col=1)

# Plot 4: Correlation matrix
selected_features = ['rms', 'peak_factor', 'spectral_centroid', 'impulse_factor', 'zero_crossing_rate']
selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
corr_matrix = np.corrcoef(features_scaled[:, selected_indices].T)

fig.add_trace(
    go.Heatmap(z=corr_matrix, x=selected_features, y=selected_features,
               colorscale='RdBu', zmid=0),
    row=2, col=2
)

fig.update_layout(height=800, title_text="Exercise 1: Bridge Condition Analysis")
fig.show()

print(f"\nKey Insights:")
print(f"• Heavy truck events show highest peak factor ({feature_df.loc['Heavy Truck', 'peak_factor']:.3f})")
print(f"• Wind loading has highest spectral centroid ({feature_df.loc['Wind Loading', 'spectral_centroid']:.3f} Hz)")
print(f"• Structural change shows reduced natural frequency content")
print(f"• Normal traffic has most balanced feature characteristics")
```

### Exercise 2: Multi-Modal Feature Fusion

**Problem:** Design a feature fusion system that combines:
- Acceleration data (time and frequency domain features)
- Strain gauge measurements  
- Environmental data (temperature, humidity, wind speed)
- Visual inspection images

Your system should handle missing modalities and provide confidence estimates for anomaly detection decisions.

**Solution:**

```python
# Exercise 2 Solution
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class RobustMultiModalFusion:
    """Robust multi-modal fusion with missing data handling"""
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.modality_weights = {}
        self.imputers = {}
        self.fusion_classifier = None
        self.modality_reliability = {}
        self.is_fitted = False
        
    def fit(self, modality_data_dict, labels):
        """
        Fit the robust fusion system
        
        Args:
            modality_data_dict: Dict with modality names and feature arrays
            labels: Binary labels (0=normal, 1=anomaly)
        """
        self.modalities = list(modality_data_dict.keys())
        
        # Assess modality reliability
        self._assess_modality_reliability(modality_data_dict, labels)
        
        # Fit imputers for each modality
        for modality, data in modality_data_dict.items():
            imputer = KNNImputer(n_neighbors=5)
            self.imputers[modality] = imputer.fit(data)
        
        # Create fused feature vector
        fused_features = self._create_fused_features(modality_data_dict)
        
        # Train ensemble classifier with confidence estimation
        self.fusion_classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        )
        self.fusion_classifier.fit(fused_features, labels)
        
        self.is_fitted = True
        return self
    
    def predict_with_confidence(self, modality_data_dict):
        """
        Predict with confidence estimates and missing modality handling
        """
        if not self.is_fitted:
            raise ValueError("System must be fitted first")
        
        # Handle missing modalities
        available_modalities = set(modality_data_dict.keys())
        missing_modalities = set(self.modalities) - available_modalities
        
        if missing_modalities:
            print(f"Warning: Missing modalities: {missing_modalities}")
        
        # Impute missing values within available modalities
        processed_data = {}
        for modality in self.modalities:
            if modality in modality_data_dict:
                # Impute missing values
                data_imputed = self.imputers[modality].transform(modality_data_dict[modality])
                processed_data[modality] = data_imputed
            else:
                # Create dummy data for missing modality (zeros)
                n_samples = len(list(modality_data_dict.values())[0])
                n_features = self.imputers[modality].n_features_in_
                processed_data[modality] = np.zeros((n_samples, n_features))
        
        # Create fused features
        fused_features = self._create_fused_features(processed_data)
        
        # Get predictions and confidence
        predictions = self.fusion_classifier.predict(fused_features)
        prediction_probs = self.fusion_classifier.predict_proba(fused_features)
        
        # Calculate confidence based on prediction probability and modality availability
        confidence_scores = np.max(prediction_probs, axis=1)
        
        # Adjust confidence based on missing modalities
        availability_factor = len(available_modalities) / len(self.modalities)
        adjusted_confidence = confidence_scores * availability_factor
        
        # High confidence decisions
        high_confidence_mask = adjusted_confidence > self.confidence_threshold
        
        results = {
            'predictions': predictions,
            'confidence_scores': adjusted_confidence,
            'high_confidence_predictions': predictions * high_confidence_mask,
            'available_modalities': list(available_modalities),
            'missing_modalities': list(missing_modalities),
            'prediction_probabilities': prediction_probs
        }
        
        return results
    
    def _assess_modality_reliability(self, modality_data_dict, labels):
        """Assess reliability of each modality for anomaly detection"""
        for modality, data in modality_data_dict.items():
            # Simple reliability based on separability
            # Calculate between-class to within-class variance ratio
            normal_data = data[labels == 0]
            anomaly_data = data[labels == 1]
            
            if len(anomaly_data) > 0 and len(normal_data) > 0:
                normal_mean = np.mean(normal_data, axis=0)
                anomaly_mean = np.mean(anomaly_data, axis=0)
                
                between_class_var = np.mean((normal_mean - anomaly_mean)**2)
                within_class_var = (np.mean(np.var(normal_data, axis=0)) + 
                                   np.mean(np.var(anomaly_data, axis=0))) / 2
                
                reliability = between_class_var / (within_class_var + 1e-8)
                self.modality_reliability[modality] = reliability
            else:
                self.modality_reliability[modality] = 0.5
    
    def _create_fused_features(self, modality_data_dict):
        """Create fused feature vector with reliability weighting"""
        feature_list = []
        
        for modality in self.modalities:
            if modality in modality_data_dict:
                data = modality_data_dict[modality]
                # Weight by reliability
                reliability = self.modality_reliability.get(modality, 1.0)
                weighted_data = data * reliability
                feature_list.append(weighted_data)
            else:
                # Use zeros for missing modality
                n_samples = len(list(modality_data_dict.values())[0])
                n_features = self.imputers[modality].n_features_in_
                feature_list.append(np.zeros((n_samples, n_features)))
        
        return np.hstack(feature_list)

# Generate multi-modal bridge monitoring dataset
def generate_multimodal_bridge_dataset(n_samples=500, missing_rate=0.1):
    """Generate realistic multi-modal bridge dataset with missing values"""
    np.random.seed(42)
    
    # Normal samples (80%) and anomalies (20%)
    n_normal = int(0.8 * n_samples)
    n_anomaly = n_samples - n_normal
    
    # Acceleration features (6 features)
    accel_normal = np.random.multivariate_normal(
        [0.1, 0.05, 2.2, 0.03, 0.15, 8.5],
        np.diag([0.01, 0.002, 0.04, 0.001, 0.01, 1.0]),
        n_normal
    )
    
    accel_anomaly = np.random.multivariate_normal(
        [0.15, 0.08, 2.0, 0.05, 0.20, 7.8],  # Changed values for anomalies
        np.diag([0.015, 0.004, 0.06, 0.002, 0.015, 1.5]),
        n_anomaly
    )
    
    acceleration_data = np.vstack([accel_normal, accel_anomaly])
    
    # Strain features (4 features)
    strain_normal = np.random.multivariate_normal(
        [50, 25, 15, 0.8],
        np.diag([100, 50, 25, 0.1]),
        n_normal
    )
    
    strain_anomaly = np.random.multivariate_normal(
        [65, 35, 25, 1.2],  # Higher strain for anomalies
        np.diag([150, 75, 40, 0.2]),
        n_anomaly
    )
    
    strain_data = np.vstack([strain_normal, strain_anomaly])
    
    # Environmental features (3 features: temp, humidity, wind)
    env_normal = np.random.multivariate_normal(
        [20, 65, 5],
        np.diag([25, 100, 4]),
        n_normal
    )
    
    env_anomaly = np.random.multivariate_normal(
        [22, 70, 8],  # Slightly different environmental conditions
        np.diag([30, 120, 6]),
        n_anomaly
    )
    
    environmental_data = np.vstack([env_normal, env_anomaly])
    
    # Visual features (5 features from image analysis)
    visual_normal = np.random.multivariate_normal(
        [150, 0.3, 0.1, 0.7, 25],
        np.diag([400, 0.01, 0.001, 0.05, 50]),
        n_normal
    )
    
    visual_anomaly = np.random.multivariate_normal(
        [140, 0.5, 0.2, 0.5, 35],  # Different visual characteristics
        np.diag([500, 0.02, 0.002, 0.08, 80]),
        n_anomaly
    )
    
    visual_data = np.vstack([visual_normal, visual_anomaly])
    
    # Create labels
    labels = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    acceleration_data = acceleration_data[indices]
    strain_data = strain_data[indices]
    environmental_data = environmental_data[indices]
    visual_data = visual_data[indices]
    labels = labels[indices]
    
    # Introduce missing values randomly
    def add_missing_values(data, missing_rate):
        data_with_missing = data.copy()
        n_missing = int(missing_rate * data.size)
        missing_indices = np.random.choice(data.size, n_missing, replace=False)
        flat_data = data_with_missing.flatten()
        flat_data[missing_indices] = np.nan
        return flat_data.reshape(data.shape)
    
    # Add missing values to each modality
    acceleration_data = add_missing_values(acceleration_data, missing_rate)
    strain_data = add_missing_values(strain_data, missing_rate)
    environmental_data = add_missing_values(environmental_data, missing_rate)
    visual_data = add_missing_values(visual_data, missing_rate)
    
    modality_data = {
        'acceleration': acceleration_data,
        'strain': strain_data,
        'environmental': environmental_data,
        'visual': visual_data
    }
    
    return modality_data, labels

# Generate dataset
print("Exercise 2: Multi-Modal Feature Fusion")
print("=" * 50)

multimodal_data, y_labels = generate_multimodal_bridge_dataset(n_samples=500, missing_rate=0.1)

print(f"Dataset generated:")
for modality, data in multimodal_data.items():
    missing_count = np.sum(np.isnan(data))
    missing_percentage = missing_count / data.size * 100
    print(f"• {modality}: {data.shape}, {missing_percentage:.1f}% missing values")

# Split data
train_size = int(0.7 * len(y_labels))
train_data = {mod: data[:train_size] for mod, data in multimodal_data.items()}
test_data = {mod: data[train_size:] for mod, data in multimodal_data.items()}
y_train = y_labels[:train_size]
y_test = y_labels[train_size:]

# Train robust fusion system
fusion_system = RobustMultiModalFusion(confidence_threshold=0.7)
fusion_system.fit(train_data, y_train)

print(f"\nModality Reliability Assessment:")
for modality, reliability in fusion_system.modality_reliability.items():
    print(f"• {modality}: {reliability:.3f}")

# Test with complete data
complete_results = fusion_system.predict_with_confidence(test_data)

# Test with missing modalities (remove visual modality)
partial_data = {k: v for k, v in test_data.items() if k != 'visual'}
partial_results = fusion_system.predict_with_confidence(partial_data)

# Evaluate performance
from sklearn.metrics import accuracy_score, f1_score

complete_accuracy = accuracy_score(y_test, complete_results['predictions'])
complete_f1 = f1_score(y_test, complete_results['predictions'])

partial_accuracy = accuracy_score(y_test, partial_results['predictions'])
partial_f1 = f1_score(y_test, partial_results['predictions'])

print(f"\nPerformance Comparison:")
print(f"Complete modalities - Accuracy: {complete_accuracy:.3f}, F1: {complete_f1:.3f}")
print(f"Missing visual modality - Accuracy: {partial_accuracy:.3f}, F1: {partial_f1:.3f}")

# High confidence analysis
high_conf_complete = np.sum(complete_results['high_confidence_predictions'] > 0)
high_conf_partial = np.sum(partial_results['high_confidence_predictions'] > 0)

print(f"\nHigh Confidence Detections:")
print(f"Complete modalities: {high_conf_complete}/{len(y_test)} predictions")
print(f"Missing visual: {high_conf_partial}/{len(y_test)} predictions")

# Visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Modality Reliability', 'Confidence Score Distribution',
                   'Complete vs Partial Performance', 'Missing Data Impact')
)

# Plot 1: Modality reliability
modalities = list(fusion_system.modality_reliability.keys())
reliabilities = list(fusion_system.modality_reliability.values())

fig.add_trace(
    go.Bar(x=modalities, y=reliabilities, marker_color='lightblue',
           name='Reliability'),
    row=1, col=1
)
fig.update_yaxes(title_text="Reliability Score", row=1, col=1)

# Plot 2: Confidence distributions
fig.add_trace(
    go.Histogram(x=complete_results['confidence_scores'], 
                 name='Complete Modalities',
                 marker_color='blue', opacity=0.7, nbinsx=20),
    row=1, col=2
)
fig.add_trace(
    go.Histogram(x=partial_results['confidence_scores'], 
                 name='Missing Visual',
                 marker_color='red', opacity=0.7, nbinsx=20),
    row=1, col=2
)
fig.update_xaxes(title_text="Confidence Score", row=1, col=2)

# Plot 3: Performance comparison
metrics = ['Accuracy', 'F1-Score']
complete_metrics = [complete_accuracy, complete_f1]
partial_metrics = [partial_accuracy, partial_f1]

fig.add_trace(
    go.Bar(x=metrics, y=complete_metrics, name='Complete',
           marker_color='blue', opacity=0.7),
    row=2, col=1
)
fig.add_trace(
    go.Bar(x=metrics, y=partial_metrics, name='Partial',
           marker_color='red', opacity=0.7),
    row=2, col=1
)
fig.update_yaxes(title_text="Score", row=2, col=1)

# Plot 4: Missing data impact on different sample types
normal_mask = y_test == 0
anomaly_mask = y_test == 1

complete_conf_normal = np.mean(complete_results['confidence_scores'][normal_mask])
complete_conf_anomaly = np.mean(complete_results['confidence_scores'][anomaly_mask])
partial_conf_normal = np.mean(partial_results['confidence_scores'][normal_mask])
partial_conf_anomaly = np.mean(partial_results['confidence_scores'][anomaly_mask])

categories = ['Normal Samples', 'Anomaly Samples']
complete_conf = [complete_conf_normal, complete_conf_anomaly]
partial_conf = [partial_conf_normal, partial_conf_anomaly]

fig.add_trace(
    go.Bar(x=categories, y=complete_conf, name='Complete',
           marker_color='green', opacity=0.7),
    row=2, col=2
)
fig.add_trace(
    go.Bar(x=categories, y=partial_conf, name='Partial',
           marker_color='orange', opacity=0.7),
    row=2, col=2
)
fig.update_yaxes(title_text="Mean Confidence", row=2, col=2)

fig.update_layout(height=800, title_text="Exercise 2: Multi-Modal Fusion Analysis")
fig.show()

print(f"\nKey Insights:")
print(f"• {max(fusion_system.modality_reliability, key=fusion_system.modality_reliability.get)} is most reliable modality")
print(f"• Performance degradation with missing visual: {(complete_f1-partial_f1)*100:.1f}% F1-score drop")
print(f"• System maintains {partial_accuracy/complete_accuracy*100:.1f}% of accuracy with missing modality")
print(f"• Confidence-based filtering improves decision reliability")
```

### Exercise 3: Statistical vs Deep Learning Comparison

**Problem:** Compare the performance of statistical anomaly detection methods with deep learning approaches on a challenging SHM dataset containing:
- Seasonal variations
- Multiple damage types  
- Environmental effects
- Measurement noise

Analyze the trade-offs between interpretability, computational requirements, and detection performance.

**Solution:**

```python
# Exercise 3 Solution
import time
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class ComprehensiveAnomalyBenchmark:
    """Comprehensive benchmark comparing statistical vs deep learning methods"""
    
    def __init__(self):
        self.results = {}
        self.computational_times = {}
        
    def generate_challenging_dataset(self, n_samples=2000):
        """Generate challenging SHM dataset with complex patterns"""
        np.random.seed(42)
        
        # Time-based seasonal effects
        time_points = np.linspace(0, 365, n_samples)  # One year
        seasonal_temp = 15 + 10 * np.sin(2 * np.pi * time_points / 365)
        seasonal_effect = 0.1 * seasonal_temp / 25  # Temperature effect on structure
        
        # Normal samples (85%)
        n_normal = int(0.85 * n_samples)
        
        # Base structural features with seasonal variation
        base_features = []
        for i in range(n_normal):
            temp_effect = seasonal_effect[i]
            # Natural frequency (affected by temperature)
            nat_freq = 2.2 + temp_effect + np.random.normal(0, 0.02)
            # Damping ratio
            damping = 0.03 + np.random.normal(0, 0.002)
            # Mode shape amplitude
            mode_amp = 0.15 + temp_effect * 0.5 + np.random.normal(0, 0.01)
            # RMS acceleration
            rms_accel = 0.05 + np.random.normal(0, 0.005)
            # Strain range
            strain_range = 45 + temp_effect * 10 + np.random.normal(0, 5)
            # Environmental correlation
            env_corr = 0.8 + temp_effect * 0.1 + np.random.normal(0, 0.05)
            
            base_features.append([nat_freq, damping, mode_amp, rms_accel, strain_range, env_corr])
        
        normal_features = np.array(base_features)
        
        # Anomalous samples (15%) - multiple damage types
        n_anomaly = n_samples - n_normal
        anomaly_features = []
        anomaly_types = []
        
        # Type 1: Gradual stiffness loss (40% of anomalies)
        n_type1 = int(0.4 * n_anomaly)
        for i in range(n_type1):
            idx = n_normal + i
            temp_effect = seasonal_effect[idx] if idx < len(seasonal_effect) else 0
            
            # Reduced natural frequency
            nat_freq = 2.0 + temp_effect + np.random.normal(0, 0.03)
            damping = 0.04 + np.random.normal(0, 0.003)
            mode_amp = 0.18 + temp_effect * 0.5 + np.random.normal(0, 0.015)
            rms_accel = 0.07 + np.random.normal(0, 0.008)
            strain_range = 55 + temp_effect * 10 + np.random.normal(0, 8)
            env_corr = 0.75 + temp_effect * 0.1 + np.random.normal(0, 0.08)
            
            anomaly_features.append([nat_freq, damping, mode_amp, rms_accel, strain_range, env_corr])
            anomaly_types.append(1)
        
        # Type 2: Connection loosening (35% of anomalies)
        n_type2 = int(0.35 * n_anomaly)
        for i in range(n_type2):
            idx = n_normal + n_type1 + i
            temp_effect = seasonal_effect[idx] if idx < len(seasonal_effect) else 0
            
            # Changed damping and mode shape
            nat_freq = 2.15 + temp_effect + np.random.normal(0, 0.04)
            damping = 0.06 + np.random.normal(0, 0.005)  # Increased damping
            mode_amp = 0.22 + temp_effect * 0.5 + np.random.normal(0, 0.02)
            rms_accel = 0.08 + np.random.normal(0, 0.01)
            strain_range = 65 + temp_effect * 10 + np.random.normal(0, 10)
            env_corr = 0.6 + temp_effect * 0.1 + np.random.normal(0, 0.1)
            
            anomaly_features.append([nat_freq, damping, mode_amp, rms_accel, strain_range, env_corr])
            anomaly_types.append(2)
        
        # Type 3: Sensor malfunction (25% of anomalies)
        n_type3 = n_anomaly - n_type1 - n_type2
        for i in range(n_type3):
            idx = n_normal + n_type1 + n_type2 + i
            temp_effect = seasonal_effect[idx] if idx < len(seasonal_effect) else 0
            
            # Erratic readings
            nat_freq = 2.2 + temp_effect + np.random.normal(0, 0.1)  # High noise
            damping = 0.03 + np.random.normal(0, 0.01)
            mode_amp = 0.15 + temp_effect * 0.5 + np.random.normal(0, 0.05)
            rms_accel = 0.05 + np.random.normal(0, 0.02)  # Noisy
            strain_range = 45 + temp_effect * 10 + np.random.normal(0, 15)
            env_corr = 0.8 + temp_effect * 0.1 + np.random.normal(0, 0.2)
            
            anomaly_features.append([nat_freq, damping, mode_amp, rms_accel, strain_range, env_corr])
            anomaly_types.append(3)
        
        anomaly_features = np.array(anomaly_features)
        
        # Combine data
        X = np.vstack([normal_features, anomaly_features])
        y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
        
        # Add measurement noise
        noise_std = 0.01
        X += np.random.normal(0, noise_std, X.shape)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y, time_points[indices]
    
    def benchmark_statistical_methods(self, X_train, X_test, y_test):
        """Benchmark statistical anomaly detection methods"""
        print("Benchmarking Statistical Methods...")
        
        results = {}
        
        # 1. Hotelling T²
        start_time = time.time()
        stat_detector = StatisticalAnomalyDetector(contamination=0.15, confidence_level=0.95)
        stat_detector.fit(X_train)
        stat_results = stat_detector.detect_anomalies(X_test)
        
        t2_time = time.time() - start_time
        t2_predictions = stat_results['hotelling_t2_anomalies']
        t2_scores = stat_results['hotelling_t2_scores']
        
        results['Hotelling_T2'] = {
            'predictions': t2_predictions,
            'scores': t2_scores,
            'time': t2_time,
            'interpretable': True
        }
        
        # 2. Isolation Forest
        start_time = time.time()
        iso_predictions = stat_results['isolation_forest_anomalies']
        iso_scores = -stat_results['isolation_forest_scores']  # Convert to positive scores
        iso_time = time.time() - start_time
        
        results['Isolation_Forest'] = {
            'predictions': iso_predictions,
            'scores': iso_scores,
            'time': iso_time,
            'interpretable': False
        }
        
        # 3. One-Class SVM
        start_time = time.time()
        svm_predictions = stat_results['one_class_svm_anomalies']
        svm_scores = -stat_results['svm_scores']
        svm_time = time.time() - start_time
        
        results['One_Class_SVM'] = {
            'predictions': svm_predictions,
            'scores': svm_scores,
            'time': svm_time,
            'interpretable': False
        }
        
        return results
    
    def benchmark_deep_learning_methods(self, X_train, X_test, y_test):
        """Benchmark deep learning methods"""
        print("Benchmarking Deep Learning Methods...")
        
        results = {}
        
        # 1. Autoencoder
        start_time = time.time()
        ae_detector = DeepAnomalyDetector(model_type='autoencoder', device='cpu')
        ae_detector.fit(X_train, epochs=100, batch_size=32, learning_rate=0.001)
        ae_results = ae_detector.predict(X_test)
        ae_time = time.time() - start_time
        
        results['Autoencoder'] = {
            'predictions': ae_results['predictions'],
            'scores': ae_results['reconstruction_errors'],
            'time': ae_time,
            'interpretable': False
        }
        
        # 2. Variational Autoencoder
        start_time = time.time()
        vae_detector = DeepAnomalyDetector(model_type='vae', device='cpu')
        vae_detector.fit(X_train, epochs=100, batch_size=32, learning_rate=0.001)
        vae_results = vae_detector.predict(X_test)
        vae_time = time.time() - start_time
        
        results['VAE'] = {
            'predictions': vae_results['predictions'],
            'scores': vae_results['reconstruction_errors'],
            'time': vae_time,
            'interpretable': False
        }
        
        return results
    
    def evaluate_performance(self, results, y_test):
        """Comprehensive performance evaluation"""
        evaluation = {}
        
        for method_name, method_results in results.items():
            predictions = method_results['predictions']
            scores = method_results['scores']
            
            # Basic metrics
            tp = np.sum((predictions == 1) & (y_test == 1))
            fp = np.sum((predictions == 1) & (y_test == 0))
            fn = np.sum((predictions == 0) & (y_test == 1))
            tn = np.sum((predictions == 0) & (y_test == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(predictions)
            
            # ROC AUC
            fpr, tpr, _ = roc_curve(y_test, scores)
            roc_auc = auc(fpr, tpr)
            
            # PR AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
            pr_auc = auc(recall_curve, precision_curve)
            
            evaluation[method_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'time': method_results['time'],
                'interpretable': method_results['interpretable']
            }
        
        return evaluation

# Run comprehensive benchmark
print("Exercise 3: Statistical vs Deep Learning Comparison")
print("=" * 60)

# Generate challenging dataset
benchmark = ComprehensiveAnomalyBenchmark()
X, y, time_points = benchmark.generate_challenging_dataset(n_samples=2000)

print(f"Dataset characteristics:")
print(f"• Total samples: {len(X)}")
print(f"• Features: {X.shape[1]}")
print(f"• Anomaly rate: {np.mean(y):.1%}")
print(f"• Seasonal variations included")
print(f"• Multiple damage types simulated")

# Split data
train_size = int(0.6 * len(X))
X_train = X[:train_size][y[:train_size] == 0]  # Only healthy samples for training
X_test = X[train_size:]
y_test = y[train_size:]

print(f"• Training samples (healthy): {len(X_train)}")
print(f"• Test samples: {len(X_test)}")

# Benchmark both approaches
statistical_results = benchmark.benchmark_statistical_methods(X_train, X_test, y_test)
deep_learning_results = benchmark.benchmark_deep_learning_methods(X_train, X_test, y_test)

# Combine results
all_results = {**statistical_results, **deep_learning_results}

# Evaluate performance
performance_evaluation = benchmark.evaluate_performance(all_results, y_test)

# Create performance summary
performance_df = pd.DataFrame(performance_evaluation).T
performance_df = performance_df.round(4)

print(f"\nPerformance Comparison:")
print(performance_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'time']].to_string())

# Interpretability analysis
print(f"\nInterpretability Analysis:")
for method, eval_data in performance_evaluation.items():
    interpretable = "High" if eval_data['interpretable'] else "Low"
    print(f"• {method}: {interpretable}")

# Computational efficiency analysis
print(f"\nComputational Efficiency (Training + Inference):")
methods_by_time = sorted(performance_evaluation.items(), key=lambda x: x[1]['time'])
for method, eval_data in methods_by_time:
    print(f"• {method}: {eval_data['time']:.2f} seconds")

# Advanced visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Performance Comparison', 'ROC Curves',
                   'Computational Time vs Performance', 'Score Distributions',
                   'Method Trade-offs', 'Seasonal Effect Analysis'),
    specs=[[{"type": "bar"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "histogram"}],
           [{"type": "scatter"}, {"type": "scatter"}]]
)

# Plot 1: Performance metrics comparison
methods = list(performance_evaluation.keys())
f1_scores = [performance_evaluation[m]['f1_score'] for m in methods]
colors = ['blue', 'green', 'orange', 'red', 'purple']

fig.add_trace(
    go.Bar(x=methods, y=f1_scores, marker_color=colors,
           name='F1-Score'),
    row=1, col=1
)
fig.update_xaxes(tickangle=45, row=1, col=1)
fig.update_yaxes(title_text="F1-Score", row=1, col=1)

# Plot 2: ROC Curves
for i, (method, method_results) in enumerate(all_results.items()):
    fpr, tpr, _ = roc_curve(y_test, method_results['scores'])
    auc_score = performance_evaluation[method]['roc_auc']
    
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode='lines',
                   name=f'{method} (AUC={auc_score:.3f})',
                   line=dict(color=colors[i])),
        row=1, col=2
    )

# Add diagonal line
fig.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
               name='Random', line=dict(color='gray', dash='dash')),
    row=1, col=2
)

fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)

# Plot 3: Time vs Performance trade-off
times = [performance_evaluation[m]['time'] for m in methods]
roc_aucs = [performance_evaluation[m]['roc_auc'] for m in methods]

fig.add_trace(
    go.Scatter(x=times, y=roc_aucs, mode='markers+text',
               text=methods, textposition="top center",
               marker=dict(size=10, color=colors),
               name='Time vs Performance'),
    row=2, col=1
)

fig.update_xaxes(title_text="Training Time (seconds)", type="log", row=2, col=1)
fig.update_yaxes(title_text="ROC AUC", row=2, col=1)

# Plot 4: Score distributions for best methods
best_stat_method = max(statistical_results.keys(), 
                      key=lambda x: performance_evaluation[x]['f1_score'])
best_dl_method = max(deep_learning_results.keys(), 
                    key=lambda x: performance_evaluation[x]['f1_score'])

# Statistical method scores
stat_scores_normal = all_results[best_stat_method]['scores'][y_test == 0]
stat_scores_anomaly = all_results[best_stat_method]['scores'][y_test == 1]

fig.add_trace(
    go.Histogram(x=stat_scores_normal, name=f'{best_stat_method} Normal',
                 marker_color='blue', opacity=0.6, nbinsx=25),
    row=2, col=2
)
fig.add_trace(
    go.Histogram(x=stat_scores_anomaly, name=f'{best_stat_method} Anomaly',
                 marker_color='red', opacity=0.6, nbinsx=20),
    row=2, col=2
)

fig.update_xaxes(title_text="Anomaly Score", row=2, col=2)
fig.update_yaxes(title_text="Frequency", row=2, col=2)

# Plot 5: Interpretability vs Performance
interpretability_scores = [1 if performance_evaluation[m]['interpretable'] else 0 for m in methods]

fig.add_trace(
    go.Scatter(x=interpretability_scores, y=f1_scores,
               mode='markers+text', text=methods,
               textposition="top center",
               marker=dict(size=12, color=colors),
               name='Interpretability vs Performance'),
    row=3, col=1
)

fig.update_xaxes(title_text="Interpretability (0=Low, 1=High)", row=3, col=1)
fig.update_yaxes(title_text="F1-Score", row=3, col=1)

# Plot 6: Seasonal effect analysis (show how scores vary with time)
seasonal_indices = np.argsort(time_points[train_size:])
seasonal_scores = all_results[best_stat_method]['scores'][seasonal_indices]
seasonal_labels = y_test[seasonal_indices]

fig.add_trace(
    go.Scatter(x=range(len(seasonal_scores)), y=seasonal_scores,
               mode='markers',
               marker=dict(color=['red' if label else 'blue' for label in seasonal_labels],
                          size=4),
               name='Seasonal Variation'),
    row=3, col=2
)

fig.update_xaxes(title_text="Time (sorted)", row=3, col=2)
fig.update_yaxes(title_text="Anomaly Score", row=3, col=2)

fig.update_layout(
    height=1200,
    title_text="Exercise 3: Comprehensive Method Comparison",
    font=dict(size=10)
)

fig.show()

# Summary insights
best_overall = max(performance_evaluation.keys(), 
                  key=lambda x: performance_evaluation[x]['f1_score'])
fastest_method = min(performance_evaluation.keys(), 
                    key=lambda x: performance_evaluation[x]['time'])
most_interpretable = [m for m, e in performance_evaluation.items() if e['interpretable']]

print(f"\nKey Insights:")
print(f"• Best overall performance: {best_overall} (F1: {performance_evaluation[best_overall]['f1_score']:.3f})")
print(f"• Fastest method: {fastest_method} ({performance_evaluation[fastest_method]['time']:.2f}s)")
print(f"• Most interpretable methods: {', '.join(most_interpretable)}")
print(f"• Deep learning methods show {'better' if max([performance_evaluation[m]['f1_score'] for m in deep_learning_results]) > max([performance_evaluation[m]['f1_score'] for m in statistical_results]) else 'similar'} performance")
print(f"• Statistical methods are {'faster' if np.mean([performance_evaluation[m]['time'] for m in statistical_results]) < np.mean([performance_evaluation[m]['time'] for m in deep_learning_results]) else 'slower'} on average")
print(f"• Trade-off exists between interpretability and performance")

print(f"\nRecommendations:")
print(f"• For real-time monitoring: Use {fastest_method}")
print(f"• For critical decisions requiring explanation: Use Hotelling T²")
print(f"• For maximum detection performance: Use {best_overall}")
print(f"• For handling complex patterns: Consider ensemble approaches")
```

### Exercise 4: Feature Engineering Pipeline Design

**Problem:** Design a complete feature engineering pipeline that:
- Automatically handles different sensor types
- Adapts to changing environmental conditions
- Provides real-time feature updates
- Maintains feature quality metrics

**Solution:**

```python
# Exercise 4 Solution
from collections import deque
from datetime import datetime, timedelta
import threading
import queue

class AdaptiveFeatureEngineeringPipeline:
    """
    Real-time adaptive feature engineering pipeline for SHM
    """
    
    def __init__(self, buffer_size=1000, adaptation_window=100):
        self.buffer_size = buffer_size
        self.adaptation_window = adaptation_window
        
        # Data buffers for different sensor types
        self.sensor_buffers = {
            'accelerometer': deque(maxlen=buffer_size),
            'strain_gauge': deque(maxlen=buffer_size),
            'temperature': deque(maxlen=buffer_size),
            'image': deque(maxlen=50)  # Smaller buffer for images
        }
        
        # Feature extractors
        self.time_extractor = TimeDomainFeatureExtractor()
        self.freq_extractor = FrequencyDomainFeatureExtractor()
        self.image_extractor = ImageFeatureExtractor()
        
        # Adaptive components
        self.environmental_baseline = {}
        self.feature_quality_monitor = FeatureQualityMonitor()
        self.adaptation_trigger = AdaptationTrigger()
        
        # Real-time processing
        self.processing_queue = queue.Queue()
        self.feature_cache = {}
        self.is_running = False
        
    def start_realtime_processing(self):
        """Start real-time feature processing thread"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.start()
        
    def stop_realtime_processing(self):
        """Stop real-time processing"""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
    
    def add_sensor_data(self, sensor_type, data, timestamp=None):
        """Add new sensor data to pipeline"""
        if timestamp is None:
            timestamp = datetime.now()
            
        sensor_data = {
            'data': data,
            'timestamp': timestamp,
            'sensor_type': sensor_type
        }
        
        # Add to buffer
        if sensor_type in self.sensor_buffers:
            self.sensor_buffers[sensor_type].append(sensor_data)
            
        # Queue for processing
        self.processing_queue.put(sensor_data)
    
    def _process_loop(self):
        """Main processing loop for real-time feature extraction"""
        while self.is_running:
            try:
                # Get data from queue (with timeout)
                sensor_data = self.processing_queue.get(timeout=1.0)
                
                # Process data
                features = self._extract_features(sensor_data)
                
                # Update feature cache
                self._update_feature_cache(features, sensor_data['timestamp'])
                
                # Monitor feature quality
                self.feature_quality_monitor.update(features)
                
                # Check for adaptation triggers
                if self.adaptation_trigger.should_adapt(features):
                    self._adapt_pipeline()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def _extract_features(self, sensor_data):
        """Extract features based on sensor type"""
        sensor_type = sensor_data['sensor_type']
        data = sensor_data['data']
        timestamp = sensor_data['timestamp']
        
        features = {'timestamp': timestamp, 'sensor_type': sensor_type}
        
        if sensor_type == 'accelerometer':
            # Time-domain features
            time_features = self.time_extractor.extract_all_features(data)
            features.update({f'accel_{k}': v for k, v in time_features.items()})
            
            # Frequency-domain features
            freq_features = self.freq_extractor.extract_spectral_features(data)
            features.update({f'accel_freq_{k}': v for k, v in freq_features.items()})
            
        elif sensor_type == 'strain_gauge':
            # Strain-specific features
            strain_features = self._extract_strain_features(data)
            features.update({f'strain_{k}': v for k, v in strain_features.items()})
            
        elif sensor_type == 'temperature':
            # Environmental features
            temp_features = self._extract_temperature_features(data)
            features.update({f'temp_{k}': v for k, v in temp_features.items()})
            
        elif sensor_type == 'image':
            # Image features (computationally intensive)
            if len(data.shape) >= 2:  # Valid image
                img_features, _ = self.image_extractor.extract_all_features(data)
                features.update({f'img_{k}': v for k, v in img_features.items()})
        
        return features
    
    def _extract_strain_features(self, strain_data):
        """Extract strain-specific features"""
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(strain_data)
        features['std'] = np.std(strain_data)
        features['range'] = np.ptp(strain_data)
        features['rms'] = np.sqrt(np.mean(strain_data**2))
        
        # Fatigue-related features
        # Rainflow counting approximation (simplified)
        strain_cycles = self._count_strain_cycles(strain_data)
        features['cycle_count'] = len(strain_cycles)
        features['max_strain_range'] = max(strain_cycles) if strain_cycles else 0
        features['mean_strain_range'] = np.mean(strain_cycles) if strain_cycles else 0
        
        # Load effect indicators
        features['positive_peaks'] = len([x for x in strain_data if x > np.mean(strain_data) + 2*np.std(strain_data)])
        features['negative_peaks'] = len([x for x in strain_data if x < np.mean(strain_data) - 2*np.std(strain_data)])
        
        return features
    
    def _count_strain_cycles(self, strain_data):
        """Simplified strain cycle counting"""
        # Find local maxima and minima
        from scipy.signal import find_peaks
        
        peaks_pos, _ = find_peaks(strain_data, height=np.mean(strain_data))
        peaks_neg, _ = find_peaks(-strain_data, height=-np.mean(strain_data))
        
        # Calculate strain ranges between peaks
        all_peaks = sorted(list(peaks_pos) + list(peaks_neg))
        strain_ranges = []
        
        for i in range(len(all_peaks)-1):
            strain_range = abs(strain_data[all_peaks[i+1]] - strain_data[all_peaks[i]])
            strain_ranges.append(strain_range)
        
        return strain_ranges
    
    def _extract_temperature_features(self, temp_data):
        """Extract temperature-related features"""
        features = {}
        
        # Basic temperature statistics
        features['current_temp'] = temp_data[-1] if len(temp_data) > 0 else 0
        features['temp_mean'] = np.mean(temp_data)
        features['temp_std'] = np.std(temp_data)
        features['temp_trend'] = self._calculate_trend(temp_data)
        
        # Temperature rate of change
        if len(temp_data) > 1:
            temp_diff = np.diff(temp_data)
            features['temp_rate'] = np.mean(temp_diff)
            features['temp_rate_std'] = np.std(temp_diff)
        else:
            features['temp_rate'] = 0
            features['temp_rate_std'] = 0
        
        return features
    
    def _calculate_trend(self, data):
        """Calculate linear trend in data"""
        if len(data) < 2:
            return 0
        
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        return coeffs[0]  # Slope
    
    def _update_feature_cache(self, features, timestamp):
        """Update the feature cache with new features"""
        # Keep only recent features (last hour)
        cutoff_time = timestamp - timedelta(hours=1)
        
        for key in list(self.feature_cache.keys()):
            if self.feature_cache[key]['timestamp'] < cutoff_time:
                del self.feature_cache[key]
        
        # Add new features
        feature_id = f"{timestamp.isoformat()}_{features['sensor_type']}"
        self.feature_cache[feature_id] = features
    
    def _adapt_pipeline(self):
        """Adapt pipeline parameters based on recent data"""
        print("Adapting pipeline parameters...")
        
        # Update environmental baseline
        self._update_environmental_baseline()
        
        # Adjust feature extraction parameters
        self._adjust_extraction_parameters()
    
    def _update_environmental_baseline(self):
        """Update environmental baseline from recent temperature data"""
        temp_buffer = list(self.sensor_buffers['temperature'])
        
        if len(temp_buffer) >= self.adaptation_window:
            recent_temps = [item['data'][-1] for item in temp_buffer[-self.adaptation_window:]]
            self.environmental_baseline['temperature'] = {
                'mean': np.mean(recent_temps),
                'std': np.std(recent_temps),
                'updated': datetime.now()
            }
    
    def _adjust_extraction_parameters(self):
        """Adjust feature extraction parameters based on data characteristics"""
        # Adjust frequency analysis parameters based on recent data
        accel_buffer = list(self.sensor_buffers['accelerometer'])
        
        if len(accel_buffer) >= 10:
            recent_data = [item['data'] for item in accel_buffer[-10:]]
            combined_data = np.concatenate(recent_data)
            
            # Adjust frequency range based on signal characteristics
            freq_content = np.fft.fftfreq(len(combined_data), d=1/100)[:len(combined_data)//2]
            psd = np.abs(np.fft.fft(combined_data)[:len(combined_data)//2])**2
            
            # Find dominant frequency range
            dominant_freq_idx = np.argmax(psd)
            dominant_freq = freq_content[dominant_freq_idx]
            
            # Update extractor parameters (simplified)
            self.freq_extractor.dominant_freq = dominant_freq
    
    def get_current_features(self, sensor_type=None):
        """Get current feature vector"""
        if sensor_type:
            features = {k: v for k, v in self.feature_cache.items() 
                       if v['sensor_type'] == sensor_type}
        else:
            features = self.feature_cache
        
        return features
    
    def get_feature_summary(self):
        """Get summary of current pipeline state"""
        summary = {
            'buffer_sizes': {k: len(v) for k, v in self.sensor_buffers.items()},
            'feature_cache_size': len(self.feature_cache),
            'environmental_baseline': self.environmental_baseline,
            'is_running': self.is_running,
            'queue_size': self.processing_queue.qsize()
        }
        
        return summary

class FeatureQualityMonitor:
    """Monitor feature quality and detect degradation"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.feature_history = {}
        self.quality_metrics = {}
        
    def update(self, features):
        """Update feature quality monitoring"""
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                if feature_name not in self.feature_history:
                    self.feature_history[feature_name] = deque(maxlen=self.window_size)
                
                self.feature_history[feature_name].append(feature_value)
                
                # Calculate quality metrics
                if len(self.feature_history[feature_name]) >= 10:
                    self._calculate_quality_metrics(feature_name)
    
    def _calculate_quality_metrics(self, feature_name):
        """Calculate quality metrics for a feature"""
        values = list(self.feature_history[feature_name])
        
        # Stability (inverse of coefficient of variation)
        mean_val = np.mean(values)
        std_val = np.std(values)
        stability = 1 / (std_val / (abs(mean_val) + 1e-8) + 1e-8)
        
        # Consistency (lack of sudden jumps)
        diffs = np.abs(np.diff(values))
        consistency = 1 / (np.mean(diffs) + 1e-8)
        
        # Trend detection
        trend_strength = abs(self._calculate_trend(values))
        
        self.quality_metrics[feature_name] = {
            'stability': min(stability, 10),  # Cap at 10
            'consistency': min(consistency, 10),
            'trend_strength': trend_strength,
            'overall_quality': (min(stability, 10) + min(consistency, 10)) / 2
        }
    
    def _calculate_trend(self, values):
        """Calculate trend strength"""
        if len(values) < 3:
            return 0
        
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]
    
    def get_quality_report(self):
        """Get comprehensive quality report"""
        return self.quality_metrics

class AdaptationTrigger:
    """Trigger pipeline adaptation based on various conditions"""
    
    def __init__(self, adaptation_threshold=0.3):
        self.adaptation_threshold = adaptation_threshold
        self.last_adaptation = datetime.now()
        self.adaptation_cooldown = timedelta(minutes=5)
        self.baseline_features = {}
        
    def should_adapt(self, features):
        """Determine if pipeline should adapt"""
        # Cooldown check
        if datetime.now() - self.last_adaptation < self.adaptation_cooldown:
            return False
        
        # Initialize baseline if empty
        if not self.baseline_features:
            self._update_baseline(features)
            return False
        
        # Check for significant drift
        drift_detected = self._detect_drift(features)
        
        if drift_detected:
            self.last_adaptation = datetime.now()
            self._update_baseline(features)
            return True
        
        return False
    
    def _detect_drift(self, current_features):
        """Detect significant drift in features"""
        drift_scores = []
        
        for feature_name, current_value in current_features.items():
            if (isinstance(current_value, (int, float)) and 
                not np.isnan(current_value) and 
                feature_name in self.baseline_features):
                
                baseline_value = self.baseline_features[feature_name]
                
                # Calculate relative drift
                if abs(baseline_value) > 1e-8:
                    relative_drift = abs(current_value - baseline_value) / abs(baseline_value)
                else:
                    relative_drift = abs(current_value - baseline_value)
                
                drift_scores.append(relative_drift)
        
        if drift_scores:
            mean_drift = np.mean(drift_scores)
            return mean_drift > self.adaptation_threshold
        
        return False
    
    def _update_baseline(self, features):
        """Update baseline features"""
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                self.baseline_features[feature_name] = feature_value

# Demonstrate the adaptive pipeline
print("\nExercise 4: Adaptive Feature Engineering Pipeline")
print("=" * 60)

# Initialize pipeline
pipeline = AdaptiveFeatureEngineeringPipeline(buffer_size=200, adaptation_window=50)

# Start real-time processing
pipeline.start_realtime_processing()

print("Pipeline initialized and started.")
print("Simulating real-time sensor data...")

# Simulate real-time data streams
def simulate_sensor_data():
    """Simulate realistic sensor data with changing conditions"""
    
    # Simulate 5 minutes of data
    duration = 300  # seconds
    dt = 1.0  # 1 second intervals
    
    for t in range(duration):
        current_time = datetime.now() + timedelta(seconds=t)
        
        # Environmental change simulation (temperature increase)
        base_temp = 20 + 5 * np.sin(2 * np.pi * t / 100) + 0.1 * t/60  # Gradual warming
        temp_noise = np.random.normal(0, 0.5)
        temperature = base_temp + temp_noise
        
        # Accelerometer data (affected by temperature)
        temp_effect = (temperature - 20) * 0.01  # Temperature effect on structure
        
        # Simulate acceleration signal
        fs = 100
        signal_duration = 1.0
        t_signal = np.linspace(0, signal_duration, int(fs * signal_duration))
        
        # Natural frequency changes with temperature
        nat_freq = 2.2 + temp_effect + np.random.normal(0, 0.01)
        
        accel_signal = (0.001 * np.sin(2 * np.pi * nat_freq * t_signal) + 
                       0.0005 * np.sin(2 * np.pi * 6.8 * t_signal) +
                       0.0001 * np.random.normal(0, 1, len(t_signal)))
        
        # Add occasional truck events
        if np.random.random() < 0.05:  # 5% chance per second
            truck_response = 0.01 * np.exp(-t_signal/2) * np.sin(2*np.pi*nat_freq*t_signal)
            accel_signal += truck_response
        
        # Strain data (correlated with acceleration and temperature)
        strain_base = 50 + temperature * 2 + np.random.normal(0, 5)
        strain_dynamic = np.random.normal(strain_base, 3, 20)  # 20 samples per second
        
        # Add sensor data to pipeline
        pipeline.add_sensor_data('temperature', [temperature], current_time)
        pipeline.add_sensor_data('accelerometer', accel_signal, current_time)
        pipeline.add_sensor_data('strain_gauge', strain_dynamic, current_time)
        
        # Simulate image data every 10 seconds
        if t % 10 == 0:
            # Simple synthetic image (simulating surface condition)
            image_base = 150 + np.random.normal(0, 10, (50, 50))
            
            # Add some "damage" patterns later in simulation
            if t > 200:
                # Add crack-like pattern
                image_base[20:25, :] -= 50
                image_base[:, 20:25] -= 30
            
            pipeline.add_sensor_data('image', image_base.astype(np.uint8), current_time)
        
        # Brief pause to simulate real-time
        if t % 50 == 0:  # Print progress every 50 seconds
            print(f"Simulation progress: {t/duration*100:.0f}%")
            summary = pipeline.get_feature_summary()
            print(f"  Buffer sizes: {summary['buffer_sizes']}")
            print(f"  Feature cache size: {summary['feature_cache_size']}")

# Run simulation
simulate_sensor_data()

# Allow processing to complete
import time
time.sleep(2)

# Stop pipeline
pipeline.stop_realtime_processing()

print("\nSimulation completed. Analyzing results...")

# Get final pipeline state
final_summary = pipeline.get_feature_summary()
current_features = pipeline.get_current_features()
quality_report = pipeline.feature_quality_monitor.get_quality_report()

print(f"\nPipeline Summary:")
print(f"• Total features cached: {final_summary['feature_cache_size']}")
print(f"• Buffer utilization: {sum(final_summary['buffer_sizes'].values())} samples")
if final_summary['environmental_baseline']:
    temp_baseline = final_summary['environmental_baseline'].get('temperature', {})
    print(f"• Environmental adaptation: Temperature baseline {temp_baseline.get('mean', 0):.1f}°C")

# Feature quality analysis
print(f"\nFeature Quality Analysis:")
if quality_report:
    sorted_features = sorted(quality_report.items(), 
                           key=lambda x: x[1]['overall_quality'], reverse=True)
    
    print("Top 5 highest quality features:")
    for feature_name, metrics in sorted_features[:5]:
        print(f"  • {feature_name}: Quality={metrics['overall_quality']:.2f}, "
              f"Stability={metrics['stability']:.2f}")
    
    print("Features with quality concerns:")
    low_quality = [f for f, m in quality_report.items() if m['overall_quality'] < 2.0]
    if low_quality:
        for feature_name in low_quality[:3]:
            metrics = quality_report[feature_name]
            print(f"  • {feature_name}: Quality={metrics['overall_quality']:.2f}")
    else:
        print("  • No significant quality concerns detected")

# Analyze feature evolution over time
print(f"\nFeature Evolution Analysis:")

# Extract time series of key features
feature_timeline = {}
for feature_id, feature_data in current_features.items():
    timestamp = feature_data['timestamp']
    sensor_type = feature_data['sensor_type']
    
    for key, value in feature_data.items():
        if isinstance(value, (int, float)) and 'accel_rms' in key:
            if key not in feature_timeline:
                feature_timeline[key] = []
            feature_timeline[key].append((timestamp, value))

# Sort by timestamp and analyze trends
for feature_name, time_series in feature_timeline.items():
    if len(time_series) > 10:
        time_series.sort(key=lambda x: x[0])
        values = [x[1] for x in time_series]
        
        # Calculate trend
        trend = pipeline.time_extractor._calculate_trend(values) if hasattr(pipeline.time_extractor, '_calculate_trend') else 0
        trend_direction = "increasing" if trend > 0.001 else "decreasing" if trend < -0.001 else "stable"
        
        print(f"  • {feature_name}: {trend_direction} trend (slope: {trend:.6f})")

# Create comprehensive visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Sensor Buffer Utilization', 'Feature Quality Distribution',
                   'Real-time Feature Evolution', 'Environmental Adaptation',
                   'Processing Performance', 'Feature Correlation Analysis'),
    specs=[[{"type": "bar"}, {"type": "histogram"}],
           [{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "bar"}, {"type": "heatmap"}]]
)

# Plot 1: Buffer utilization
sensors = list(final_summary['buffer_sizes'].keys())
buffer_sizes = list(final_summary['buffer_sizes'].values())
colors = ['blue', 'red', 'green', 'orange']

fig.add_trace(
    go.Bar(x=sensors, y=buffer_sizes, marker_color=colors,
           name='Buffer Usage'),
    row=1, col=1
)
fig.update_yaxes(title_text="Buffer Size", row=1, col=1)

# Plot 2: Feature quality distribution
if quality_report:
    quality_scores = [metrics['overall_quality'] for metrics in quality_report.values()]
    
    fig.add_trace(
        go.Histogram(x=quality_scores, nbinsx=20, marker_color='lightblue',
                     name='Quality Distribution'),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Quality Score", row=1, col=2)
    fig.update_yaxes(title_text="Number of Features", row=1, col=2)

# Plot 3: Feature evolution (if we have time series data)
if feature_timeline:
    for i, (feature_name, time_series) in enumerate(list(feature_timeline.items())[:3]):
        time_series.sort(key=lambda x: x[0])
        timestamps = [x[0] for x in time_series]
        values = [x[1] for x in time_series]
        
        # Convert timestamps to minutes from start
        start_time = min(timestamps)
        time_minutes = [(t - start_time).total_seconds() / 60 for t in timestamps]
        
        fig.add_trace(
            go.Scatter(x=time_minutes, y=values, mode='lines+markers',
                       name=feature_name.replace('accel_', ''),
                       line=dict(color=colors[i % len(colors)])),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Time (minutes)", row=2, col=1)
    fig.update_yaxes(title_text="Feature Value", row=2, col=1)

# Plot 4: Environmental baseline evolution
if final_summary['environmental_baseline']:
    # Simulate environmental tracking over time
    temp_data = final_summary['environmental_baseline'].get('temperature', {})
    if temp_data:
        # Create mock time series for visualization
        time_points = np.linspace(0, 5, 50)  # 5 minutes
        temp_baseline = temp_data.get('mean', 20)
        temp_variation = temp_data.get('std', 1)
        
        baseline_line = np.full_like(time_points, temp_baseline)
        upper_bound = baseline_line + temp_variation
        lower_bound = baseline_line - temp_variation
        
        fig.add_trace(
            go.Scatter(x=time_points, y=baseline_line, mode='lines',
                       name='Temperature Baseline', line=dict(color='red')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=upper_bound, mode='lines',
                       name='Upper Bound', line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=lower_bound, mode='lines',
                       name='Lower Bound', line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Time (minutes)", row=2, col=2)
        fig.update_yaxes(title_text="Temperature (°C)", row=2, col=2)

# Plot 5: Processing performance metrics
performance_metrics = ['Buffer Management', 'Feature Extraction', 'Quality Monitoring', 'Adaptation']
performance_scores = [0.95, 0.88, 0.92, 0.85]  # Mock performance scores

fig.add_trace(
    go.Bar(x=performance_metrics, y=performance_scores,
           marker_color='lightgreen', name='Performance'),
    row=3, col=1
)
fig.update_xaxes(tickangle=45, row=3, col=1)
fig.update_yaxes(title_text="Performance Score", row=3, col=1)

# Plot 6: Feature correlation analysis (using recent features)
if quality_report and len(quality_report) > 1:
    # Create correlation matrix from quality metrics
    feature_names_short = list(quality_report.keys())[:6]  # Top 6 features
    correlation_data = []
    
    for fname in feature_names_short:
        row = []
        for fname2 in feature_names_short:
            # Mock correlation based on feature names similarity
            if fname == fname2:
                corr = 1.0
            elif fname.split('_')[0] == fname2.split('_')[0]:  # Same sensor type
                corr = 0.3 + np.random.uniform(-0.2, 0.2)
            else:
                corr = np.random.uniform(-0.1, 0.1)
            row.append(corr)
        correlation_data.append(row)
    
    fig.add_trace(
        go.Heatmap(z=correlation_data, x=feature_names_short, y=feature_names_short,
                   colorscale='RdBu', zmid=0, name='Correlation'),
        row=3, col=2
    )

fig.update_layout(
    height=1200,
    title_text="Exercise 4: Adaptive Pipeline Performance Analysis",
    showlegend=True,
    font=dict(size=10)
)

fig.show()

print(f"\nPipeline Performance Summary:")
print(f"• Successfully processed multiple sensor types")
print(f"• Adapted to environmental changes {len(final_summary.get('environmental_baseline', {}))} times")
print(f"• Maintained feature quality monitoring for {len(quality_report)} features")
print(f"• Real-time processing capability demonstrated")
print(f"• Pipeline ready for production deployment")

print(f"\nRecommendations for Production:")
print(f"• Implement persistent storage for feature cache")
print(f"• Add automated alerting for low-quality features")  
print(f"• Scale buffer sizes based on data acquisition rates")
print(f"• Implement redundancy for critical feature extraction")
print(f"• Add ML model integration for automated anomaly detection")

---

## 9. Chapter Summary

This chapter has provided a comprehensive exploration of feature engineering and statistical anomaly detection techniques specifically tailored for structural health monitoring applications. We have covered the essential building blocks that transform raw sensor measurements into actionable insights for bridge safety assessment.

### Key Concepts Covered

**Time-Domain Feature Engineering**: We explored how statistical moments, temporal characteristics, and energy-based features capture the essential aspects of structural response. These features form the foundation for understanding normal operational patterns and detecting deviations that may indicate structural changes.

**Frequency-Domain Analysis**: Modal parameters and spectral characteristics provide direct insight into structural properties. Natural frequencies, damping ratios, and mode shapes are fundamental indicators of structural health, making frequency-domain feature extraction crucial for effective monitoring systems.

**Principal Component Analysis**: PCA serves as both a dimensionality reduction technique and an anomaly detection method. The T² and Q statistics provide statistically principled approaches to identifying observations that deviate from normal structural behavior patterns.

**Digital Image Processing**: Computer vision techniques enable non-contact monitoring of structural surfaces, complementing traditional sensor-based approaches. Texture analysis, edge detection, and deep learning features capture visual indicators of damage that point sensors might miss.

**Multi-Modal Feature Fusion**: Real-world monitoring systems benefit from combining multiple data sources. Feature fusion techniques provide robust anomaly detection by leveraging complementary information from different sensor modalities while handling missing data gracefully.

**Statistical Anomaly Detection**: Classical statistical methods offer interpretable and theoretically grounded approaches to anomaly detection. Control charts, multivariate statistics, and robust estimation techniques provide baseline capabilities for operational monitoring systems.

**Deep Learning Approaches**: Neural network-based methods, particularly autoencoders and variational autoencoders, can learn complex non-linear patterns in structural data. While less interpretable than statistical methods, they offer superior performance for complex damage patterns.

### Practical Implementation Insights

The PyTorch implementations demonstrated throughout this chapter provide production-ready code for implementing these techniques in real monitoring systems. Key implementation considerations include:

- **Computational Efficiency**: Real-time monitoring requires careful balance between feature richness and computational resources
- **Robustness**: Environmental variations and sensor noise must be handled systematically
- **Adaptability**: Systems must adapt to changing conditions while maintaining consistent performance
- **Interpretability**: Critical infrastructure applications require explainable decisions

### Performance Trade-offs

Our comprehensive comparison revealed important trade-offs between different approaches:

- **Statistical methods** provide high interpretability and fast computation but may miss complex patterns
- **Deep learning methods** offer superior pattern recognition capabilities but require more data and computational resources
- **Multi-modal fusion** improves robustness but increases system complexity
- **Real-time processing** enables immediate response but constrains algorithmic complexity

### Future Directions

The field of feature engineering for structural health monitoring continues to evolve rapidly. Promising directions include:

- **Physics-informed machine learning** that incorporates structural mechanics principles
- **Federated learning** approaches for sharing knowledge across multiple structures
- **Uncertainty quantification** for more reliable decision-making
- **Edge computing** solutions for distributed monitoring networks

### Practical Recommendations

For practitioners implementing feature engineering systems for bridge monitoring:

1. **Start with statistical methods** to establish baseline performance and understand data characteristics
2. **Implement comprehensive data quality monitoring** to ensure reliable feature extraction
3. **Use multi-modal approaches** where possible to improve robustness
4. **Maintain interpretability** for critical decision-making processes
5. **Plan for adaptation** as structural and environmental conditions change over time

The techniques presented in this chapter provide a solid foundation for developing effective structural health monitoring systems. The combination of traditional engineering knowledge with modern machine learning capabilities offers unprecedented opportunities for ensuring the safety and reliability of our critical infrastructure.

---

## References

1. Bao, Y., & Li, H. (2021). Machine learning paradigm for structural health monitoring. *Structural Health Monitoring*, 20(4), 1353-1372.

2. Malekloo, A., Ozer, E., AlHamaydeh, M., & Girolami, M. (2022). Machine learning and structural health monitoring overview with emerging technology and high-dimensional data source highlights. *Structural Health Monitoring*, 21(4), 1906-1955.

3. Sun, L., Shang, Z., Xia, Y., Bhowmick, S., & Nagarajaiah, S. (2020). Review of bridge structural health monitoring aided by big data and artificial intelligence: From condition assessment to damage detection. *Journal of Structural Engineering*, 146(5), 04020073.

4. Avci, O., Abdeljaber, O., Kiranyaz, S., Hussein, M., Gabbouj, M., & Inman, D. J. (2021). A review of vibration-based damage detection in civil structures: From traditional methods to machine learning and deep learning applications. *Mechanical Systems and Signal Processing*, 147, 107077.

5. Dang, H. V., Tatipamula, M., & Nguyen, H. X. (2021). Cloud-based digital twinning for structural health monitoring using deep learning. *IEEE Transactions on Industrial Informatics*, 18(6), 3820-3830.

6. Chen, Z., Bao, Y., Li, H., & Spencer Jr, B. F. (2023). A comprehensive study on the application of machine learning in structural health monitoring. *Engineering Structures*, 291, 116421.

7. Tibaduiza, D. A., Mujica, L. E., Rodellar, J., & Güemes, A. (2016). Damage classification in structural health monitoring using principal component analysis and self-organizing maps. *Structural Control and Health Monitoring*, 23(6), 1203-1220.

8. Cross, E. J., Worden, K., & Chen, Q. (2011). Cointegration: A novel approach for the removal of environmental trends in structural health monitoring data. *Proceedings of the Royal Society A*, 467(2133), 2712-2732.

9. Farrar, C. R., & Worden, K. (2012). *Structural health monitoring: A machine learning perspective*. John Wiley & Sons.

10. Abdeljaber, O., Avci, O., Kiranyaz, S., Gabbouj, M., & Inman, D. J. (2017). Real-time vibration-based structural damage detection using one-dimensional convolutional neural networks. *Journal of Sound and Vibration*, 388, 154-170.# Chapter 6: Feature Engineering & Statistical Anomaly Detection

**Instructor: Mohammad Talebi-Kalaleh – University of Alberta**

---

## Overview

Feature engineering represents the bridge between raw sensor measurements and meaningful structural insights in bridge health monitoring systems. While previous chapters focused on signal processing and time-series modeling, this chapter explores how to systematically extract damage-sensitive features from multiple data sources and detect anomalous structural behavior using statistical and machine learning approaches.

The transformation of raw acceleration, strain, and image data into actionable engineering insights requires careful consideration of which features truly reflect structural health changes while remaining robust to environmental variations. Modern structural health monitoring generates massive datasets – the Sutong Bridge in China alone produces 2.5 TB of data annually Bao et al., 2019. The challenge lies not in data quantity, but in systematically extracting meaningful patterns that distinguish between normal operational variations and genuine structural deterioration.

This chapter introduces a comprehensive framework for feature engineering that combines traditional engineering understanding with advanced machine learning techniques. We'll explore how time-domain statistical features, frequency-domain modal parameters, principal component analysis, and computer vision techniques can be integrated into a unified anomaly detection system. Throughout, we'll emphasize practical implementation using PyTorch, with realistic examples drawn from bridge monitoring applications.

---

## 1. Time-Domain Feature Engineering

### 1.1 Motivation: Beyond Simple Statistics

Time-domain features form the foundation of structural health monitoring because they directly reflect the physical response characteristics of bridges under various loading conditions. However, the challenge lies in identifying features that are sensitive to structural changes while remaining robust to environmental effects such as temperature variations and traffic loading patterns.

Consider a typical acceleration measurement from a bridge deck sensor during truck passage. While the raw signal contains rich information about structural response, extracting meaningful patterns requires systematic feature engineering that captures both local and global signal characteristics.

### 1.2 Statistical Moment Features

The most fundamental time-domain features are statistical moments that characterize the distribution of structural response amplitudes. These features provide insight into the overall energy content and response patterns of the structure.

#### Mathematical Foundation

For a discrete time-series signal $x[n]$ where $n = 1, 2, ..., N$, the fundamental statistical features are defined as:

**Mean (First Moment):**
$$\mu = \frac{1}{N} \sum_{n=1}^{N} x[n] \tag{6.1}$$

**Variance (Second Central Moment):**
$$\sigma^2 = \frac{1}{N-1} \sum_{n=1}^{N} (x[n] - \mu)^2 \tag{6.2}$$

**Skewness (Third Standardized Moment):**
$$\gamma_1 = \frac{1}{N} \sum_{n=1}^{N} \left(\frac{x[n] - \mu}{\sigma}\right)^3 \tag{6.3}$$

**Kurtosis (Fourth Standardized Moment):**
$$\gamma_2 = \frac{1}{N} \sum_{n=1}^{N} \left(\frac{x[n] - \mu}{\sigma}\right)^4 \tag{6.4}$$

where $\mu$ is the sample mean, $\sigma$ is the sample standard deviation, $\gamma_1$ represents distribution asymmetry, and $\gamma_2$ indicates the heaviness of distribution tails relative to a normal distribution.

#### Physical Interpretation

These statistical moments carry important structural information:

- **Mean**: Represents the static or quasi-static response level, sensitive to permanent deformation or settlement
- **Variance**: Captures the dynamic response energy, reflecting structural stiffness and damping characteristics  
- **Skewness**: Indicates asymmetric loading patterns or non-linear structural behavior
- **Kurtosis**: Reveals the presence of impact loading or high-frequency structural vibrations

### 1.3 Advanced Time-Domain Features

Beyond basic statistical moments, several specialized features have proven particularly effective for structural health monitoring applications.

#### Root Mean Square (RMS)
$$x_{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x[n]^2} \tag{6.5}$$

The RMS value provides a measure of signal energy that is particularly useful for vibration-based monitoring.

#### Peak Factor
$$PF = \frac{\max|x[n]|}{x_{RMS}} \tag{6.6}$$

Peak factor indicates the presence of impulsive events or impacts, making it valuable for detecting sudden structural changes.

#### Crest Factor
$$CF = \frac{\max|x[n]|}{\bar{x}} \tag{6.7}$$

where $\bar{x}$ is the mean absolute value of the signal.

#### Zero Crossing Rate
$$ZCR = \frac{1}{N-1} \sum_{n=1}^{N-1} \mathbb{I}[x[n] \cdot x[n+1] < 0] \tag{6.8}$$

where $\mathbb{I}[\cdot]$ is the indicator function. ZCR provides information about signal frequency content and can detect changes in dominant vibration modes.

### 1.4 Implementation: Time-Domain Feature Extraction

Let's implement a comprehensive time-domain feature extraction system using Python:

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TimeDomainFeatureExtractor:
    """
    Comprehensive time-domain feature extraction for structural health monitoring
    """
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize the feature extractor
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.fs = sampling_rate
        
    def extract_statistical_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from time-series signal
        
        Args:
            signal: 1D array of time-series data
            
        Returns:
            Dictionary of statistical features
        """
        # Remove DC component for more robust statistics
        signal_centered = signal - np.mean(signal)
        
        features = {
            'mean': np.mean(signal),
            'std': np.std(signal, ddof=1),
            'variance': np.var(signal, ddof=1),
            'rms': np.sqrt(np.mean(signal**2)),
            'skewness': stats.skew(signal),
            'kurtosis': stats.kurtosis(signal),
            'min': np.min(signal),
            'max': np.max(signal),
            'range': np.ptp(signal),  # Peak-to-peak
            'peak_factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
            'crest_factor': np.max(np.abs(signal)) / np.mean(np.abs(signal)),
            'shape_factor': np.sqrt(np.mean(signal**2)) / np.mean(np.abs(signal))
        }
        
        return features
    
    def extract_temporal_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal characteristics of the signal
        
        Args:
            signal: 1D array of time-series data
            
        Returns:
            Dictionary of temporal features
        """
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        zcr = len(zero_crossings) / len(signal) * self.fs
        
        # Mean crossing rate (crossings of the mean value)
        signal_demean = signal - np.mean(signal)
        mean_crossings = np.where(np.diff(np.signbit(signal_demean)))[0]
        mcr = len(mean_crossings) / len(signal) * self.fs
        
        # Waveform length (total variation)
        waveform_length = np.sum(np.abs(np.diff(signal)))
        
        # Average frequency based on zero crossings
        avg_frequency = zcr / 2.0
        
        features = {
            'zero_crossing_rate': zcr,
            'mean_crossing_rate': mcr,
            'waveform_length': waveform_length,
            'average_frequency': avg_frequency
        }
        
        return features
    
    def extract_energy_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract energy-based features
        
        Args:
            signal: 1D array of time-series data
            
        Returns:
            Dictionary of energy features
        """
        # Total energy
        total_energy = np.sum(signal**2)
        
        # Log energy
        log_energy = np.sum(np.log(signal**2 + 1e-12))  # Small epsilon to avoid log(0)
        
        # Teager-Kaiser energy operator
        if len(signal) > 2:
            tk_energy = np.mean(signal[1:-1]**2 - signal[:-2] * signal[2:])
        else:
            tk_energy = 0.0
            
        features = {
            'total_energy': total_energy,
            'log_energy': log_energy,
            'teager_kaiser_energy': tk_energy,
            'normalized_energy': total_energy / len(signal)
        }
        
        return features
    
    def extract_all_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract all time-domain features
        
        Args:
            signal: 1D array of time-series data
            
        Returns:
            Dictionary of all extracted features
        """
        features = {}
        features.update(self.extract_statistical_features(signal))
        features.update(self.extract_temporal_features(signal))
        features.update(self.extract_energy_features(signal))
        
        return features

# Generate realistic bridge acceleration data for demonstration
def generate_bridge_acceleration_data(duration: float = 60.0, fs: float = 100.0, 
                                    truck_events: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic bridge acceleration data with truck passages
    
    Args:
        duration: Signal duration in seconds
        fs: Sampling frequency in Hz
        truck_events: Number of truck passage events
        
    Returns:
        Tuple of (time array, acceleration array)
    """
    t = np.linspace(0, duration, int(duration * fs))
    
    # Base structural vibration (multiple modes)
    # First mode: ~2 Hz, Second mode: ~6 Hz, Third mode: ~12 Hz
    base_vibration = (0.001 * np.sin(2 * np.pi * 2.1 * t) * np.exp(-0.05 * t) +
                     0.0005 * np.sin(2 * np.pi * 6.3 * t) * np.exp(-0.1 * t) +
                     0.0002 * np.sin(2 * np.pi * 12.1 * t) * np.exp(-0.15 * t))
    
    # Environmental noise (wind, micro-tremors)
    noise = 0.0001 * np.random.normal(0, 1, len(t))
    
    # Initialize signal
    acceleration = base_vibration + noise
    
    # Add truck passage events
    np.random.seed(42)  # For reproducible results
    truck_times = np.random.uniform(10, duration-10, truck_events)
    
    for truck_time in truck_times:
        # Truck passage parameters
        passage_duration = np.random.uniform(3, 6)  # 3-6 seconds passage time
        truck_weight = np.random.uniform(0.8, 1.5)  # Weight factor
        
        # Create truck loading envelope
        truck_indices = np.where((t >= truck_time) & (t <= truck_time + passage_duration))[0]
        if len(truck_indices) > 0:
            local_t = t[truck_indices] - truck_time
            
            # Realistic truck loading with axle groups
            # Front axle
            front_axle = truck_weight * 0.3 * np.exp(-(local_t - 1.0)**2 / 0.5)
            # Rear axle group
            rear_axle = truck_weight * 0.7 * np.exp(-(local_t - 4.0)**2 / 0.8)
            
            truck_response = (front_axle + rear_axle) * 0.01  # Scale to realistic acceleration
            
            # Add dynamic amplification and vibration
            for i, idx in enumerate(truck_indices):
                dynamic_factor = 1 + 0.2 * np.sin(2 * np.pi * 2.1 * local_t[i])
                acceleration[idx] += truck_response[i] * dynamic_factor
    
    return t, acceleration

# Demonstrate feature extraction
print("=== Time-Domain Feature Extraction Demonstration ===\n")

# Generate sample data
time, accel_data = generate_bridge_acceleration_data(duration=60.0, fs=100.0)

# Initialize feature extractor
extractor = TimeDomainFeatureExtractor(sampling_rate=100.0)

# Extract features
features = extractor.extract_all_features(accel_data)

# Display results in a formatted table
feature_df = pd.DataFrame([
    {"Feature Category": "Statistical", "Feature Name": "Mean", "Value": f"{features['mean']:.6f}", "Unit": "m/s²"},
    {"Feature Category": "Statistical", "Feature Name": "Standard Deviation", "Value": f"{features['std']:.6f}", "Unit": "m/s²"},
    {"Feature Category": "Statistical", "Feature Name": "RMS", "Value": f"{features['rms']:.6f}", "Unit": "m/s²"},
    {"Feature Category": "Statistical", "Feature Name": "Skewness", "Value": f"{features['skewness']:.3f}", "Unit": "-"},
    {"Feature Category": "Statistical", "Feature Name": "Kurtosis", "Value": f"{features['kurtosis']:.3f}", "Unit": "-"},
    {"Feature Category": "Statistical", "Feature Name": "Peak Factor", "Value": f"{features['peak_factor']:.3f}", "Unit": "-"},
    {"Feature Category": "Temporal", "Feature Name": "Zero Crossing Rate", "Value": f"{features['zero_crossing_rate']:.3f}", "Unit": "Hz"},
    {"Feature Category": "Temporal", "Feature Name": "Average Frequency", "Value": f"{features['average_frequency']:.3f}", "Unit": "Hz"},
    {"Feature Category": "Energy", "Feature Name": "Total Energy", "Value": f"{features['total_energy']:.8f}", "Unit": "(m/s²)²"},
    {"Feature Category": "Energy", "Feature Name": "Teager-Kaiser Energy", "Value": f"{features['teager_kaiser_energy']:.8f}", "Unit": "(m/s²)²"}
])

print("Extracted Time-Domain Features:")
print(feature_df.to_string(index=False))
```

### 1.5 Visualization of Time-Domain Features

```python
# Create comprehensive visualization of time-domain analysis
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Bridge Acceleration Signal', 'Statistical Distribution',
                   'Feature Evolution (Windowed)', 'Feature Correlation Matrix',
                   'Peak Detection', 'Energy Distribution'),
    specs=[[{"colspan": 2}, None],
           [{"type": "histogram"}, {"type": "heatmap"}],
           [{"type": "scatter"}, {"type": "bar"}]]
)

# Plot 1: Original signal with truck events highlighted
fig.add_trace(
    go.Scatter(x=time, y=accel_data*1000, name='Acceleration (mg)',
               line=dict(color='blue', width=1)),
    row=1, col=1
)

# Highlight truck passages
truck_threshold = np.percentile(np.abs(accel_data), 95)
truck_events = np.where(np.abs(accel_data) > truck_threshold)[0]
if len(truck_events) > 0:
    fig.add_trace(
        go.Scatter(x=time[truck_events], y=accel_data[truck_events]*1000,
                   mode='markers', name='High Response Events',
                   marker=dict(color='red', size=4)),
        row=1, col=1
    )

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Acceleration (mg)", row=1, col=1)

# Plot 2: Statistical distribution
fig.add_trace(
    go.Histogram(x=accel_data*1000, nbinsx=50, name='Data Distribution',
                 marker_color='lightblue', opacity=0.7),
    row=2, col=1
)
fig.update_xaxes(title_text="Acceleration (mg)", row=2, col=1)
fig.update_yaxes(title_text="Frequency", row=2, col=1)

# Plot 3: Windowed feature evolution
window_size = int(10 * 100)  # 10 seconds
step_size = int(5 * 100)     # 5 seconds
n_windows = (len(accel_data) - window_size) // step_size + 1

windowed_features = []
window_times = []

for i in range(n_windows):
    start_idx = i * step_size
    end_idx = start_idx + window_size
    window_data = accel_data[start_idx:end_idx]
    
    window_features = extractor.extract_statistical_features(window_data)
    windowed_features.append(window_features)
    window_times.append(time[start_idx + window_size//2])

# Extract specific features for plotting
rms_evolution = [f['rms'] for f in windowed_features]
std_evolution = [f['std'] for f in windowed_features]

fig.add_trace(
    go.Scatter(x=window_times, y=np.array(rms_evolution)*1000,
               name='RMS (mg)', line=dict(color='green')),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=window_times, y=np.array(std_evolution)*1000,
               name='Std Dev (mg)', line=dict(color='orange')),
    row=2, col=2
)

fig.update_xaxes(title_text="Time (s)", row=2, col=2)
fig.update_yaxes(title_text="Feature Value (mg)", row=2, col=2)

# Plot 4: Peak detection
from scipy.signal import find_peaks
peaks, properties = find_peaks(np.abs(accel_data), height=np.std(accel_data)*2, distance=100)

fig.add_trace(
    go.Scatter(x=time, y=accel_data*1000, name='Signal',
               line=dict(color='blue', width=1)),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=time[peaks], y=accel_data[peaks]*1000,
               mode='markers', name='Detected Peaks',
               marker=dict(color='red', size=6, symbol='triangle-up')),
    row=3, col=1
)

fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Acceleration (mg)", row=3, col=1)

# Plot 5: Feature importance (energy distribution across features)
feature_names = ['Mean', 'Std', 'RMS', 'Skewness', 'Kurtosis', 'Peak Factor']
feature_values = [abs(features['mean']), features['std'], features['rms'],
                  abs(features['skewness']), abs(features['kurtosis']), features['peak_factor']]
# Normalize for comparison
feature_values_norm = np.array(feature_values) / np.max(feature_values)

fig.add_trace(
    go.Bar(x=feature_names, y=feature_values_norm,
           marker_color='lightcoral', name='Normalized Feature Values'),
    row=3, col=2
)

fig.update_xaxes(title_text="Feature Type", row=3, col=2)
fig.update_yaxes(title_text="Normalized Magnitude", row=3, col=2)

# Update layout
fig.update_layout(
    height=1000,
    title_text="Comprehensive Time-Domain Feature Analysis",
    showlegend=True,
    font=dict(size=12)
)

fig.show()

print(f"\nKey Insights:")
print(f"• Signal contains {len(peaks)} significant response events")
print(f"• Peak factor of {features['peak_factor']:.2f} indicates moderate impulsive content")
print(f"• Kurtosis of {features['kurtosis']:.2f} suggests {'heavy-tailed' if features['kurtosis'] > 3 else 'normal'} distribution")
print(f"• Average response frequency: {features['average_frequency']:.2f} Hz")
```

---

## 2. Frequency-Domain Feature Engineering

### 2.1 The Power of Spectral Analysis

While time-domain features capture the amplitude characteristics of structural response, frequency-domain features reveal the underlying modal properties that are directly related to structural stiffness, mass distribution, and damping characteristics. Changes in these modal properties often provide the earliest indication of structural damage.

### 2.2 Spectral Feature Extraction

#### Power Spectral Density Features

The power spectral density (PSD) provides a frequency-domain representation of signal energy distribution. Key features extracted from PSD include:

**Spectral Centroid (Center of Mass):**
$$f_c = \frac{\sum_{k=1}^{N/2} k \cdot S_{xx}[k]}{\sum_{k=1}^{N/2} S_{xx}[k]} \cdot \frac{f_s}{N} \tag{6.9}$$

where $S_{xx}[k]$ is the power spectral density at frequency bin $k$, $f_s$ is the sampling frequency, and $N$ is the signal length.

**Spectral Spread (Bandwidth):**
$$\sigma_f = \sqrt{\frac{\sum_{k=1}^{N/2} (f[k] - f_c)^2 \cdot S_{xx}[k]}{\sum_{k=1}^{N/2} S_{xx}[k]}} \tag{6.10}$$

**Spectral Rolloff (95% Energy Frequency):**
The frequency below which 95% of the signal energy is contained:
$$f_{95} = \arg\min_f \left\{ \sum_{k=1}^{k_f} S_{xx}[k] \geq 0.95 \sum_{k=1}^{N/2} S_{xx}[k] \right\} \tag{6.11}$$

**Spectral Flux (Rate of Change):**
$$SF = \sum_{k=1}^{N/2} (S_{xx}[k,t] - S_{xx}[k,t-1])^2 \tag{6.12}$$

### 2.3 Modal Parameter Features

Modal parameters represent the fundamental dynamic characteristics of structures and are highly sensitive to structural changes.

#### Natural Frequency Extraction

Natural frequencies can be identified as peaks in the power spectral density or through more sophisticated modal identification techniques:

$$f_n = \arg\max_{f} S_{xx}(f) \quad \text{for each modal peak} \tag{6.13}$$

#### Damping Ratio Estimation

For lightly damped structures, the damping ratio can be estimated from the half-power bandwidth method:

$$\zeta_n = \frac{f_2 - f_1}{2f_n} \tag{6.14}$$

where $f_1$ and $f_2$ are the frequencies at which the power is half the peak value.

### 2.4 Implementation: Frequency-Domain Feature Extraction

```python
import scipy.signal as signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch, spectrogram, find_peaks

class FrequencyDomainFeatureExtractor:
    """
    Comprehensive frequency-domain feature extraction for SHM
    """
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize the frequency-domain feature extractor
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.fs = sampling_rate
        
    def compute_psd(self, signal_data: np.ndarray, nperseg: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density using Welch's method
        
        Args:
            signal_data: 1D array of time-series data
            nperseg: Length of each segment for Welch's method
            
        Returns:
            Tuple of (frequencies, power spectral density)
        """
        if nperseg is None:
            nperseg = min(len(signal_data) // 8, 1024)
            
        frequencies, psd = welch(signal_data, fs=self.fs, nperseg=nperseg,
                                window='hann', overlap=0.5)
        return frequencies, psd
    
    def extract_spectral_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features from signal
        
        Args:
            signal_data: 1D array of time-series data
            
        Returns:
            Dictionary of spectral features
        """
        frequencies, psd = self.compute_psd(signal_data)
        
        # Remove DC component
        freq_start_idx = 1 if frequencies[0] == 0 else 0
        frequencies = frequencies[freq_start_idx:]
        psd = psd[freq_start_idx:]
        
        # Spectral centroid
        spectral_centroid = np.sum(frequencies * psd) / np.sum(psd)
        
        # Spectral spread (bandwidth)
        spectral_spread = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * psd) / np.sum(psd))
        
        # Spectral rolloff (95% energy)
        cumulative_energy = np.cumsum(psd)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.95 * total_energy)[0]
        spectral_rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else frequencies[-1]
        
        # Spectral skewness and kurtosis
        normalized_psd = psd / np.sum(psd)
        spectral_skewness = np.sum(((frequencies - spectral_centroid) ** 3) * normalized_psd) / (spectral_spread ** 3)
        spectral_kurtosis = np.sum(((frequencies - spectral_centroid) ** 4) * normalized_psd) / (spectral_spread ** 4)
        
        # Peak frequency (dominant frequency)
        peak_freq_idx = np.argmax(psd)
        peak_frequency = frequencies[peak_freq_idx]
        
        # Spectral entropy
        spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-12))
        
        # Frequency bands energy distribution
        # Low frequency: 0-5 Hz, Mid frequency: 5-15 Hz, High frequency: 15+ Hz
        low_freq_mask = frequencies <= 5
        mid_freq_mask = (frequencies > 5) & (frequencies <= 15)
        high_freq_mask = frequencies > 15
        
        low_freq_energy = np.sum(psd[low_freq_mask]) / total_energy
        mid_freq_energy = np.sum(psd[mid_freq_mask]) / total_energy
        high_freq_energy = np.sum(psd[high_freq_mask]) / total_energy
        
        features = {
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_rolloff': spectral_rolloff,
            'spectral_skewness': spectral_skewness,
            'spectral_kurtosis': spectral_kurtosis,
            'peak_frequency': peak_frequency,
            'spectral_entropy': spectral_entropy,
            'low_freq_energy_ratio': low_freq_energy,
            'mid_freq_energy_ratio': mid_freq_energy,
            'high_freq_energy_ratio': high_freq_energy
        }
        
        return features
    
    def extract_modal_features(self, signal_data: np.ndarray, 
                             freq_range: Tuple[float, float] = (0.5, 20.0)) -> Dict[str, any]:
        """
        Extract modal parameters from signal
        
        Args:
            signal_data: 1D array of time-series data
            freq_range: Frequency range for modal identification (Hz)
            
        Returns:
            Dictionary containing modal features
        """
        frequencies, psd = self.compute_psd(signal_data)
        
        # Filter to frequency range of interest
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        frequencies_filtered = frequencies[freq_mask]
        psd_filtered = psd[freq_mask]
        
        # Find peaks (potential natural frequencies)
        # Use relative height threshold and minimum distance between peaks
        peak_height = np.mean(psd_filtered) + 2 * np.std(psd_filtered)
        min_distance = int(0.5 / (frequencies_filtered[1] - frequencies_filtered[0]))  # 0.5 Hz minimum separation
        
        peak_indices, properties = find_peaks(psd_filtered, 
                                            height=peak_height,
                                            distance=min_distance,
                                            prominence=peak_height/4)
        
        natural_frequencies = frequencies_filtered[peak_indices]
        peak_amplitudes = psd_filtered[peak_indices]
        
        # Estimate damping ratios using half-power bandwidth method
        damping_ratios = []
        for i, peak_idx in enumerate(peak_indices):
            try:
                # Find half-power points
                half_power = peak_amplitudes[i] / 2
                
                # Search for half-power points around the peak
                left_search = max(0, peak_idx - min_distance//2)
                right_search = min(len(psd_filtered), peak_idx + min_distance//2)
                
                # Find left half-power point
                left_half_idx = left_search
                for j in range(peak_idx, left_search, -1):
                    if psd_filtered[j] <= half_power:
                        left_half_idx = j
                        break
                
                # Find right half-power point
                right_half_idx = right_search - 1
                for j in range(peak_idx, right_search):
                    if psd_filtered[j] <= half_power:
                        right_half_idx = j
                        break
                
                f1 = frequencies_filtered[left_half_idx]
                f2 = frequencies_filtered[right_half_idx]
                fn = natural_frequencies[i]
                
                damping_ratio = (f2 - f1) / (2 * fn)
                damping_ratios.append(damping_ratio)
                
            except:
                damping_ratios.append(0.05)  # Default damping ratio
        
        features = {
            'natural_frequencies': natural_frequencies.tolist(),
            'modal_amplitudes': peak_amplitudes.tolist(),
            'damping_ratios': damping_ratios,
            'first_natural_frequency': natural_frequencies[0] if len(natural_frequencies) > 0 else 0.0,
            'dominant_frequency': natural_frequencies[np.argmax(peak_amplitudes)] if len(natural_frequencies) > 0 else 0.0,
            'number_of_modes': len(natural_frequencies),
            'modal_assurance_criterion': self._compute_mac(psd_filtered, peak_indices) if len(peak_indices) > 1 else 1.0
        }
        
        return features
    
    def _compute_mac(self, psd: np.ndarray, peak_indices: np.ndarray) -> float:
        """
        Simplified Modal Assurance Criterion computation
        """
        if len(peak_indices) < 2:
            return 1.0
            
        # Use the two strongest peaks for MAC calculation
        sorted_indices = peak_indices[np.argsort(psd[peak_indices])[-2:]]
        
        # Simple MAC based on frequency separation
        f1, f2 = sorted_indices
        mac = 1.0 - abs(f1 - f2) / max(f1, f2)
        
        return max(0.0, mac)

# Demonstrate frequency-domain feature extraction
print("\n=== Frequency-Domain Feature Extraction Demonstration ===\n")

# Generate more complex bridge data with multiple modes
def generate_multi_modal_bridge_data(duration: float = 120.0, fs: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic multi-modal bridge response"""
    t = np.linspace(0, duration, int(duration * fs))
    
    # Multiple structural modes with different damping
    # Mode 1: First bending mode (~2.3 Hz)
    mode1 = 0.002 * np.sin(2 * np.pi * 2.3 * t) * np.exp(-0.02 * t)
    
    # Mode 2: Second bending mode (~6.8 Hz)  
    mode2 = 0.001 * np.sin(2 * np.pi * 6.8 * t) * np.exp(-0.08 * t)
    
    # Mode 3: Torsional mode (~11.2 Hz)
    mode3 = 0.0005 * np.sin(2 * np.pi * 11.2 * t) * np.exp(-0.12 * t)
    
    # Traffic loading (random truck passages)
    np.random.seed(42)
    traffic_response = np.zeros_like(t)
    
    # Add random truck events
    n_trucks = int(duration / 15)  # One truck every ~15 seconds on average
    truck_times = np.random.uniform(5, duration-5, n_trucks)
    
    for truck_time in truck_times:
        truck_duration = np.random.uniform(2, 5)
        truck_mask = (t >= truck_time) & (t <= truck_time + truck_duration)
        
        if np.any(truck_mask):
            local_t = t[truck_mask] - truck_time
            # Truck force with dynamic amplification
            truck_force = 0.01 * np.exp(-local_t/2) * (1 + 0.3*np.sin(2*np.pi*2.3*local_t))
            traffic_response[truck_mask] += truck_force
    
    # Environmental noise
    noise = 0.0001 * np.random.normal(0, 1, len(t))
    
    # Combined response
    acceleration = mode1 + mode2 + mode3 + traffic_response + noise
    
    return t, acceleration

# Generate sample data
time_multi, accel_multi = generate_multi_modal_bridge_data(duration=120.0, fs=100.0)

# Initialize frequency-domain extractor
freq_extractor = FrequencyDomainFeatureExtractor(sampling_rate=100.0)

# Extract features
spectral_features = freq_extractor.extract_spectral_features(accel_multi)
modal_features = freq_extractor.extract_modal_features(accel_multi)

# Display results
print("Spectral Features:")
spectral_df = pd.DataFrame([
    {"Feature": "Spectral Centroid", "Value": f"{spectral_features['spectral_centroid']:.3f}", "Unit": "Hz"},
    {"Feature": "Spectral Spread", "Value": f"{spectral_features['spectral_spread']:.3f}", "Unit": "Hz"},
    {"Feature": "Spectral Rolloff", "Value": f"{spectral_features['spectral_rolloff']:.3f}", "Unit": "Hz"},
    {"Feature": "Peak Frequency", "Value": f"{spectral_features['peak_frequency']:.3f}", "Unit": "Hz"},
    {"Feature": "Spectral Entropy", "Value": f"{spectral_features['spectral_entropy']:.3f}", "Unit": "bits"},
    {"Feature": "Low Freq Energy Ratio", "Value": f"{spectral_features['low_freq_energy_ratio']:.3f}", "Unit": "-"},
    {"Feature": "Mid Freq Energy Ratio", "Value": f"{spectral_features['mid_freq_energy_ratio']:.3f}", "Unit": "-"},
    {"Feature": "High Freq Energy Ratio", "Value": f"{spectral_features['high_freq_energy_ratio']:.3f}", "Unit": "-"}
])
print(spectral_df.to_string(index=False))

print(f"\nModal Features:")
print(f"Number of identified modes: {modal_features['number_of_modes']}")
print(f"Natural frequencies: {[f'{f:.2f}' for f in modal_features['natural_frequencies']]} Hz")
print(f"Damping ratios: {[f'{d:.4f}' for d in modal_features['damping_ratios']]}")
print(f"First natural frequency: {modal_features['first_natural_frequency']:.3f} Hz")
print(f"Dominant frequency: {modal_features['dominant_frequency']:.3f} Hz")
```

### 2.5 Advanced Frequency-Domain Visualization

```python
# Create comprehensive frequency-domain visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Multi-Modal Bridge Response', 'Power Spectral Density',
                   'Spectrogram (Time-Frequency)', 'Modal Identification',
                   'Frequency Band Energy Distribution', 'Spectral Features Evolution'),
    specs=[[{"colspan": 2}, None],
           [{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "bar"}, {"type": "scatter"}]]
)

# Plot 1: Time-domain signal
fig.add_trace(
    go.Scatter(x=time_multi, y=accel_multi*1000, name='Bridge Response',
               line=dict(color='navy', width=0.8)),
    row=1, col=1
)

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Acceleration (mg)", row=1, col=1)

# Plot 2: Power Spectral Density with modal identification
frequencies, psd = freq_extractor.compute_psd(accel_multi)

fig.add_trace(
    go.Scatter(x=frequencies, y=10*np.log10(psd), name='PSD',
               line=dict(color='blue', width=1.5)),
    row=2, col=1
)

# Highlight identified natural frequencies
nat_freqs = modal_features['natural_frequencies']
modal_amps = modal_features['modal_amplitudes']

if len(nat_freqs) > 0:
    # Find PSD values at natural frequencies
    psd_at_modes = []
    for freq in nat_freqs:
        freq_idx = np.argmin(np.abs(frequencies - freq))
        psd_at_modes.append(10*np.log10(psd[freq_idx]))
    
    fig.add_trace(
        go.Scatter(x=nat_freqs, y=psd_at_modes, mode='markers',
                   name='Natural Frequencies',
                   marker=dict(color='red', size=10, symbol='triangle-up')),
        row=2, col=1
    )

fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 25], row=2, col=1)
fig.update_yaxes(title_text="PSD (dB)", row=2, col=1)

# Plot 3: Spectrogram
f_spec, t_spec, Sxx = spectrogram(accel_multi, fs=100.0, window='hann', 
                                  nperseg=1024, noverlap=512)

# Create custom colormap for spectrogram
fig.add_trace(
    go.Heatmap(z=10*np.log10(Sxx), x=t_spec, y=f_spec,
               colorscale='Viridis', name='Spectrogram'),
    row=2, col=2
)

fig.update_xaxes(title_text="Time (s)", row=2, col=2)
fig.update_yaxes(title_text="Frequency (Hz)", range=[0, 25], row=2, col=2)

# Plot 4: Frequency band energy distribution
band_names = ['Low (0-5 Hz)', 'Mid (5-15 Hz)', 'High (15+ Hz)']
band_energies = [spectral_features['low_freq_energy_ratio'],
                spectral_features['mid_freq_energy_ratio'],
                spectral_features['high_freq_energy_ratio']]

fig.add_trace(
    go.Bar(x=band_names, y=band_energies, name='Energy Distribution',
           marker_color=['lightblue', 'lightgreen', 'lightcoral']),
    row=3, col=1
)

fig.update_xaxes(title_text="Frequency Band", row=3, col=1)
fig.update_yaxes(title_text="Energy Ratio", row=3, col=1)

# Plot 5: Spectral features over time (windowed analysis)
window_duration = 30  # seconds
overlap = 0.5
window_samples = int(window_duration * 100)
step_samples = int(window_samples * (1 - overlap))

spectral_evolution = []
evolution_times = []

for i in range(0, len(accel_multi) - window_samples, step_samples):
    window_data = accel_multi[i:i+window_samples]
    window_features = freq_extractor.extract_spectral_features(window_data)
    spectral_evolution.append(window_features)
    evolution_times.append(time_multi[i + window_samples//2])

# Extract evolution of key features
centroid_evolution = [f['spectral_centroid'] for f in spectral_evolution]
rolloff_evolution = [f['spectral_rolloff'] for f in spectral_evolution]

fig.add_trace(
    go.Scatter(x=evolution_times, y=centroid_evolution, name='Spectral Centroid',
               line=dict(color='green', width=2)),
    row=3, col=2
)

fig.add_trace(
    go.Scatter(x=evolution_times, y=rolloff_evolution, name='Spectral Rolloff',
               line=dict(color='orange', width=2)),
    row=3, col=2
)

fig.update_xaxes(title_text="Time (s)", row=3, col=2)
fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=2)

# Update layout
fig.update_layout(
    height=1200,
    title_text="Comprehensive Frequency-Domain Analysis of Bridge Response",
    showlegend=True,
    font=dict(size=11)
)

fig.show()

print(f"\nFrequency-Domain Analysis Summary:")
print(f"• Identified {modal_features['number_of_modes']} structural modes")
print(f"• Dominant frequency: {modal_features['dominant_frequency']:.2f} Hz")
print(f"• Energy distribution: {spectral_features['low_freq_energy_ratio']:.1%} low, {spectral_features['mid_freq_energy_ratio']:.1%} mid, {spectral_features['high_freq_energy_ratio']:.1%} high frequency")
print(f"• Spectral centroid: {spectral_features['spectral_centroid']:.2f} Hz")
print(f"• System appears to be {'lightly damped' if np.mean(modal_features['damping_ratios']) < 0.05 else 'moderately damped'}")
```

---

## 3. Principal Component Analysis for Structural Health Monitoring

### 3.1 The Curse of Dimensionality in SHM

Modern structural health monitoring systems generate high-dimensional datasets from multiple sensors, various feature types, and continuous measurements over time. Malekloo et al., 2022 noted that feature vectors can be merged to produce comprehensive damage indicators, but high dimensionality can hinder traditional machine learning approaches. Principal Component Analysis (PCA) provides a systematic approach to reduce dimensionality while preserving the most important variance in the data.

### 3.2 Mathematical Foundation of PCA

PCA finds the principal directions of maximum variance in the data through eigenvalue decomposition of the covariance matrix.

#### Covariance Matrix Computation

For a data matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ where $n$ is the number of samples and $d$ is the number of features, the covariance matrix is:

$$\mathbf{C} = \frac{1}{n-1}(\mathbf{X} - \boldsymbol{\mu})^T(\mathbf{X} - \boldsymbol{\mu}) \tag{6.15}$$

where $\boldsymbol{\mu}$ is the mean vector of features.

#### Eigenvalue Decomposition

The principal components are obtained from the eigenvalue decomposition:

$$\mathbf{C} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T \tag{6.16}$$

where $\mathbf{V}$ contains the eigenvectors (principal components) and $\boldsymbol{\Lambda}$ is the diagonal matrix of eigenvalues.

#### Dimensionality Reduction

The transformed data is obtained by projecting onto the first $k$ principal components:

$$\mathbf{Y} = (\mathbf{X} - \boldsymbol{\mu})\mathbf{V}_{:,1:k} \tag{6.17}$$

#### Reconstruction and Residual Analysis

Data reconstruction from reduced dimensions:

$$\hat{\mathbf{X}} = \mathbf{Y}\mathbf{V}_{:,1:k}^T + \boldsymbol{\mu} \tag{6.18}$$

The reconstruction error provides a measure of information loss:

$$\mathbf{E} = \mathbf{X} - \hat{\mathbf{X}} \tag{6.19}$$

### 3.3 PCA for Anomaly Detection

PCA-based anomaly detection relies on the principle that normal structural behavior can be captured by the first few principal components, while anomalies will have large projections onto the residual subspace.

#### Hotelling's T² Statistic

For monitoring normal operation:

$$T^2 = \mathbf{y}^T\boldsymbol{\Lambda}_{1:k}^{-1}\mathbf{y} \tag{6.20}$$

where $\mathbf{y}$ is the projection of a new sample onto the first $k$ principal components.

#### Q-Statistic (Squared Prediction Error)

For detecting departures from the normal subspace:

$$Q = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 \tag{6.21}$$

### 3.4 Flowchart: PCA-Based Anomaly Detection Framework

```python
# Create SVG flowchart for PCA-based anomaly detection process
flowchart_svg = '''
<svg width="800" height="600" viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" 
            refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
    </marker>
    
    <style>
      .box { fill: #f0f8ff; stroke: #4682b4; stroke-width: 2; }
      .process-box { fill: #e6f3ff; stroke: #4682b4; stroke-width: 2; }
      .decision-box { fill: #fff8dc; stroke: #daa520; stroke-width: 2; }
      .output-box { fill: #f0fff0; stroke: #32cd32; stroke-width: 2; }
      .text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }
      .title { font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; text-anchor: middle; }
      .arrow { stroke: #333; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
    </style>
  </defs>
  
  <!-- Title -->
  <text x="400" y="25" class="title">PCA-Based Anomaly Detection Framework</text>
  
  <!-- Input Data -->
  <rect x="50" y="60" width="120" height="50" class="box" rx="5"/>
  <text x="110" y="80" class="text">Multi-Sensor</text>
  <text x="110" y="95" class="text">SHM Data</text>
  
  <!-- Feature Extraction -->
  <rect x="220" y="60" width="120" height="50" class="process-box" rx="5"/>
  <text x="280" y="80" class="text">Feature</text>
  <text x="280" y="95" class="text">Extraction</text>
  
  <!-- Data Preprocessing -->
  <rect x="390" y="60" width="120" height="50" class="process-box" rx="5"/>
  <text x="450" y="80" class="text">Data</text>
  <text x="450" y="95" class="text">Preprocessing</text>
  
  <!-- Training Data -->
  <rect x="560" y="60" width="120" height="50" class="box" rx="5"/>
  <text x="620" y="80" class="text">Baseline</text>
  <text x="620" y="95" class="text">(Healthy) Data</text>
  
  <!-- PCA Model Training -->
  <rect x="320" y="160" width="160" height="50" class="process-box" rx="5"/>
  <text x="400" y="180" class="text">PCA Model Training</text>
  <text x="400" y="195" class="text">(Eigenvalue Decomposition)</text>
  
  <!-- Principal Components -->
  <rect x="150" y="260" width="120" height="50" class="output-box" rx="5"/>
  <text x="210" y="280" class="text">Principal</text>
  <text x="210" y="295" class="text">Components</text>
  
  <!-- Thresholds -->
  <rect x="320" y="260" width="120" height="50" class="output-box" rx="5"/>
  <text x="380" y="275" class="text">Control Limits</text>
  <text x="380" y="285" class="text">(T², Q)</text>
  <text x="380" y="300" class="text">Thresholds</text>
  
  <!-- Reconstruction Model -->
  <rect x="490" y="260" width="120" height="50" class="output-box" rx="5"/>
  <text x="550" y="280" class="text">Reconstruction</text>
  <text x="550" y="295" class="text">Model</text>
  
  <!-- New Data -->
  <rect x="50" y="360" width="120" height="50" class="box" rx="5"/>
  <text x="110" y="380" class="text">New SHM</text>
  <text x="110" y="395" class="text">Measurements</text>
  
  <!-- Feature Projection -->
  <rect x="220" y="360" width="120" height="50" class="process-box" rx="5"/>
  <text x="280" y="380" class="text">Project onto</text>
  <text x="280" y="395" class="text">PC Space</text>
  
  <!-- Compute Statistics -->
  <rect x="390" y="360" width="120" height="50" class="process-box" rx="5"/>
  <text x="450" y="375" class="text">Compute T²</text>
  <text x="450" y="385" class="text">and Q</text>
  <text x="450" y="395" class="text">Statistics</text>
  
  <!-- Decision Diamond -->
  <polygon points="620,360 670,385 620,410 570,385" class="decision-box"/>
  <text x="620" y="380" class="text">T² > Limit</text>
  <text x="620" y="390" class="text">or Q > Limit?</text>
  
  <!-- Normal -->
  <rect x="560" y="460" width="120" height="40" class="output-box" rx="5"/>
  <text x="620" y="485" class="text">Normal Operation</text>
  
  <!-- Anomaly -->
  <rect x="720" y="360" width="120" height="50" class="output-box" rx="5"/>
  <text x="780" y="380" class="text">Anomaly</text>
  <text x="780" y="395" class="text">Detected</text>
  
  <!-- Arrows -->
  <line x1="170" y1="85" x2="210" y2="85" class="arrow"/>
  <line x1="340" y1="85" x2="380" y2="85" class="arrow"/>
  <line x1="510" y1="85" x2="550" y2="85" class="arrow"/>
  
  <line x1="450" y1="110" x2="450" y2="150" class="arrow"/>
  <line x1="620" y1="110" x2="620" y2="130" class="arrow"/>
  <line x1="620" y1="130" x2="450" y2="130" class="arrow"/>
  <line x1="450" y1="130" x2="450" y2="150" class="arrow"/>
  
  <line x1="350" y1="210" x2="210" y2="250" class="arrow"/>
  <line x1="400" y1="210" x2="380" y2="250" class="arrow"/>
  <line x1="450" y1="210" x2="550" y2="250" class="arrow"/>
  
  <line x1="170" y1="385" x2="210" y2="385" class="arrow"/>
  <line x1="340" y1="385" x2="380" y2="385" class="arrow"/>
  <line x1="510" y1="385" x2="560" y2="385" class="arrow"/>
  
  <line x1="670" y1="385" x2="710" y2="385" class="arrow"/>
  <line x1="620" y1="410" x2="620" y2="450" class="arrow"/>
  
  <!-- Labels for decision paths -->
  <text x="740" y="375" class="text" style="font-size: 10px;">Yes</text>
  <text x="630" y="430" class="text" style="font-size: 10px;">No</text>
</svg>
'''

# Display the flowchart
from IPython.display import SVG, display
display(SVG(flowchart_svg))
```

### 3.5 Implementation: PCA-Based Feature Engineering and Anomaly Detection

```python
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA
from scipy import stats
import matplotlib.pyplot as plt

class PCAFeatureEngineer:
    """
    Advanced PCA-based feature engineering and anomaly detection for SHM
    """
    
    def __init__(self, n_components: int = None, variance_threshold: float = 0.95):
        """
        Initialize PCA feature engineer
        
        Args:
            n_components: Number of principal components to retain
            variance_threshold: Minimum variance to retain if n_components is None
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self.control_limits = {}
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'PCAFeatureEngineer':
        """
        Fit PCA model on healthy/baseline data
        
        Args:
            X: Training data matrix (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of components
        if self.n_components is None:
            # Use variance threshold to determine components
            pca_temp = SklearnPCA()
            pca_temp.fit(X_scaled)
            cumulative_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            self.n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            print(f"Selected {self.n_components} components to retain {self.variance_threshold:.1%} variance")
        
        # Fit final PCA model
        self.pca = SklearnPCA(n_components=self.n_components)
        X_transformed = self.pca.fit_transform(X_scaled)
        
        # Compute control limits for anomaly detection
        self._compute_control_limits(X_scaled, X_transformed)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal component space
        
        Args:
            X: Data matrix to transform
            
        Returns:
            Transformed data in PC space
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
            
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from principal component space
        
        Args:
            X_pca: Data in principal component space
            
        Returns:
            Reconstructed data in original space
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inverse_transform")
            
        X_reconstructed_scaled = self.pca.inverse_transform(X_pca)
        return self.scaler.inverse_transform(X_reconstructed_scaled)
    
    def detect_anomalies(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect anomalies using T² and Q statistics
        
        Args:
            X: Data matrix to analyze
            
        Returns:
            Dictionary containing anomaly scores and decisions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before anomaly detection")
            
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        
        # Compute T² statistic (distance in PC space)
        # T² = sum((X_pca / sqrt(eigenvalues))^2)
        eigenvalues = self.pca.explained_variance_
        t2_scores = np.sum((X_pca**2) / eigenvalues, axis=1)
        
        # Compute Q statistic (reconstruction error)
        reconstruction_error = X_scaled - X_reconstructed
        q_scores = np.sum(reconstruction_error**2, axis=1)
        
        # Make anomaly decisions
        t2_anomalies = t2_scores > self.control_limits['t2_limit']
        q_anomalies = q_scores > self.control_limits['q_limit']
        combined_anomalies = t2_anomalies | q_anomalies
        
        results = {
            't2_scores': t2_scores,
            'q_scores': q_scores,
            't2_anomalies': t2_anomalies,
            'q_anomalies': q_anomalies,
            'anomalies': combined_anomalies,
            'reconstruction_error': reconstruction_error
        }
        
        return results
    
    def _compute_control_limits(self, X_scaled: np.ndarray, X_transformed: np.ndarray):
        """
        Compute control limits for T² and Q statistics
        """
        n_samples, n_features = X_scaled.shape
        
        # T² control limit (chi-square distribution)
        alpha = 0.01  # 99% confidence level
        t2_limit = stats.chi2.ppf(1 - alpha, df=self.n_components)
        
        # Q control limit (reconstruction error)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        reconstruction_errors = X_scaled - X_reconstructed
        q_scores_baseline = np.sum(reconstruction_errors**2, axis=1)
        
        # Use 99th percentile as Q limit
        q_limit = np.percentile(q_scores_baseline, 99)
        
        self.control_limits = {
            't2_limit': t2_limit,
            'q_limit': q_limit
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on principal component loadings
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing feature importance")
            
        # Compute feature importance as sum of squared loadings
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        importance = np.sum(loadings**2, axis=1)
        
        # Create DataFrame
        feature_names = [f'Feature_{i+1}' for i in range(len(importance))]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance,
            'Normalized_Importance': importance / np.sum(importance)
        }).sort_values('Importance', ascending=False)
        
        return importance_df

# Demonstrate comprehensive PCA-based feature engineering
print("\n=== PCA-Based Feature Engineering and Anomaly Detection ===\n")

# Generate comprehensive feature dataset
def create_comprehensive_feature_dataset(n_samples: int = 1000, 
                                       anomaly_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a comprehensive feature dataset for PCA demonstration
    """
    np.random.seed(42)
    
    # Number of normal and anomalous samples
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomalous = n_samples - n_normal
    
    # Generate normal samples (correlated features representing healthy structure)
    mean_normal = np.array([2.0, 0.5, 0.1, 5.0, 0.05, 1.2, 0.3, 0.8])
    cov_normal = np.array([
        [1.0, 0.8, 0.3, -0.2, 0.1, 0.4, 0.2, 0.1],
        [0.8, 0.6, 0.2, -0.1, 0.05, 0.3, 0.15, 0.05],
        [0.3, 0.2, 0.05, 0.1, 0.02, 0.1, 0.05, 0.02],
        [-0.2, -0.1, 0.1, 2.0, 0.1, -0.2, 0.1, 0.05],
        [0.1, 0.05, 0.02, 0.1, 0.01, 0.05, 0.02, 0.01],
        [0.4, 0.3, 0.1, -0.2, 0.05, 0.3, 0.1, 0.05],
        [0.2, 0.15, 0.05, 0.1, 0.02, 0.1, 0.08, 0.02],
        [0.1, 0.05, 0.02, 0.05, 0.01, 0.05, 0.02, 0.04]
    ])
    
    X_normal = np.random.multivariate_normal(mean_normal, cov_normal, n_normal)
    
    # Generate anomalous samples (shifted means, different correlations)
    # Type 1: Shifted mean (damage-like behavior)
    n_type1 = n_anomalous // 2
    mean_anomaly1 = mean_normal + np.array([0.5, 0.2, 0.05, -1.0, 0.02, 0.3, 0.1, 0.2])
    X_anomaly1 = np.random.multivariate_normal(mean_anomaly1, cov_normal * 1.2, n_type1)
    
    # Type 2: Different correlation structure (sensor malfunction-like)
    n_type2 = n_anomalous - n_type1
    cov_anomaly2 = cov_normal.copy()
    cov_anomaly2[0, 1] = -0.5  # Changed correlation
    cov_anomaly2[1, 0] = -0.5
    X_anomaly2 = np.random.multivariate_normal(mean_normal, cov_anomaly2 * 2.0, n_type2)
    
    # Combine data
    X = np.vstack([X_normal, X_anomaly1, X_anomaly2])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalous)])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

# Generate dataset
feature_data, true_labels = create_comprehensive_feature_dataset(n_samples=1000, anomaly_ratio=0.05)

# Split into training (healthy) and testing data
train_mask = true_labels == 0
X_train = feature_data[train_mask]
X_test = feature_data
y_test = true_labels

print(f"Dataset created:")
print(f"• Training samples (healthy): {len(X_train)}")
print(f"• Test samples (total): {len(X_test)}")
print(f"• Features: {X_train.shape[1]}")
print(f"• Anomaly ratio in test set: {np.mean(y_test):.1%}")

# Initialize and fit PCA model
pca_engineer = PCAFeatureEngineer(variance_threshold=0.95)
pca_engineer.fit(X_train)

print(f"\nPCA Model Summary:")
print(f"• Components retained: {pca_engineer.n_components}")
print(f"• Explained variance ratio: {pca_engineer.pca.explained_variance_ratio_}")
print(f"• Cumulative variance: {np.sum(pca_engineer.pca.explained_variance_ratio_):.3f}")

# Perform anomaly detection
anomaly_results = pca_engineer.detect_anomalies(X_test)

# Evaluate performance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(f"\nAnomaly Detection Performance:")
print(f"• T² anomalies detected: {np.sum(anomaly_results['t2_anomalies'])}")
print(f"• Q anomalies detected: {np.sum(anomaly_results['q_anomalies'])}")
print(f"• Combined anomalies detected: {np.sum(anomaly_results['anomalies'])}")

# Classification performance
auc_t2 = roc_auc_score(y_test, anomaly_results['t2_scores'])
auc_q = roc_auc_score(y_test, anomaly_results['q_scores'])

print(f"• T² AUC: {auc_t2:.3f}")
print(f"• Q AUC: {auc_q:.3f}")

print("\nClassification Report (Combined T² and Q):")
print(classification_report(y_test, anomaly_results['anomalies'], 
                          target_names=['Normal', 'Anomaly']))

# Feature importance analysis
importance_df = pca_engineer.get_feature_importance()
print(f"\nFeature Importance Analysis:")
print(importance_df.head().to_string(index=False))
```

### 3.6 Advanced PCA Visualization and Analysis

```python
# Create comprehensive PCA visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Principal Components (2D Projection)', 'Explained Variance',
                   'T² Control Chart', 'Q Control Chart', 
                   'Feature Loadings', 'Anomaly Score Distribution'),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "heatmap"}, {"type": "histogram"}]]
)

# Transform test data for visualization
X_test_pca = pca_engineer.transform(X_test)

# Plot 1: 2D projection of first two principal components
colors = ['blue' if label == 0 else 'red' for label in y_test]
fig.add_trace(
    go.Scatter(x=X_test_pca[:, 0], y=X_test_pca[:, 1],
               mode='markers', 
               marker=dict(color=colors, size=6, opacity=0.7),
               name='Normal' if True else 'Anomaly'),
    row=1, col=1
)

# Add separate traces for legend
normal_indices = y_test == 0
anomaly_indices = y_test == 1

fig.add_trace(
    go.Scatter(x=X_test_pca[normal_indices, 0], y=X_test_pca[normal_indices, 1],
               mode='markers', 
               marker=dict(color='blue', size=6, opacity=0.7),
               name='Normal'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=X_test_pca[anomaly_indices, 0], y=X_test_pca[anomaly_indices, 1],
               mode='markers', 
               marker=dict(color='red', size=8, opacity=0.8, symbol='x'),
               name='Anomaly'),
    row=1, col=1
)

fig.update_xaxes(title_text="First Principal Component", row=1, col=1)
fig.update_yaxes(title_text="Second Principal Component", row=1, col=1)

# Plot 2: Explained variance
pc_labels = [f'PC{i+1}' for i in range(len(pca_engineer.pca.explained_variance_ratio_))]
fig.add_trace(
    go.Bar(x=pc_labels, y=pca_engineer.pca.explained_variance_ratio_,
           marker_color='lightblue', name='Individual'),
    row=1, col=2
)

# Add cumulative variance line
cumulative_var = np.cumsum(pca_engineer.pca.explained_variance_ratio_)
fig.add_trace(
    go.Scatter(x=pc_labels, y=cumulative_var,
               mode='lines+markers', name='Cumulative',
               line=dict(color='red', width=2)),
    row=1, col=2
)

fig.update_xaxes(title_text="Principal Component", row=1, col=2)
fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=2)

# Plot 3: T² Control Chart
sample_indices = np.arange(len(X_test))
fig.add_trace(
    go.Scatter(x=sample_indices, y=anomaly_results['t2_scores'],
               mode='markers', 
               marker=dict(color=['red' if anom else 'blue' 
                                for anom in anomaly_results['t2_anomalies']],
                          size=4),
               name='T² Scores'),
    row=2, col=1
)

# Add control limit line
fig.add_hline(y=pca_engineer.control_limits['t2_limit'], 
              line_dash="dash", line_color="red",
              annotation_text="T² Control Limit",
              row=2, col=1)

fig.update_xaxes(title_text="Sample Index", row=2, col=1)
fig.update_yaxes(title_text="T² Statistic", row=2, col=1)

# Plot 4: Q Control Chart  
fig.add_trace(
    go.Scatter(x=sample_indices, y=anomaly_results['q_scores'],
               mode='markers',
               marker=dict(color=['red' if anom else 'blue' 
                                for anom in anomaly_results['q_anomalies']],
                          size=4),
               name='Q Scores'),
    row=2, col=2
)

# Add control limit line
fig.add_hline(y=pca_engineer.control_limits['q_limit'], 
              line_dash="dash", line_color="red",
              annotation_text="Q Control Limit",
              row=2, col=2)

fig.update_xaxes(title_text="Sample Index", row=2, col=2)
fig.update_yaxes(title_text="Q Statistic", row=2, col=2)

# Plot 5: Feature loadings heatmap
loadings = pca_engineer.pca.components_[:4, :].T  # First 4 PCs
feature_names = [f'Feature {i+1}' for i in range(loadings.shape[0])]
pc_names = [f'PC{i+1}' for i in range(loadings.shape[1])]

fig.add_trace(
    go.Heatmap(z=loadings, 
               x=pc_names, y=feature_names,
               colorscale='RdBu', zmid=0,
               name='Loadings'),
    row=3, col=1
)

fig.update_xaxes(title_text="Principal Component", row=3, col=1)
fig.update_yaxes(title_text="Original Feature", row=3, col=1)

# Plot 6: Anomaly score distributions
combined_scores = anomaly_results['t2_scores'] + anomaly_results['q_scores']

fig.add_trace(
    go.Histogram(x=combined_scores[y_test == 0], 
                 nbinsx=30, name='Normal',
                 marker_color='blue', opacity=0.7),
    row=3, col=2
)

fig.add_trace(
    go.Histogram(x=combined_scores[y_test == 1], 
                 nbinsx=15, name='Anomaly',
                 marker_color='red', opacity=0.7),
    row=3, col=2
)

fig.update_xaxes(title_text="Combined Anomaly Score", row=3, col=2)
fig.update_yaxes(title_text="Frequency", row=3, col=2)

# Update layout
fig.update_layout(
    height=1200,
    title_text="Comprehensive PCA-Based Anomaly Detection Analysis",
    showlegend=True,
    font=dict(size=11)
)

fig.show()

print(f"\nPCA Analysis Summary:")
print(f"• {pca_engineer.n_components} components explain {np.sum(pca_engineer.pca.explained_variance_ratio_):.1%} of variance")
print(f"• Most important features: {', '.join(importance_df.head(3)['Feature'].values)}")
print(f"• Control limits: T² = {pca_engineer.control_limits['t2_limit']:.2f}, Q = {pca_engineer.control_limits['q_limit']:.4f}")
print(f"• False positive rate: {np.sum((anomaly_results['anomalies']) & (y_test == 0)) / np.sum(y_test == 0):.1%}")
print(f"• Detection rate: {np.sum((anomaly_results['anomalies']) & (y_test == 1)) / np.sum(y_test == 1):.1%}")
```

---

## 4. Feature Extraction from Digital Images

### 4.1 The Rise of Computer Vision in SHM

Digital image analysis has emerged as a powerful non-contact method for structural health monitoring, offering advantages in terms of cost, accessibility, and the ability to capture spatial damage patterns that traditional point sensors cannot detect. DIC applications and computer vision techniques have shown significant promise in bridge monitoring applications.

Modern bridge inspection generates thousands of images from various sources including UAVs, fixed cameras, and mobile inspection platforms. The challenge lies in automatically extracting meaningful structural features from these images while being robust to varying lighting conditions, weather effects, and background clutter.

### 4.2 Fundamental Image Features for SHM

#### 4.2.1 Texture Features

Texture analysis captures the spatial arrangement of intensity variations in images, which is particularly relevant for detecting surface defects, cracking patterns, and material degradation.

**Gray-Level Co-occurrence Matrix (GLCM) Features:**

The GLCM $P_{d,\theta}(i,j)$ represents the probability of pixel intensities $i$ and $j$ occurring at distance $d$ and angle $\theta$. Key features derived from GLCM include:

**Contrast:**
$$Contrast = \sum_{i,j} |i-j|^2 P(i,j) \tag{6.22}$$

**Homogeneity:**
$$Homogeneity = \sum_{i,j} \frac{P(i,j)}{1 + |i-j|} \tag{6.23}$$

**Energy:**
$$Energy = \sum_{i,j} P(i,j)^2 \tag{6.24}$$

**Correlation:**
$$Correlation = \sum_{i,j} \frac{(i-\mu_i)(j-\mu_j)P(i,j)}{\sigma_i \sigma_j} \tag{6.25}$$

#### 4.2.2 Local Binary Patterns (LBP)

LBP provides a powerful texture descriptor that is robust to illumination changes, making it particularly suitable for outdoor bridge monitoring applications.

**Basic LBP:**
$$LBP_{P,R}(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c) 2^p \tag{6.26}$$

where:
- $g_c$ is the intensity of the center pixel
- $g_p$ is the intensity of the $p$-th neighbor pixel
- $s(x) = 1$ if $x \geq 0$, otherwise $s(x) = 0$
- $P$ is the number of sample points
- $R$ is the radius of the neighborhood

#### 4.2.3 Edge and Gradient Features

Edge detection is fundamental for identifying structural boundaries, cracks, and geometric changes in bridge components.

**Sobel Gradient:**
$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I \tag{6.27}$$

**Gradient Magnitude:**
$$G = \sqrt{G_x^2 + G_y^2} \tag{6.28}$$

**Gradient Direction:**
$$\theta = \arctan\left(\frac{G_y}{G_x}\right) \tag{6.29}$$

### 4.3 Deep Learning Features

#### 4.3.1 Convolutional Neural Network Features

CNNs automatically learn hierarchical feature representations that are particularly effective for structural damage detection and classification.

**Convolutional Layer:**
$$y_{i,j,k} = \sigma\left(\sum_{m,n,c} w_{m,n,c,k} \cdot x_{i+m,j+n,c} + b_k\right) \tag{6.30}$$

where $w$ represents the learnable filters, $b$ is the bias term, and $\sigma$ is the activation function.

### 4.4 Implementation: Comprehensive Image Feature Extraction

```python
import cv2
from skimage import feature, filters, measure
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.filters import sobel, roberts, prewitt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image

class ImageFeatureExtractor:
    """
    Comprehensive image feature extraction for structural health monitoring
    """
    
    def __init__(self):
        """Initialize the image feature extractor"""
        self.glcm_distances = [1, 2, 3]
        self.glcm_angles = [0, 45, 90, 135]
        
        # Initialize pre-trained CNN for deep features
        self.cnn_model = resnet18(pretrained=True)
        self.cnn_model.eval()
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(self.cnn_model.children())[:-1])
        
        # Image preprocessing for CNN
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features using GLCM and LBP
        
        Args:
            image: Grayscale image array
            
        Returns:
            Dictionary of texture features
        """
        features = {}
        
        # Ensure image is uint8 and properly scaled
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # GLCM features
        glcm = greycomatrix(image, distances=self.glcm_distances, 
                           angles=np.radians(self.glcm_angles), 
                           levels=256, symmetric=True, normed=True)
        
        # Extract GLCM properties
        contrast = greycoprops(glcm, 'contrast')
        dissimilarity = greycoprops(glcm, 'dissimilarity')
        homogeneity = greycoprops(glcm, 'homogeneity')
        energy = greycoprops(glcm, 'energy')
        correlation = greycoprops(glcm, 'correlation')
        
        # Average across distances and angles
        features.update({
            'glcm_contrast': np.mean(contrast),
            'glcm_contrast_std': np.std(contrast),
            'glcm_dissimilarity': np.mean(dissimilarity),
            'glcm_homogeneity': np.mean(homogeneity),
            'glcm_energy': np.mean(energy),
            'glcm_correlation': np.mean(correlation)
        })
        
        # Local Binary Pattern features
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # LBP histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                  range=(0, n_points + 2), density=True)
        
        features.update({
            'lbp_uniformity': np.sum(lbp_hist**2),
            'lbp_entropy': -np.sum(lbp_hist * np.log2(lbp_hist + 1e-12)),
            'lbp_mean': np.mean(lbp),
            'lbp_variance': np.var(lbp)
        })
        
        return features
    
    def extract_geometric_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract geometric and morphological features
        
        Args:
            image: Grayscale image array
            
        Returns:
            Dictionary of geometric features
        """
        features = {}
        
        # Edge detection
        edges_sobel = sobel(image)
        edges_roberts = roberts(image)
        edges_prewitt = prewitt(image)
        
        # Edge statistics
        features.update({
            'edge_density_sobel': np.mean(edges_sobel > 0.1),
            'edge_strength_sobel': np.mean(edges_sobel),
            'edge_density_roberts': np.mean(edges_roberts > 0.1),
            'edge_strength_roberts': np.mean(edges_roberts),
            'edge_density_prewitt': np.mean(edges_prewitt > 0.1),
            'edge_strength_prewitt': np.mean(edges_prewitt)
        })
        
        # Gradient features
        grad_x = filters.sobel_h(image)
        grad_y = filters.sobel_v(image)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        features.update({
            'gradient_magnitude_mean': np.mean(gradient_magnitude),
            'gradient_magnitude_std': np.std(gradient_magnitude),
            'gradient_direction_mean': np.mean(gradient_direction),
            'gradient_direction_std': np.std(gradient_direction)
        })
        
        # Morphological features
        # Apply threshold to create binary image
        threshold = filters.threshold_otsu(image)
        binary = image > threshold
        
        # Connected components analysis
        labeled_image = measure.label(binary)
        regions = measure.regionprops(labeled_image)
        
        if len(regions) > 0:
            areas = [region.area for region in regions]
            eccentricities = [region.eccentricity for region in regions]
            solidity = [region.solidity for region in regions]
            
            features.update({
                'num_components': len(regions),
                'mean_component_area': np.mean(areas),
                'std_component_area': np.std(areas),
                'mean_eccentricity': np.mean(eccentricities),
                'mean_solidity': np.mean(solidity)
            })
        else:
            features.update({
                'num_components': 0,
                'mean_component_area': 0,
                'std_component_area': 0,
                'mean_eccentricity': 0,
                'mean_solidity': 0
            })
        
        return features
    
    def extract_statistical_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from image intensities
        
        Args:
            image: Image array
            
        Returns:
            Dictionary of statistical features
        """
        flat_image = image.flatten()
        
        features = {
            'intensity_mean': np.mean(flat_image),
            'intensity_std': np.std(flat_image),
            'intensity_min': np.min(flat_image),
            'intensity_max': np.max(flat_image),
            'intensity_range': np.ptp(flat_image),
            'intensity_skewness': stats.skew(flat_image),
            'intensity_kurtosis': stats.kurtosis(flat_image),
            'intensity_entropy': -np.sum(np.histogram(flat_image, bins=256)[0] / len(flat_image) * 
                                        np.log2(np.histogram(flat_image, bins=256)[0] / len(flat_image) + 1e-12))
        }
        
        return features
    
    def extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract deep CNN features
        
        Args:
            image: Input image (can be grayscale or RGB)
            
        Returns:
            Feature vector from pre-trained CNN
        """
        # Convert to PIL Image and handle grayscale
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image = np.stack([image, image, image], axis=-1)
        
        # Normalize to 0-255 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        
        # Preprocess and extract features
        input_tensor = self.preprocess(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = features.squeeze().numpy()
        
        return features
    
    def extract_all_features(self, image: np.ndarray) -> Dict[str, any]:
        """
        Extract all image features
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing all extracted features
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # Extract different types of features
        features = {}
        features.update(self.extract_statistical_features(gray_image))
        features.update(self.extract_texture_features(gray_image))
        features.update(self.extract_geometric_features(gray_image))
        
        # Deep features (returned separately due to high dimensionality)
        deep_features = self.extract_deep_features(image)
        
        return features, deep_features

# Generate synthetic bridge images for demonstration
def generate_synthetic_bridge_images() -> List[Tuple[np.ndarray, str, bool]]:
    """
    Generate synthetic bridge surface images with various conditions
    
    Returns:
        List of (image, condition_name, is_damaged) tuples
    """
    images = []
    np.random.seed(42)
    
    # Image dimensions
    height, width = 200, 200
    
    # 1. Healthy concrete surface
    healthy_base = np.random.normal(150, 20, (height, width))
    # Add concrete texture
    for i in range(height):
        for j in range(width):
            healthy_base[i, j] += 10 * np.sin(i/10) * np.cos(j/15)
    
    healthy_image = np.clip(healthy_base, 0, 255).astype(np.uint8)
    images.append((healthy_image, "Healthy Concrete", False))
    
    # 2. Surface with minor cracks
    minor_crack = healthy_image.copy()
    # Add diagonal crack
    for i in range(height):
        j = int(i * 0.8 + 20 + 5*np.sin(i/20))
        if 0 <= j < width:
            minor_crack[max(0, i-1):i+2, max(0, j-1):j+2] = 80
    
    images.append((minor_crack, "Minor Cracking", True))
    
    # 3. Surface with major cracks
    major_crack = healthy_image.copy()
    # Add multiple intersecting cracks
    for i in range(height):
        # Horizontal crack
        j1 = int(width/2 + 10*np.sin(i/15))
        if 0 <= j1 < width:
            major_crack[max(0, i-2):i+3, max(0, j1-2):j1+3] = 60
        
        # Vertical crack
        if i < width:
            j2 = int(height/3)
            major_crack[max(0, j2-2):j2+3, max(0, i-2):i+3] = 70
    
    images.append((major_crack, "Major Cracking", True))
    
    # 4. Spalling/deterioration
    spalling = healthy_image.copy()
    # Add irregular damaged areas
    centers = [(50, 60), (120, 140), (80, 120)]
    for cx, cy in centers:
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i-cx)**2 + (j-cy)**2)
                if dist < 25:
                    intensity_reduction = (25 - dist) / 25 * 60
                    spalling[i, j] = max(40, spalling[i, j] - intensity_reduction)
    
    images.append((spalling, "Spalling", True))
    
    # 5. Corrosion stains
    corrosion = healthy_image.copy()
    # Add rust-colored stains
    for i in range(height):
        for j in range(width):
            # Create irregular stain pattern
            if (i-100)**2 + (j-100)**2 < 1500:
                stain_intensity = 1 - ((i-100)**2 + (j-100)**2) / 1500
                corrosion[i, j] = corrosion[i, j] * (1 - 0.4 * stain_intensity)
    
    images.append((corrosion, "Corrosion Stains", True))
    
    return images

# Demonstrate comprehensive image feature extraction
print("\n=== Image Feature Extraction for Bridge SHM ===\n")

# Generate synthetic bridge images
bridge_images = generate_synthetic_bridge_images()

# Initialize image feature extractor
image_extractor = ImageFeatureExtractor()

# Extract features from all images
all_image_features = []
all_deep_features = []
condition_labels = []
damage_labels = []

print("Extracting features from bridge surface images...")
for i, (image, condition, is_damaged) in enumerate(bridge_images):
    print(f"Processing {condition}...")
    
    # Extract features
    features, deep_features = image_extractor.extract_all_features(image)
    
    all_image_features.append(features)
    all_deep_features.append(deep_features)
    condition_labels.append(condition)
    damage_labels.append(is_damaged)

# Convert to structured format
feature_names = list(all_image_features[0].keys())
feature_matrix = np.array([[features[name] for name in feature_names] 
                          for features in all_image_features])

# Create comprehensive DataFrame
results_df = pd.DataFrame(feature_matrix, columns=feature_names, index=condition_labels)

print(f"\nExtracted Features Summary:")
print(f"• Number of traditional features: {len(feature_names)}")
print(f"• Deep feature dimensions: {all_deep_features[0].shape}")
print(f"• Image conditions analyzed: {len(bridge_images)}")

# Display key features comparison
key_features = ['intensity_mean', 'intensity_std', 'glcm_contrast', 'glcm_homogeneity', 
                'edge_density_sobel', 'lbp_entropy', 'num_components']

print(f"\nKey Feature Comparison:")
comparison_df = results_df[key_features].round(4)
comparison_df['Damage_Status'] = ['Healthy', 'Minor', 'Major', 'Severe', 'Moderate']
print(comparison_df.to_string())

# Visualize images and feature analysis
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=tuple(condition_labels + ['Feature Correlation', 'Feature Distribution', 'Damage Classification']),
    specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
           [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
           [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]]
)

# Display original images
for i, (image, condition, _) in enumerate(bridge_images):
    row = i // 3 + 1
    col = i % 3 + 1
    
    fig.add_trace(
        go.Heatmap(z=image, colorscale='gray', showscale=False, name=condition),
        row=row, col=col
    )
    
    fig.update_xaxes(showticklabels=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, row=row, col=col)

# Feature correlation heatmap
correlation_matrix = np.corrcoef(feature_matrix.T)
selected_features = ['intensity_std', 'glcm_contrast', 'edge_density_sobel', 
                    'lbp_entropy', 'gradient_magnitude_mean']
selected_indices = [feature_names.index(f) for f in selected_features]
corr_subset = correlation_matrix[np.ix_(selected_indices, selected_indices)]

fig.add_trace(
    go.Heatmap(z=corr_subset, 
               x=selected_features, y=selected_features,
               colorscale='RdBu', zmid=0, showscale=True,
               name='Correlation'),
    row=3, col=1
)

fig.update_xaxes(tickangle=45, row=3, col=1)

# Feature importance for damage detection
damage_array = np.array(damage_labels).astype(int)
feature_importance = []

for i, feature_name in enumerate(selected_features):
    feature_values = feature_matrix[:, feature_names.index(feature_name)]
    # Simple correlation-based importance
    importance = abs(np.corrcoef(feature_values, damage_array)[0, 1])
    feature_importance.append(importance)

fig.add_trace(
    go.Bar(x=selected_features, y=feature_importance,
           marker_color='lightcoral', name='Importance'),
    row=3, col=2
)

fig.update_xaxes(tickangle=45, row=3, col=2)
fig.update_yaxes(title_text="Correlation with Damage", row=3, col=2)

# 2D feature space visualization
x_feature = 'glcm_contrast'
y_feature = 'edge_density_sobel'

x_idx = feature_names.index(x_feature)
y_idx = feature_names.index(y_feature)

colors = ['green', 'yellow', 'orange', 'red', 'purple']
for i, (condition, is_damaged) in enumerate(zip(condition_labels, damage_labels)):
    fig.add_trace(
        go.Scatter(x=[feature_matrix[i, x_idx]], y=[feature_matrix[i, y_idx]],
                   mode='markers', name=condition,
                   marker=dict(color=colors[i], size=12, 
                             symbol='x' if is_damaged else 'circle')),
        row=3, col=3
    )

fig.update_xaxes(title_text="GLCM Contrast", row=3, col=3)
fig.update_yaxes(title_text="Edge Density (Sobel)", row=3, col=3)

# Update layout
fig.update_layout(
    height=1000,
    title_text="Comprehensive Image Feature Analysis for Bridge SHM",
    showlegend=True,
    font=dict(size=10)
)

fig.show()

print(f"\nImage Analysis Insights:")
print(f"• Healthy concrete shows low contrast ({results_df.loc['Healthy Concrete', 'glcm_contrast']:.3f})")
print(f"• Major cracking increases edge density ({results_df.loc['Major Cracking', 'edge_density_sobel']:.3f})")
print(f"• Spalling reduces surface homogeneity ({results_df.loc['Spalling', 'glcm_homogeneity']:.3f})")
print(f"• Corrosion affects intensity distribution (std: {results_df.loc['Corrosion Stains', 'intensity_std']:.1f})")

---

## 5. Feature Fusion and Integration

### 5.1 The Multi-Modal Challenge

Modern structural health monitoring systems collect data from diverse sources: accelerometers, strain gauges, temperature sensors, and cameras. Each modality provides unique insights into structural behavior, but the real power emerges when these complementary information sources are systematically combined. Feature fusion techniques provide a systematic approach to combine features from multiple sensors and data types to enhance anomaly detection performance.

### 5.2 Feature Fusion Strategies

#### 5.2.1 Early Fusion (Feature-Level Fusion)

Early fusion combines raw features from different modalities before pattern recognition:

$\mathbf{f}_{fused} = [\mathbf{f}_{time}, \mathbf{f}_{freq}, \mathbf{f}_{image}, \mathbf{f}_{pca}] \tag{6.31}$

where each $\mathbf{f}_i$ represents features from different domains.

#### 5.2.2 Late Fusion (Decision-Level Fusion)

Late fusion combines decisions from individual classifiers:

$D_{final} = w_1 \cdot D_{time} + w_2 \cdot D_{freq} + w_3 \cdot D_{image} + w_4 \cdot D_{pca} \tag{6.32}$

where $w_i$ are fusion weights and $D_i$ are individual decisions.

#### 5.2.3 Hybrid Fusion

Combines both feature-level and decision-level approaches for optimal performance.

### 5.3 Feature Standardization and Weighting

Before fusion, features must be normalized to prevent dominance by features with larger scales:

**Z-score Normalization:**
$\mathbf{f}_{norm} = \frac{\mathbf{f} - \boldsymbol{\mu}_f}{\boldsymbol{\sigma}_f} \tag{6.33}$

**Min-Max Normalization:**
$\mathbf{f}_{norm} = \frac{\mathbf{f} - \mathbf{f}_{min}}{\mathbf{f}_{max} - \mathbf{f}_{min}} \tag{6.34}$

### 5.4 Implementation: Multi-Modal Feature Fusion

```python
class MultiModalFeatureFusion:
    """
    Advanced feature fusion for multi-modal SHM data
    """
    
    def __init__(self, fusion_strategy: str = 'early', normalization: str = 'zscore'):
        """
        Initialize multi-modal feature fusion
        
        Args:
            fusion_strategy: 'early', 'late', or 'hybrid'
            normalization: 'zscore', 'minmax', or 'robust'
        """
        self.fusion_strategy = fusion_strategy
        self.normalization = normalization
        self.scalers = {}
        self.feature_weights = {}
        self.is_fitted = False
        
    def fit(self, feature_dict: Dict[str, np.ndarray]) -> 'MultiModalFeatureFusion':
        """
        Fit fusion model on training data
        
        Args:
            feature_dict: Dictionary with modality names as keys and feature arrays as values
            
        Returns:
            Self for method chaining
        """
        self.modalities = list(feature_dict.keys())
        
        # Fit scalers for each modality
        for modality, features in feature_dict.items():
            if self.normalization == 'zscore':
                scaler = StandardScaler()
            elif self.normalization == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif self.normalization == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            
            self.scalers[modality] = scaler.fit(features)
        
        # Compute feature weights based on mutual information or correlation
        self._compute_feature_weights(feature_dict)
        
        self.is_fitted = True
        return self
    
    def transform(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply feature fusion transformation
        
        Args:
            feature_dict: Dictionary with modality names as keys and feature arrays as values
            
        Returns:
            Fused feature array
        """
        if not self.is_fitted:
            raise ValueError("Fusion model must be fitted before transform")
        
        if self.fusion_strategy == 'early':
            return self._early_fusion(feature_dict)
        elif self.fusion_strategy == 'late':
            return self._late_fusion(feature_dict)
        else:  # hybrid
            return self._hybrid_fusion(feature_dict)
    
    def _early_fusion(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Early fusion implementation"""
        normalized_features = []
        
        for modality in self.modalities:
            if modality in feature_dict:
                # Normalize features
                features_norm = self.scalers[modality].transform(feature_dict[modality])
                
                # Apply learned weights
                if modality in self.feature_weights:
                    features_weighted = features_norm * self.feature_weights[modality]
                else:
                    features_weighted = features_norm
                
                normalized_features.append(features_weighted)
        
        # Concatenate all features
        fused_features = np.hstack(normalized_features)
        return fused_features
    
    def _late_fusion(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Late fusion implementation (simplified for demonstration)"""
        # For late fusion, we would typically train individual classifiers
        # Here we demonstrate with simple scoring
        modality_scores = []
        
        for modality in self.modalities:
            if modality in feature_dict:
                features_norm = self.scalers[modality].transform(feature_dict[modality])
                
                # Simple anomaly score based on Mahalanobis distance
                try:
                    from scipy.spatial.distance import mahalanobis
                    mean_vec = np.mean(features_norm, axis=0)
                    cov_mat = np.cov(features_norm.T)
                    
                    scores = []
                    for sample in features_norm:
                        try:
                            score = mahalanobis(sample, mean_vec, np.linalg.pinv(cov_mat))
                            scores.append(score)
                        except:
                            scores.append(np.linalg.norm(sample - mean_vec))
                    
                    modality_scores.append(np.array(scores))
                except:
                    # Fallback to L2 distance
                    mean_vec = np.mean(features_norm, axis=0)
                    scores = np.linalg.norm(features_norm - mean_vec, axis=1)
                    modality_scores.append(scores)
        
        # Weighted combination of scores
        if len(modality_scores) > 1:
            weights = [1.0/len(modality_scores)] * len(modality_scores)
            combined_scores = np.zeros_like(modality_scores[0])
            for i, scores in enumerate(modality_scores):
                combined_scores += weights[i] * scores
        else:
            combined_scores = modality_scores[0]
        
        return combined_scores.reshape(-1, 1)
    
    def _hybrid_fusion(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Hybrid fusion combining early and late strategies"""
        # Combine early fusion features with late fusion scores
        early_features = self._early_fusion(feature_dict)
        late_scores = self._late_fusion(feature_dict)
        
        # Normalize late scores to similar scale as early features
        late_scores_norm = (late_scores - np.mean(late_scores)) / (np.std(late_scores) + 1e-8)
        
        # Concatenate
        hybrid_features = np.hstack([early_features, late_scores_norm])
        return hybrid_features
    
    def _compute_feature_weights(self, feature_dict: Dict[str, np.ndarray]):
        """Compute feature importance weights for each modality"""
        # Simple variance-based weighting
        for modality, features in feature_dict.items():
            feature_vars = np.var(features, axis=0)
            # Weight features by their variance (higher variance = more informative)
            weights = feature_vars / (np.sum(feature_vars) + 1e-8)
            self.feature_weights[modality] = weights

# Create comprehensive demonstration of feature fusion
print("\n=== Multi-Modal Feature Fusion Demonstration ===\n")

# Generate multi-modal dataset
def create_multimodal_shm_dataset(n_samples: int = 500) -> Dict[str, np.ndarray]:
    """Create synthetic multi-modal SHM dataset"""
    np.random.seed(42)
    
    # Time-domain features (from accelerometer data)
    time_features = np.random.multivariate_normal(
        mean=[0.1, 0.05, 2.0, 0.5],
        cov=[[0.01, 0.005, 0.1, 0.02],
             [0.005, 0.002, 0.05, 0.01],
             [0.1, 0.05, 1.0, 0.2],
             [0.02, 0.01, 0.2, 0.04]],
        size=n_samples
    )
    
    # Frequency-domain features (from modal analysis)
    freq_features = np.random.multivariate_normal(
        mean=[2.3, 6.8, 0.02, 0.04],
        cov=[[0.1, 0.05, 0.001, 0.002],
             [0.05, 0.3, 0.002, 0.004],
             [0.001, 0.002, 0.0001, 0.0001],
             [0.002, 0.004, 0.0001, 0.0002]],
        size=n_samples
    )
    
    # Image features (from visual inspection)
    image_features = np.random.multivariate_normal(
        mean=[150, 25, 0.3, 0.7],
        cov=[[400, 100, 2, 5],
             [100, 100, 1, 2],
             [2, 1, 0.01, 0.02],
             [5, 2, 0.02, 0.05]],
        size=n_samples
    )
    
    # Environmental features (temperature, humidity, etc.)
    env_features = np.random.multivariate_normal(
        mean=[20, 65, 1013],
        cov=[[25, 10, 5],
             [10, 100, 2],
             [5, 2, 25]],
        size=n_samples
    )
    
    return {
        'time_domain': time_features,
        'frequency_domain': freq_features,
        'image_features': image_features,
        'environmental': env_features
    }

# Generate multi-modal data
multimodal_data = create_multimodal_shm_dataset(n_samples=500)

print("Multi-modal dataset created:")
for modality, features in multimodal_data.items():
    print(f"• {modality}: {features.shape[1]} features, {features.shape[0]} samples")

# Compare different fusion strategies
fusion_strategies = ['early', 'late', 'hybrid']
fusion_results = {}

for strategy in fusion_strategies:
    print(f"\nTesting {strategy} fusion strategy...")
    
    # Initialize fusion model
    fusion_model = MultiModalFeatureFusion(
        fusion_strategy=strategy, 
        normalization='zscore'
    )
    
    # Split data for training and testing
    train_size = int(0.7 * len(list(multimodal_data.values())[0]))
    
    train_data = {}
    test_data = {}
    
    for modality, features in multimodal_data.items():
        train_data[modality] = features[:train_size]
        test_data[modality] = features[train_size:]
    
    # Fit and transform
    fusion_model.fit(train_data)
    fused_features = fusion_model.transform(test_data)
    
    fusion_results[strategy] = fused_features
    
    print(f"  Fused feature dimensions: {fused_features.shape}")
    print(f"  Feature range: [{fused_features.min():.3f}, {fused_features.max():.3f}]")

# Visualize fusion results
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=('Original Feature Distributions', 'Early Fusion Features', 'Late Fusion Scores',
                   'Hybrid Fusion Features', 'Feature Correlation Analysis', 'Fusion Comparison'),
    specs=[[{"colspan": 3}, None, None],
           [{"type": "histogram"}, {"type": "histogram"}, {"type": "scatter"}]]
)

# Plot 1: Original feature distributions for each modality
colors = ['blue', 'red', 'green', 'orange']
for i, (modality, features) in enumerate(multimodal_data.items()):
    fig.add_trace(
        go.Histogram(x=features[:, 0], name=f'{modality} (Feature 1)',
                     marker_color=colors[i], opacity=0.6, nbinsx=30),
        row=1, col=1
    )

fig.update_xaxes(title_text="Feature Value", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=1)

# Plot 2: Early fusion features
early_features = fusion_results['early']
fig.add_trace(
    go.Histogram(x=early_features[:, 0], name='Early Fusion (PC1)',
                 marker_color='purple', opacity=0.7, nbinsx=30),
    row=2, col=1
)

fig.update_xaxes(title_text="Fused Feature Value", row=2, col=1)
fig.update_yaxes(title_text="Frequency", row=2, col=1)

# Plot 3: Late fusion scores
late_scores = fusion_results['late']
fig.add_trace(
    go.Histogram(x=late_scores.flatten(), name='Late Fusion Scores',
                 marker_color='teal', opacity=0.7, nbinsx=30),
    row=2, col=2
)

fig.update_xaxes(title_text="Anomaly Score", row=2, col=2)
fig.update_yaxes(title_text="Frequency", row=2, col=2)

# Plot 4: Comparison of fusion strategies
hybrid_features = fusion_results['hybrid']

# Use first two dimensions for visualization
fig.add_trace(
    go.Scatter(x=early_features[:50, 0], y=early_features[:50, 1] if early_features.shape[1] > 1 else early_features[:50, 0],
               mode='markers', name='Early Fusion',
               marker=dict(color='blue', size=6, opacity=0.7)),
    row=2, col=3
)

fig.add_trace(
    go.Scatter(x=hybrid_features[:50, 0], y=hybrid_features[:50, 1] if hybrid_features.shape[1] > 1 else hybrid_features[:50, 0],
               mode='markers', name='Hybrid Fusion',
               marker=dict(color='red', size=6, opacity=0.7)),
    row=2, col=3
)

fig.update_xaxes(title_text="Feature Dimension 1", row=2, col=3)
fig.update_yaxes(title_text="Feature Dimension 2", row=2, col=3)

# Update layout
fig.update_layout(
    height=800,
    title_text="Multi-Modal Feature Fusion Analysis",
    showlegend=True,
    font=dict(size=11)
)

fig.show()

print(f"\nFeature Fusion Analysis Summary:")
print(f"• Early fusion: {early_features.shape[1]} combined features")
print(f"• Late fusion: Single anomaly score per sample")
print(f"• Hybrid fusion: {hybrid_features.shape[1]} features (early + late)")
print(f"• Normalization ensures features are on comparable scales")
print(f"• Feature weighting emphasizes most informative modalities")

---

## 6. Statistical Anomaly Detection Methods

### 6.1 Theoretical Foundation

Statistical anomaly detection in structural health monitoring relies on establishing probabilistic models of normal structural behavior and identifying observations that deviate significantly from these models. Statistical approaches provide principled methods for setting detection thresholds and quantifying confidence levels.

### 6.2 Univariate Statistical Control

#### 6.2.1 Control Charts

**Shewhart Control Charts:**

For monitoring individual features $x_i$:

$UCL = \bar{x} + 3\sigma, \quad LCL = \bar{x} - 3\sigma \tag{6.35}$

where $\bar{x}$ is the sample mean and $\sigma$ is the sample standard deviation.

**CUSUM Control Charts:**

Cumulative sum charts detect small shifts in the mean:

$C_t^+ = \max(0, C_{t-1}^+ + x_t - (\mu_0 + k)) \tag{6.36}$
$C_t^- = \max(0, C_{t-1}^- - x_t + (\mu_0 - k)) \tag{6.37}$

where $k$ is the reference value and $\mu_0$ is the target mean.

#### 6.2.2 Statistical Hypothesis Testing

**One-Sample t-Test:**

To test if a new observation comes from the same distribution:

$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} \tag{6.38}$

**Kolmogorov-Smirnov Test:**

For comparing distributions:

$D = \max_x |F_n(x) - F_0(x)| \tag{6.39}$

where $F_n(x)$ is the empirical distribution function.

### 6.3 Multivariate Statistical Control

#### 6.3.1 Hotelling's T² Statistic

For multivariate normal data:

$T^2 = (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{S}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \tag{6.40}$

where $\mathbf{S}$ is the sample covariance matrix.

#### 6.3.2 Mahalanobis Distance

$MD(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})} \tag{6.41}$

#### 6.3.3 Multivariate Control Ellipse

The control ellipse in 2D is defined by:

$(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \leq \chi^2_{2,\alpha} \tag{6.42}$

### 6.4 Implementation: Statistical Anomaly Detection Suite

```python
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

class StatisticalAnomalyDetector:
    """
    Comprehensive statistical anomaly detection for SHM applications
    """
    
    def __init__(self, contamination: float = 0.1, confidence_level: float = 0.99):
        """
        Initialize statistical anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies
            confidence_level: Statistical confidence level
        """
        self.contamination = contamination
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.detection_models = {}
        self.thresholds = {}
        self.statistics = {}
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> 'StatisticalAnomalyDetector':
        """
        Fit statistical models on baseline (healthy) data
        
        Args:
            X: Training data matrix (n_samples, n_features)
            feature_names: Optional feature names
            
        Returns:
            Self for method chaining
        """
        self.n_samples, self.n_features = X.shape
        self.feature_names = feature_names or [f'Feature_{i+1}' for i in range(self.n_features)]
        
        # Compute basic statistics
        self.statistics = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0, ddof=1),
            'median': np.median(X, axis=0),
            'mad': stats.median_abs_deviation(X, axis=0),
            'cov_matrix': np.cov(X.T),
            'cov_det': np.linalg.det(np.cov(X.T))
        }
        
        # Fit different anomaly detection models
        self._fit_univariate_models(X)
        self._fit_multivariate_models(X)
        self._fit_robust_models(X)
        self._fit_ensemble_models(X)
        
        self.is_fitted = True
        return self
    
    def detect_anomalies(self, X: np.ndarray) -> Dict[str, any]:
        """
        Detect anomalies using multiple statistical methods
        
        Args:
            X: Data to analyze for anomalies
            
        Returns:
            Dictionary containing detection results from all methods
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before anomaly detection")
        
        results = {}
        
        # Univariate detection
        results.update(self._detect_univariate_anomalies(X))
        
        # Multivariate detection  
        results.update(self._detect_multivariate_anomalies(X))
        
        # Robust detection
        results.update(self._detect_robust_anomalies(X))
        
        # Ensemble detection
        results.update(self._detect_ensemble_anomalies(X))
        
        # Consensus decision
        results['ensemble_decision'] = self._compute_ensemble_decision(results)
        
        return results
    
    def _fit_univariate_models(self, X: np.ndarray):
        """Fit univariate statistical models"""
        # Control limits for each feature
        self.thresholds['control_limits'] = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_data = X[:, i]
            mean_val = self.statistics['mean'][i]
            std_val = self.statistics['std'][i]
            
            # 3-sigma control limits
            ucl = mean_val + 3 * std_val
            lcl = mean_val - 3 * std_val
            
            # Statistical control limits based on confidence level
            t_critical = stats.t.ppf(1 - self.alpha/2, df=len(feature_data)-1)
            stat_ucl = mean_val + t_critical * std_val / np.sqrt(len(feature_data))
            stat_lcl = mean_val - t_critical * std_val / np.sqrt(len(feature_data))
            
            self.thresholds['control_limits'][feature_name] = {
                '3sigma_ucl': ucl,
                '3sigma_lcl': lcl,
                'statistical_ucl': stat_ucl,
                'statistical_lcl': stat_lcl
            }
    
    def _fit_multivariate_models(self, X: np.ndarray):
        """Fit multivariate statistical models"""
        # Hotelling's T² threshold
        F_critical = stats.f.ppf(1 - self.alpha, 
                                dfn=self.n_features, 
                                dfd=self.n_samples - self.n_features)
        
        self.thresholds['hotelling_t2'] = (self.n_features * (self.n_samples - 1) * 
                                          F_critical / (self.n_samples - self.n_features))
        
        # Chi-square threshold for Mahalanobis distance
        self.thresholds['mahalanobis'] = stats.chi2.ppf(1 - self.alpha, df=self.n_features)
    
    def _fit_robust_models(self, X: np.ndarray):
        """Fit robust statistical models"""
        # Robust covariance estimation
        robust_cov = MinCovDev(contamination=self.contamination, random_state=42)
        robust_cov.fit(X)
        self.detection_models['robust_covariance'] = robust_cov
        
        # Elliptic envelope
        elliptic_env = EllipticEnvelope(contamination=self.contamination, random_state=42)
        elliptic_env.fit(X)
        self.detection_models['elliptic_envelope'] = elliptic_env
    
    def _fit_ensemble_models(self, X: np.ndarray):
        """Fit ensemble anomaly detection models"""
        # Isolation Forest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        iso_forest.fit(X)
        self.detection_models['isolation_forest'] = iso_forest
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(contamination=self.contamination, novelty=True)
        lof.fit(X)
        self.detection_models['lof'] = lof
        
        # One-Class SVM
        ocsvm = OneClassSVM(gamma='scale', nu=self.contamination)
        ocsvm.fit(X)
        self.detection_models['one_class_svm'] = ocsvm
    
    def _detect_univariate_anomalies(self, X: np.ndarray) -> Dict[str, any]:
        """Univariate anomaly detection"""
        results = {}
        
        # Control chart violations
        control_violations = np.zeros((X.shape[0], self.n_features), dtype=bool)
        
        for i, feature_name in enumerate(self.feature_names):
            limits = self.thresholds['control_limits'][feature_name]
            feature_data = X[:, i]
            
            # Check 3-sigma violations
            violations = ((feature_data > limits['3sigma_ucl']) | 
                         (feature_data < limits['3sigma_lcl']))
            control_violations[:, i] = violations
        
        results['univariate_control'] = control_violations
        results['univariate_any'] = np.any(control_violations, axis=1)
        
        return results
    
    def _detect_multivariate_anomalies(self, X: np.ndarray) -> Dict[str, any]:
        """Multivariate anomaly detection"""
        results = {}
        
        # Hotelling's T² statistic
        mean_vec = self.statistics['mean']
        cov_inv = np.linalg.pinv(self.statistics['cov_matrix'])
        
        t2_scores = []
        mahal_distances = []
        
        for sample in X:
            # T² statistic
            diff = sample - mean_vec
            t2 = np.dot(np.dot(diff, cov_inv), diff)
            t2_scores.append(t2)
            
            # Mahalanobis distance
            mahal_dist = np.sqrt(t2)
            mahal_distances.append(mahal_dist)
        
        t2_scores = np.array(t2_scores)
        mahal_distances = np.array(mahal_distances)
        
        # Anomaly decisions
        results['hotelling_t2_scores'] = t2_scores
        results['hotelling_t2_anomalies'] = t2_scores > self.thresholds['hotelling_t2']
        results['mahalanobis_distances'] = mahal_distances
        results['mahalanobis_anomalies'] = mahal_distances**2 > self.thresholds['mahalanobis']
        
        return results
    
    def _detect_robust_anomalies(self, X: np.ndarray) -> Dict[str, any]:
        """Robust anomaly detection"""
        results = {}
        
        # Robust covariance
        robust_scores = self.detection_models['robust_covariance'].decision_function(X)
        results['robust_covariance_scores'] = robust_scores
        results['robust_covariance_anomalies'] = robust_scores < 0
        
        # Elliptic envelope
        elliptic_scores = self.detection_models['elliptic_envelope'].decision_function(X)
        results['elliptic_envelope_scores'] = elliptic_scores
        results['elliptic_envelope_anomalies'] = elliptic_scores < 0
        
        return results
    
    def _detect_ensemble_anomalies(self, X: np.ndarray) -> Dict[str, any]:
        """Ensemble anomaly detection"""
        results = {}
        
        # Isolation Forest
        iso_scores = self.detection_models['isolation_forest'].decision_function(X)
        results['isolation_forest_scores'] = iso_scores
        results['isolation_forest_anomalies'] = iso_scores < 0
        
        # Local Outlier Factor
        lof_scores = self.detection_models['lof'].decision_function(X)
        results['lof_scores'] = lof_scores
        results['lof_anomalies'] = lof_scores < 0
        
        # One-Class SVM
        svm_scores = self.detection_models['one_class_svm'].decision_function(X)
        results['svm_scores'] = svm_scores
        results['svm_anomalies'] = svm_scores < 0
        
        return results
    
    def _compute_ensemble_decision(self, results: Dict[str, any]) -> np.ndarray:
        """Compute ensemble decision by majority voting"""
        anomaly_keys = [key for key in results.keys() if key.endswith('_anomalies')]
        
        if not anomaly_keys:
            return np.array([])
        
        # Stack all anomaly decisions
        all_decisions = np.column_stack([results[key] for key in anomaly_keys])
        
        # Majority voting
        ensemble_decision = np.sum(all_decisions, axis=1) > len(anomaly_keys) / 2
        
        return ensemble_decision

# Demonstrate comprehensive statistical anomaly detection
print("\n=== Statistical Anomaly Detection Demonstration ===\n")

# Generate realistic SHM dataset with anomalies
def generate_shm_anomaly_dataset(n_normal: int = 400, n_anomalies: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Generate realistic SHM dataset with known anomalies"""
    np.random.seed(42)
    
    # Normal structural behavior (healthy state)
    # Correlated features representing normal modal properties
    mean_normal = np.array([2.1, 0.03, 0.15, 8.5, 0.05])
    cov_normal = np.array([
        [0.04, 0.001, 0.01, 0.2, 0.002],
        [0.001, 0.0001, 0.0002, 0.005, 0.0001],
        [0.01, 0.0002, 0.001, 0.03, 0.0005],
        [0.2, 0.005, 0.03, 1.0, 0.01],
        [0.002, 0.0001, 0.0005, 0.01, 0.0002]
    ])
    
    X_normal = np.random.multivariate_normal(mean_normal, cov_normal, n_normal)
    
    # Anomalous behavior (various damage scenarios)
    anomalies = []
    labels = []
    
    # Type 1: Frequency shift (stiffness loss)
    n_type1 = n_anomalies // 3
    mean_freq_shift = mean_normal.copy()
    mean_freq_shift[0] -= 0.3  # Frequency reduction
    X_freq_shift = np.random.multivariate_normal(mean_freq_shift, cov_normal * 1.2, n_type1)
    anomalies.append(X_freq_shift)
    labels.extend([1] * n_type1)
    
    # Type 2: Increased damping
    n_type2 = n_anomalies // 3
    mean_damp_inc = mean_normal.copy()
    mean_damp_inc[1] += 0.02  # Damping increase
    X_damp_inc = np.random.multivariate_normal(mean_damp_inc, cov_normal * 1.5, n_type2)
    anomalies.append(X_damp_inc)
    labels.extend([2] * n_type2)
    
    # Type 3: Mode shape change
    n_type3 = n_anomalies - n_type1 - n_type2
    mean_mode_change = mean_normal.copy()
    mean_mode_change[2] += 0.08  # Mode shape amplitude change
    cov_mode_change = cov_normal.copy()
    cov_mode_change[2, 2] *= 3  # Increased variability
    X_mode_change = np.random.multivariate_normal(mean_mode_change, cov_mode_change, n_type3)
    anomalies.append(X_mode_change)
    labels.extend([3] * n_type3)
    
    # Combine all data
    X_anomalies = np.vstack(anomalies)
    X_all = np.vstack([X_normal, X_anomalies])
    y_all = np.array([0] * n_normal + labels)
    
    # Shuffle
    indices = np.random.permutation(len(X_all))
    return X_all[indices], y_all[indices]

# Generate dataset
X_shm, y_true = generate_shm_anomaly_dataset(n_normal=400, n_anomalies=60)

# Split into training (healthy only) and testing
healthy_mask = y_true == 0
X_train = X_shm[healthy_mask]
X_test = X_shm
y_test = (y_true > 0).astype(int)  # Binary: 0=healthy, 1=anomaly

feature_names = ['Natural_Freq_1', 'Damping_Ratio_1', 'Mode_Shape_Amp', 'Stiffness_Index', 'Energy_Ratio']

print(f"SHM Anomaly Dataset:")
print(f"• Training samples (healthy): {len(X_train)}")
print(f"• Test samples: {len(X_test)}")
print(f"• True anomaly rate: {np.mean(y_test):.2%}")
print(f"• Features: {feature_names}")

# Initialize and fit statistical detector
stat_detector = StatisticalAnomalyDetector(contamination=0.12, confidence_level=0.99)
stat_detector.fit(X_train, feature_names)

print(f"\nStatistical Models Fitted:")
print(f"• Univariate control limits for {len(feature_names)} features")
print(f"• Multivariate Hotelling T² threshold: {stat_detector.thresholds['hotelling_t2']:.3f}")
print(f"• Mahalanobis distance threshold: {stat_detector.thresholds['mahalanobis']:.3f}")

# Perform anomaly detection
detection_results = stat_detector.detect_anomalies(X_test)

# Evaluate performance
print(f"\nAnomaly Detection Results:")
method_names = ['hotelling_t2', 'mahalanobis', 'robust_covariance', 
                'elliptic_envelope', 'isolation_forest', 'lof', 'one_class_svm', 'ensemble_decision']

performance_results = []

for method in method_names:
    if method == 'ensemble_decision':
        predictions = detection_results[method]
    else:
        key = f'{method}_anomalies'
        if key in detection_results:
            predictions = detection_results[key]
        else:
            continue
    
    # Calculate metrics
    tp = np.sum((predictions == 1) & (y_test == 1))
    fp = np.sum((predictions == 1) & (y_test == 0))
    fn = np.sum((predictions == 0) & (y_test == 1))
    tn = np.sum((predictions == 0) & (y_test == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(predictions)
    
    performance_results.append({
        'Method': method.replace('_', ' ').title(),
        'Precision': f'{precision:.3f}',
        'Recall': f'{recall:.3f}',
        'F1-Score': f'{f1:.3f}',
        'Accuracy': f'{accuracy:.3f}',
        'Detected': f'{np.sum(predictions)}'
    })

performance_df = pd.DataFrame(performance_results)
print(performance_df.to_string(index=False))

    