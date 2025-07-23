# Chapter 7: Machine Learning Feature Extraction for Structural Health Monitoring

**Mohammad Talebi-Kalaleh – University of Alberta**

---

## Overview

Machine learning has revolutionized structural health monitoring by providing intelligent tools to extract meaningful patterns from complex sensor data, detect anomalies, and identify structural damage with unprecedented accuracy. This chapter explores fundamental machine learning techniques specifically tailored for SHM applications, with emphasis on unsupervised learning methods that can operate with limited labeled damage data—a common challenge in real-world bridge monitoring systems.

Conventional damage detection techniques are gradually being replaced by state-of-the-art smart monitoring and decision-making solutions, with machine learning algorithms providing the necessary tools to augment the capabilities of SHM systems and provide intelligent solutions for the challenges of the past. Machine Learning algorithms can identify cracks and fatigue in bridge structures with over 90% accuracy, demonstrating the transformative potential of these approaches.

The unique challenges of SHM—including high-dimensional sensor data, limited labeled damage examples, environmental variability, and the need for real-time processing—make machine learning particularly well-suited for advancing the field. This chapter provides both theoretical foundations and practical implementations using PyTorch, focusing on three key approaches: k-means clustering for behavioral pattern analysis, autoencoders for anomaly detection, and generative adversarial networks for data augmentation.

---

## 7.1 Introduction and Motivation

### 7.1.1 Why Machine Learning for SHM?

Traditional structural health monitoring approaches rely heavily on physics-based models and threshold-based damage detection methods. While these approaches have served the field well, they face several fundamental limitations when applied to complex, real-world bridge structures:

**Challenge 1: High-Dimensional Data Complexity**
Modern SHM systems generate massive amounts of multi-sensor data—acceleration time series, strain measurements, temperature readings, and visual imagery. For complex bridge structures, diagnosing structural health based on highly incomplete monitoring data presents an inherent high-dimensional problem. Traditional analysis methods struggle to identify subtle patterns and relationships within such complex datasets.

**Challenge 2: Limited Labeled Damage Data**
Unlike many machine learning applications where abundant labeled training data exists, SHM faces a fundamental scarcity of labeled damage examples. Structures spend most of their operational life in healthy conditions, and catastrophic failures (thankfully) remain rare events. This creates a significant class imbalance that machine learning techniques must address.

**Challenge 3: Environmental and Operational Variability**
Bridge responses vary significantly due to environmental factors (temperature, humidity, wind) and operational conditions (traffic loading, seasonal effects). Although supervised methods have been proven to be effective for detecting data anomalies, two unresolved challenges reduce the accuracy of anomaly detection: (1) the class imbalance and (2) incompleteness of anomalous patterns of training dataset.

**Challenge 4: Real-Time Decision Making**
Modern infrastructure demands continuous, automated monitoring with immediate alerts for potential safety concerns. Machine learning algorithms can process streaming sensor data in real-time, identifying anomalies and potential damage as they occur.

### 7.1.2 Machine Learning Advantages for SHM

Machine learning methods are particularly well-suited for addressing these issues due to their capabilities in effective feature extraction, efficient optimization, and robust scalability. The key advantages include:

**Automated Pattern Recognition**: ML algorithms can automatically discover complex, non-linear relationships in sensor data that would be impossible to identify manually or through traditional statistical methods.

**Unsupervised Learning Capabilities**: Many ML techniques can learn from healthy structural behavior without requiring labeled damage examples, making them practical for real-world deployment.

**Scalability**: Once trained, ML models can process large volumes of sensor data efficiently, making them suitable for continuous monitoring of multiple structures.

**Adaptation and Learning**: Advanced ML approaches can continuously adapt to changing environmental conditions and structural aging, maintaining accuracy over time.

### 7.1.3 SHM-Specific Machine Learning Workflow

The integration of machine learning into structural health monitoring follows a specialized workflow designed to address the unique characteristics of infrastructure data:

```svg
<svg width="800" height="400" viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title-text { font-family: 'Arial', sans-serif; font-size: 14px; font-weight: bold; fill: #2c3e50; }
      .step-text { font-family: 'Arial', sans-serif; font-size: 12px; fill: #34495e; }
      .box { fill: #ecf0f1; stroke: #34495e; stroke-width: 2; rx: 8; }
      .data-box { fill: #e8f5e8; stroke: #27ae60; stroke-width: 2; rx: 8; }
      .ml-box { fill: #e8f4fd; stroke: #3498db; stroke-width: 2; rx: 8; }
      .output-box { fill: #fdf2e8; stroke: #e67e22; stroke-width: 2; rx: 8; }
      .arrow { stroke: #34495e; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e" />
    </marker>
  </defs>
  
  <!-- Title -->
  <text x="400" y="25" text-anchor="middle" class="title-text">Machine Learning Workflow for Structural Health Monitoring</text>
  
  <!-- Data Collection -->
  <rect x="50" y="60" width="120" height="80" class="data-box"/>
  <text x="110" y="85" text-anchor="middle" class="step-text">Multi-Sensor</text>
  <text x="110" y="100" text-anchor="middle" class="step-text">Data Collection</text>
  <text x="110" y="115" text-anchor="middle" class="step-text">• Acceleration</text>
  <text x="110" y="125" text-anchor="middle" class="step-text">• Strain</text>
  <text x="110" y="135" text-anchor="middle" class="step-text">• Environmental</text>
  
  <!-- Data Preprocessing -->
  <rect x="220" y="60" width="120" height="80" class="box"/>
  <text x="280" y="85" text-anchor="middle" class="step-text">Data</text>
  <text x="280" y="100" text-anchor="middle" class="step-text">Preprocessing</text>
  <text x="280" y="115" text-anchor="middle" class="step-text">• Denoising</text>
  <text x="280" y="125" text-anchor="middle" class="step-text">• Normalization</text>
  <text x="280" y="135" text-anchor="middle" class="step-text">• Feature Extraction</text>
  
  <!-- ML Training -->
  <rect x="390" y="60" width="120" height="80" class="ml-box"/>
  <text x="450" y="85" text-anchor="middle" class="step-text">ML Model</text>
  <text x="450" y="100" text-anchor="middle" class="step-text">Training</text>
  <text x="450" y="115" text-anchor="middle" class="step-text">• Clustering</text>
  <text x="450" y="125" text-anchor="middle" class="step-text">• Autoencoders</text>
  <text x="450" y="135" text-anchor="middle" class="step-text">• GANs</text>
  
  <!-- Decision Making -->
  <rect x="560" y="60" width="120" height="80" class="output-box"/>
  <text x="620" y="85" text-anchor="middle" class="step-text">Automated</text>
  <text x="620" y="100" text-anchor="middle" class="step-text">Decision Making</text>
  <text x="620" y="115" text-anchor="middle" class="step-text">• Anomaly Detection</text>
  <text x="620" y="125" text-anchor="middle" class="step-text">• Damage Classification</text>
  <text x="620" y="135" text-anchor="middle" class="step-text">• Health Assessment</text>
  
  <!-- Arrows -->
  <line x1="170" y1="100" x2="220" y2="100" class="arrow"/>
  <line x1="340" y1="100" x2="390" y2="100" class="arrow"/>
  <line x1="510" y1="100" x2="560" y2="100" class="arrow"/>
  
  <!-- Feedback Loops -->
  <path d="M 620 160 Q 620 200 450 200 Q 280 200 280 160" class="arrow"/>
  <text x="450" y="220" text-anchor="middle" class="step-text">Continuous Learning and Model Refinement</text>
  
  <!-- Real-time Processing -->
  <rect x="50" y="260" width="630" height="60" class="box"/>
  <text x="365" y="285" text-anchor="middle" class="step-text">Real-Time Processing Pipeline</text>
  <text x="365" y="300" text-anchor="middle" class="step-text">Streaming sensor data → Preprocessing → Trained ML models → Immediate alerts/decisions</text>
  
  <!-- Environmental Adaptation -->
  <rect x="50" y="340" width="300" height="40" class="ml-box"/>
  <text x="200" y="360" text-anchor="middle" class="step-text">Environmental Adaptation</text>
  <text x="200" y="375" text-anchor="middle" class="step-text">Temperature, seasonal, operational variations</text>
  
  <rect x="380" y="340" width="300" height="40" class="output-box"/>
  <text x="530" y="360" text-anchor="middle" class="step-text">Uncertainty Quantification</text>
  <text x="530" y="375" text-anchor="middle" class="step-text">Confidence levels and reliability metrics</text>
</svg>
```

*Figure 7.1: Machine Learning workflow specifically designed for structural health monitoring applications, highlighting the integration of multi-sensor data, unsupervised learning approaches, and real-time decision making capabilities.*

---

## 7.2 Fundamentals of Machine Learning for SHM

### 7.2.1 Supervised vs. Unsupervised Learning in SHM Context

The distinction between supervised and unsupervised learning takes on special significance in structural health monitoring due to the unique characteristics of infrastructure data.

**Supervised Learning in SHM**
Supervised learning requires labeled examples of both healthy and damaged structural states. While this approach can achieve high accuracy when sufficient labeled data exists, it faces significant challenges in SHM applications:

- **Limited Damage Examples**: Catastrophic structural failures are rare, and controlled damage experiments are expensive and often impractical on full-scale structures.
- **Damage Variability**: Different types of damage (fatigue cracks, corrosion, foundation settlement) may manifest differently, requiring extensive labeled datasets for each damage type.
- **Environmental Confounding**: Supervised models may incorrectly associate environmental variations with damage states if not properly trained.

**Unsupervised Learning in SHM**
Unsupervised learning methods work by learning patterns from healthy structural data without requiring labeled damage examples. This approach is particularly well-suited for SHM because:

- **Abundance of Healthy Data**: Structures operate in healthy conditions for most of their service life, providing extensive training data for normal behavior.
- **Anomaly Detection**: Once normal behavior is learned, deviations can be flagged as potential damage or sensor issues.
- **Adaptability**: Models can continuously update their understanding of normal behavior as structures age and environmental conditions change.

### 7.2.2 Feature Engineering for SHM Data

Before applying machine learning algorithms, raw sensor data must be transformed into meaningful features that capture the essential characteristics of structural behavior.

**Time-Domain Features**
Raw acceleration and strain measurements contain rich information about structural dynamics that can be extracted through statistical and signal processing techniques:

$$\text{Mean: } \mu = \frac{1}{N} \sum_{i=1}^{N} x_i \tag{7.1}$$

$$\text{Standard Deviation: } \sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \mu)^2} \tag{7.2}$$

$$\text{Root Mean Square: } \text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2} \tag{7.3}$$

$$\text{Skewness: } S = \frac{1}{N} \sum_{i=1}^{N} \left(\frac{x_i - \mu}{\sigma}\right)^3 \tag{7.4}$$

$$\text{Kurtosis: } K = \frac{1}{N} \sum_{i=1}^{N} \left(\frac{x_i - \mu}{\sigma}\right)^4 \tag{7.5}$$

where $x_i$ represents individual sensor measurements, $N$ is the number of data points, $\mu$ is the mean, and $\sigma$ is the standard deviation.

**Frequency-Domain Features**
Spectral characteristics often provide more sensitive damage indicators than time-domain features, as structural damage typically manifests as changes in modal properties:

$$\text{Power Spectral Density: } S_{xx}(f) = \mathcal{F}\{R_{xx}(\tau)\} \tag{7.6}$$

$$\text{Spectral Centroid: } f_c = \frac{\sum_{k} f_k S_{xx}(f_k)}{\sum_{k} S_{xx}(f_k)} \tag{7.7}$$

where $\mathcal{F}$ denotes the Fourier transform, $R_{xx}(\tau)$ is the autocorrelation function, $f_k$ are frequency bins, and $S_{xx}(f_k)$ are the corresponding spectral values.

Let's implement a comprehensive feature extraction framework:

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.stats import skew, kurtosis
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class SHMFeatureExtractor:
    """
    Comprehensive feature extraction for structural health monitoring data.
    Focuses on bridge acceleration and strain measurements.
    """
    
    def __init__(self, sampling_rate=200):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate (float): Sampling frequency in Hz
        """
        self.fs = sampling_rate
        
    def extract_time_domain_features(self, signal_data):
        """
        Extract statistical features from time-domain signals.
        
        Args:
            signal_data (array): Time series signal data
            
        Returns:
            dict: Dictionary containing extracted features
        """
        features = {}
        
        # Basic statistical moments
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        
        # Higher-order moments
        features['skewness'] = skew(signal_data)
        features['kurtosis'] = kurtosis(signal_data)
        
        # Peak and extreme value features
        features['peak_to_peak'] = np.ptp(signal_data)
        features['crest_factor'] = np.max(np.abs(signal_data)) / features['rms']
        
        # Energy-based features
        features['energy'] = np.sum(signal_data**2)
        features['power'] = features['energy'] / len(signal_data)
        
        return features
    
    def extract_frequency_domain_features(self, signal_data, nperseg=1024):
        """
        Extract features from frequency domain representation.
        
        Args:
            signal_data (array): Time series signal data
            nperseg (int): Length of each segment for spectral analysis
            
        Returns:
            dict: Dictionary containing frequency domain features
        """
        features = {}
        
        # Compute power spectral density
        f, psd = signal.welch(signal_data, fs=self.fs, nperseg=nperseg)
        
        # Spectral moments and centroids
        features['spectral_energy'] = np.sum(psd)
        features['spectral_centroid'] = np.sum(f * psd) / np.sum(psd)
        features['spectral_spread'] = np.sqrt(np.sum(((f - features['spectral_centroid'])**2) * psd) / np.sum(psd))
        
        # Frequency band power ratios (useful for modal analysis)
        # Low frequency band (0-5 Hz) - typical for bridge fundamental modes
        low_freq_mask = (f >= 0) & (f <= 5)
        mid_freq_mask = (f > 5) & (f <= 20)
        high_freq_mask = f > 20
        
        total_power = np.sum(psd)
        features['low_freq_power_ratio'] = np.sum(psd[low_freq_mask]) / total_power
        features['mid_freq_power_ratio'] = np.sum(psd[mid_freq_mask]) / total_power
        features['high_freq_power_ratio'] = np.sum(psd[high_freq_mask]) / total_power
        
        # Dominant frequency (fundamental frequency)
        dominant_freq_idx = np.argmax(psd)
        features['dominant_frequency'] = f[dominant_freq_idx]
        features['dominant_frequency_magnitude'] = psd[dominant_freq_idx]
        
        return features
    
    def extract_all_features(self, signal_data):
        """
        Extract comprehensive feature set from signal data.
        
        Args:
            signal_data (array): Time series signal data
            
        Returns:
            dict: Combined time and frequency domain features
        """
        time_features = self.extract_time_domain_features(signal_data)
        freq_features = self.extract_frequency_domain_features(signal_data)
        
        # Combine all features
        all_features = {**time_features, **freq_features}
        
        return all_features

# Demonstration with realistic bridge acceleration data
def generate_realistic_bridge_data(duration=60, fs=200, bridge_type='cable_stayed'):
    """
    Generate realistic bridge acceleration data for demonstration.
    Simulates typical bridge response under traffic loading.
    
    Args:
        duration (float): Duration in seconds
        fs (float): Sampling frequency
        bridge_type (str): Type of bridge (affects modal frequencies)
        
    Returns:
        tuple: (time_vector, acceleration_data)
    """
    t = np.linspace(0, duration, int(duration * fs))
    
    # Bridge modal parameters (typical for different bridge types)
    if bridge_type == 'cable_stayed':
        # First few modal frequencies for cable-stayed bridges
        modal_freqs = [0.8, 1.2, 2.1, 3.5, 4.8]  # Hz
        modal_dampings = [0.02, 0.015, 0.02, 0.025, 0.03]  # Damping ratios
    elif bridge_type == 'suspension':
        modal_freqs = [0.3, 0.6, 1.1, 1.8, 2.5]
        modal_dampings = [0.015, 0.01, 0.015, 0.02, 0.025]
    else:  # simply supported
        modal_freqs = [2.5, 4.8, 7.2, 9.8, 12.5]
        modal_dampings = [0.02, 0.02, 0.025, 0.03, 0.035]
    
    # Initialize acceleration signal
    acceleration = np.zeros_like(t)
    
    # Add modal responses with realistic amplitudes
    for i, (freq, damping) in enumerate(zip(modal_freqs, modal_dampings)):
        # Modal amplitude decreases with mode number
        amplitude = 0.1 * (0.8 ** i)  # m/s²
        
        # Damped sinusoidal response
        modal_response = amplitude * np.exp(-2 * np.pi * freq * damping * t) * \
                        np.sin(2 * np.pi * freq * np.sqrt(1 - damping**2) * t)
        
        acceleration += modal_response
    
    # Add traffic-induced random excitation
    traffic_noise = np.random.normal(0, 0.02, len(t))  # Background traffic
    
    # Add individual vehicle passages (random intervals)
    n_vehicles = np.random.poisson(duration / 20)  # Average 1 vehicle per 20 seconds
    for _ in range(n_vehicles):
        vehicle_time = np.random.uniform(0, duration)
        vehicle_duration = np.random.uniform(3, 8)  # Vehicle passage duration
        
        # Gaussian pulse for vehicle loading
        vehicle_mask = np.abs(t - vehicle_time) <= vehicle_duration / 2
        if np.any(vehicle_mask):
            vehicle_loading = 0.05 * np.exp(-((t[vehicle_mask] - vehicle_time) / (vehicle_duration / 4))**2)
            acceleration[vehicle_mask] += vehicle_loading
    
    # Add environmental noise
    environmental_noise = np.random.normal(0, 0.005, len(t))
    
    # Combine all components
    total_acceleration = acceleration + traffic_noise + environmental_noise
    
    return t, total_acceleration

# Generate demonstration data
print("Generating realistic bridge acceleration data...")
time, acc_healthy = generate_realistic_bridge_data(duration=120, bridge_type='cable_stayed')

# Extract features
feature_extractor = SHMFeatureExtractor(sampling_rate=200)
features = feature_extractor.extract_all_features(acc_healthy)

# Display extracted features in a nice table format
feature_df = pd.DataFrame([features]).T
feature_df.columns = ['Value']
feature_df.index.name = 'Feature'

print("\nExtracted Features from Bridge Acceleration Data:")
print("=" * 50)
print(feature_df.round(6))
```

This implementation demonstrates how to extract meaningful features from bridge acceleration data. The features capture both time-domain characteristics (statistical moments, energy content) and frequency-domain properties (spectral content, modal information) that are essential for structural health monitoring.

### 7.2.3 Visualizing Feature Extraction Results

Let's create comprehensive visualizations to understand the feature extraction process:

```python
def visualize_feature_extraction(time, signal_data, features, feature_extractor):
    """
    Create comprehensive visualization of feature extraction process.
    """
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Time-Domain Signal', 'Power Spectral Density',
                       'Statistical Features', 'Frequency Domain Features',
                       'Modal Analysis', 'Feature Summary'],
        specs=[[{"colspan": 2}, None],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "table"}]]
    )
    
    # Time domain plot
    fig.add_trace(
        go.Scatter(x=time, y=signal_data, 
                  line=dict(color='#2E86AB', width=1),
                  name='Acceleration'),
        row=1, col=1
    )
    
    # Frequency domain analysis
    f, psd = signal.welch(signal_data, fs=feature_extractor.fs, nperseg=1024)
    fig.add_trace(
        go.Scatter(x=f, y=10*np.log10(psd),
                  line=dict(color='#A23B72', width=2),
                  name='Power Spectral Density'),
        row=1, col=1
    )
    
    # Statistical features bar plot
    time_domain_features = ['mean', 'std', 'rms', 'skewness', 'kurtosis', 'crest_factor']
    time_values = [features[key] for key in time_domain_features]
    
    fig.add_trace(
        go.Bar(x=time_domain_features, y=time_values,
               marker_color='#F18F01', name='Time Domain'),
        row=2, col=1
    )
    
    # Frequency domain features
    freq_features = ['spectral_centroid', 'spectral_spread', 'dominant_frequency']
    freq_values = [features[key] for key in freq_features]
    
    fig.add_trace(
        go.Bar(x=freq_features, y=freq_values,
               marker_color='#C73E1D', name='Frequency Domain'),
        row=2, col=2
    )
    
    # Modal content visualization
    freq_bands = ['Low (0-5 Hz)', 'Mid (5-20 Hz)', 'High (>20 Hz)']
    power_ratios = [features['low_freq_power_ratio'], 
                   features['mid_freq_power_ratio'],
                   features['high_freq_power_ratio']]
    
    fig.add_trace(
        go.Scatter(x=freq_bands, y=power_ratios,
                  mode='markers+lines',
                  marker=dict(size=12, color='#2E86AB'),
                  line=dict(width=3),
                  name='Power Distribution'),
        row=3, col=1
    )
    
    # Feature summary table
    feature_summary = pd.DataFrame([
        ['Dominant Frequency', f"{features['dominant_frequency']:.2f} Hz"],
        ['RMS Acceleration', f"{features['rms']:.4f} m/s²"],
        ['Spectral Centroid', f"{features['spectral_centroid']:.2f} Hz"],
        ['Crest Factor', f"{features['crest_factor']:.2f}"],
        ['Total Energy', f"{features['energy']:.2e} (m/s²)²·s"]
    ], columns=['Parameter', 'Value'])
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Parameter', 'Value'],
                       fill_color='#2E86AB',
                       font=dict(color='white', size=12)),
            cells=dict(values=[feature_summary['Parameter'], feature_summary['Value']],
                      fill_color='#ECF0F1',
                      font=dict(size=11))
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Comprehensive Feature Extraction from Bridge Acceleration Data",
        title_font_size=16,
        height=900,
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Feature Value", row=2, col=1)
    fig.update_yaxes(title_text="Feature Value", row=2, col=2)
    fig.update_yaxes(title_text="Power Ratio", row=3, col=1)
    
    return fig

# Create the visualization
fig = visualize_feature_extraction(time, acc_healthy, features, feature_extractor)
fig.show()

# Additional analysis: Feature sensitivity to damage
print("\nAnalyzing Feature Sensitivity to Structural Changes...")
print("=" * 55)

# Simulate slight damage by reducing stiffness (lowering frequencies)
time_damaged, acc_damaged = generate_realistic_bridge_data(duration=120, bridge_type='cable_stayed')
# Add slight frequency shift to simulate damage
acc_damaged = acc_damaged + 0.02 * np.sin(2 * np.pi * 0.75 * time_damaged)  # Slight frequency reduction

features_damaged = feature_extractor.extract_all_features(acc_damaged)

# Compare key features
comparison_features = ['dominant_frequency', 'spectral_centroid', 'rms', 'spectral_energy']
print(f"{'Feature':<20} {'Healthy':<12} {'Damaged':<12} {'Change (%)':<12}")
print("-" * 56)

for feature in comparison_features:
    healthy_val = features[feature]
    damaged_val = features_damaged[feature]
    change_percent = ((damaged_val - healthy_val) / healthy_val) * 100
    
    print(f"{feature:<20} {healthy_val:<12.4f} {damaged_val:<12.4f} {change_percent:<12.2f}")
```

This comprehensive feature extraction framework provides the foundation for machine learning applications in SHM. The extracted features capture essential characteristics of structural behavior that can be used for clustering, anomaly detection, and damage identification.

---

## 7.3 k-means Clustering for Structural Behavior Analysis

### 7.3.1 Motivation for Clustering in SHM

Clustering is a technique in which subgroups are assembled based on either features or samples. It performs a partition of data into K non-overlapping clusters. In structural health monitoring, clustering serves several critical purposes:

**Behavioral Pattern Identification**: Bridges exhibit different response patterns under various operational conditions—rush hour traffic, seasonal temperature variations, and different loading scenarios. Clustering can automatically identify these distinct operational states without requiring prior labeling.

**Sensor Network Analysis**: Diez et al. proposed a Clustering-based data-driven machine learning approach, using the k-mean clustering algorithm, to group joints with similar behavior on the bridge to separate the ones working in normal condition from the ones working in abnormal condition. This approach is particularly valuable for large-scale monitoring systems with hundreds of sensors.

**Anomaly Detection Foundation**: By clustering normal operational data, any measurements that fall outside established cluster boundaries can be flagged as potential anomalies or damage indicators.

**Environmental Effect Characterization**: Burrello et al. leveraged the SHM data with an anomaly detection technique to identify traffic load from the acceleration peaks and utilize the K-means algorithm to distinguish amplitude and damping duration associated with heavy traffic and cars, respectively.

### 7.3.2 Mathematical Foundation of k-means Clustering

The k-means algorithm partitions data into k clusters by minimizing the within-cluster sum of squared distances. For SHM applications, this translates to grouping similar structural response patterns together.

**Objective Function**
The algorithm minimizes the following objective function:

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2 \tag{7.8}$$

where:
- $k$ is the number of clusters
- $C_i$ is the $i$-th cluster
- $\mu_i$ is the centroid of cluster $C_i$
- $x$ represents feature vectors extracted from sensor data
- $||x - \mu_i||^2$ is the squared Euclidean distance

**Centroid Update Rule**
The centroid of each cluster is updated iteratively:

$$\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x \tag{7.9}$$

where $|C_i|$ denotes the number of data points in cluster $C_i$.

**Assignment Rule**
Each data point is assigned to the cluster with the nearest centroid:

$$c^{(i)} = \arg\min_j ||x^{(i)} - \mu_j||^2 \tag{7.10}$$

where $c^{(i)}$ is the cluster assignment for data point $x^{(i)}$.

### 7.3.3 SHM-Specific k-means Implementation

Let's implement a specialized k-means clustering approach tailored for structural health monitoring data:

```python
class SHMKMeansCluster:
    """
    Specialized k-means clustering for structural health monitoring data.
    Includes SHM-specific preprocessing and interpretation methods.
    """
    
    def __init__(self, n_clusters=5, max_iters=100, random_state=42):
        """
        Initialize SHM k-means clustering.
        
        Args:
            n_clusters (int): Number of clusters (operational states)
            max_iters (int): Maximum iterations for convergence
            random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def preprocess_shm_data(self, feature_matrix):
        """
        Preprocess SHM feature data for clustering.
        
        Args:
            feature_matrix (array): Matrix of extracted features (n_samples, n_features)
            
        Returns:
            array: Preprocessed and normalized feature matrix
        """
        # Remove any NaN or infinite values that might occur in feature extraction
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features to have zero mean and unit variance
        # This is crucial because different features have different scales
        normalized_features = self.scaler.fit_transform(feature_matrix)
        
        return normalized_features
    
    def fit(self, feature_matrix):
        """
        Fit k-means clustering to SHM feature data.
        
        Args:
            feature_matrix (array): Matrix of extracted features
        """
        # Preprocess data
        self.normalized_data = self.preprocess_shm_data(feature_matrix)
        n_samples, n_features = self.normalized_data.shape
        
        # Initialize centroids randomly
        np.random.seed(self.random_state)
        self.centroids = np.random.randn(self.n_clusters, n_features)
        
        # Store iteration history for analysis
        self.cost_history = []
        
        for iteration in range(self.max_iters):
            # Assign points to closest centroids
            distances = np.sqrt(((self.normalized_data - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([
                self.normalized_data[self.labels == k].mean(axis=0) if np.sum(self.labels == k) > 0 
                else self.centroids[k] 
                for k in range(self.n_clusters)
            ])
            
            # Calculate cost (within-cluster sum of squares)
            cost = np.sum([
                np.sum((self.normalized_data[self.labels == k] - new_centroids[k])**2)
                for k in range(self.n_clusters) if np.sum(self.labels == k) > 0
            ])
            self.cost_history.append(cost)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids, atol=1e-6):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.centroids = new_centroids
            
        # Calculate cluster statistics
        self.calculate_cluster_statistics()
        
    def calculate_cluster_statistics(self):
        """Calculate statistics for each cluster to aid interpretation."""
        self.cluster_stats = {}
        
        for k in range(self.n_clusters):
            cluster_mask = self.labels == k
            cluster_data = self.normalized_data[cluster_mask]
            
            if len(cluster_data) > 0:
                self.cluster_stats[k] = {
                    'size': len(cluster_data),
                    'centroid': self.centroids[k],
                    'inertia': np.sum((cluster_data - self.centroids[k])**2),
                    'avg_distance_to_centroid': np.mean(np.linalg.norm(cluster_data - self.centroids[k], axis=1))
                }
            else:
                self.cluster_stats[k] = {
                    'size': 0,
                    'centroid': self.centroids[k],
                    'inertia': 0,
                    'avg_distance_to_centroid': 0
                }
    
    def predict(self, new_features):
        """
        Predict cluster assignment for new feature data.
        
        Args:
            new_features (array): New feature matrix to classify
            
        Returns:
            array: Cluster assignments
        """
        # Normalize using the same scaler
        normalized_new = self.scaler.transform(new_features)
        
        # Calculate distances to all centroids
        distances = np.sqrt(((normalized_new - self.centroids[:, np.newaxis])**2).sum(axis=2))
        predictions = np.argmin(distances, axis=0)
        
        return predictions
    
    def detect_anomalies(self, new_features, threshold_percentile=95):
        """
        Detect anomalies based on distance to nearest cluster centroid.
        
        Args:
            new_features (array): Feature matrix to analyze
            threshold_percentile (float): Percentile threshold for anomaly detection
            
        Returns:
            tuple: (anomaly_flags, distances, threshold)
        """
        normalized_new = self.scaler.transform(new_features)
        
        # Calculate distance to nearest centroid for each point
        distances = np.sqrt(((normalized_new - self.centroids[:, np.newaxis])**2).sum(axis=2))
        min_distances = np.min(distances, axis=0)
        
        # Determine threshold based on training data
        training_distances = np.sqrt(((self.normalized_data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        training_min_distances = np.min(training_distances, axis=0)
        threshold = np.percentile(training_min_distances, threshold_percentile)
        
        # Flag anomalies
        anomaly_flags = min_distances > threshold
        
        return anomaly_flags, min_distances, threshold

# Generate multiple operational scenarios for clustering demonstration
def generate_multiple_bridge_scenarios(n_scenarios=5, duration_each=60):
    """
    Generate bridge data under different operational conditions for clustering.
    
    Args:
        n_scenarios (int): Number of different operational scenarios
        duration_each (float): Duration of each scenario in seconds
        
    Returns:
        tuple: (all_features, scenario_labels, time_vectors, signals)
    """
    all_features = []
    scenario_labels = []
    all_signals = []
    all_times = []
    
    feature_extractor = SHMFeatureExtractor(sampling_rate=200)
    
    scenarios = [
        ('Light Traffic', {'amplitude_scale': 0.5, 'vehicle_rate': 30, 'noise_level': 0.003}),
        ('Heavy Traffic', {'amplitude_scale': 1.5, 'vehicle_rate': 8, 'noise_level': 0.008}),
        ('Night Conditions', {'amplitude_scale': 0.2, 'vehicle_rate': 60, 'noise_level': 0.002}),
        ('Wind Loading', {'amplitude_scale': 0.8, 'vehicle_rate': 25, 'noise_level': 0.012}),
        ('Temperature Effect', {'amplitude_scale': 0.9, 'vehicle_rate': 20, 'noise_level': 0.005})
    ]
    
    for scenario_name, params in scenarios:
        print(f"Generating {scenario_name} scenario...")
        
        # Generate base signal
        time, signal = generate_realistic_bridge_data(duration=duration_each, bridge_type='cable_stayed')
        
        # Apply scenario-specific modifications
        signal *= params['amplitude_scale']
        
        # Add scenario-specific noise
        additional_noise = np.random.normal(0, params['noise_level'], len(signal))
        signal += additional_noise
        
        # For wind loading, add low-frequency content
        if 'Wind' in scenario_name:
            wind_component = 0.03 * np.sin(2 * np.pi * 0.1 * time) * np.random.randn(len(time))
            signal += wind_component
        
        # For temperature effects, add thermal expansion effects (very low frequency)
        if 'Temperature' in scenario_name:
            thermal_drift = 0.01 * np.sin(2 * np.pi * time / (duration_each * 2))
            signal += thermal_drift
        
        # Extract features
        features = feature_extractor.extract_all_features(signal)
        all_features.append(list(features.values()))
        scenario_labels.append(scenario_name)
        all_signals.append(signal)
        all_times.append(time)
        
        # Generate multiple samples per scenario to have enough data for clustering
        for _ in range(19):  # 20 samples per scenario total
            time_sample, signal_sample = generate_realistic_bridge_data(duration=duration_each, bridge_type='cable_stayed')
            signal_sample *= params['amplitude_scale']
            signal_sample += np.random.normal(0, params['noise_level'], len(signal_sample))
            
            if 'Wind' in scenario_name:
                wind_component = 0.03 * np.sin(2 * np.pi * 0.1 * time_sample) * np.random.randn(len(time_sample))
                signal_sample += wind_component
            
            if 'Temperature' in scenario_name:
                thermal_drift = 0.01 * np.sin(2 * np.pi * time_sample / (duration_each * 2))
                signal_sample += thermal_drift
            
            features_sample = feature_extractor.extract_all_features(signal_sample)
            all_features.append(list(features_sample.values()))
            scenario_labels.append(scenario_name)
    
    return np.array(all_features), scenario_labels, all_times, all_signals

# Generate demonstration data
print("Generating multiple bridge operational scenarios...")
feature_matrix, true_labels, times, signals = generate_multiple_bridge_scenarios()

print(f"Generated feature matrix shape: {feature_matrix.shape}")
print(f"Features extracted: {len(feature_extractor.extract_all_features(signals[0]))}")

# Apply k-means clustering
shm_kmeans = SHMKMeansCluster(n_clusters=5, random_state=42)
shm_kmeans.fit(feature_matrix)

print(f"\nClustering Results:")
print(f"Converged with {len(shm_kmeans.cost_history)} iterations")
print(f"Final cost: {shm_kmeans.cost_history[-1]:.2f}")

# Analyze cluster assignments
print(f"\nCluster Statistics:")
print("-" * 40)
for k in range(shm_kmeans.n_clusters):
    stats = shm_kmeans.cluster_stats[k]
    print(f"Cluster {k}: {stats['size']} samples, inertia: {stats['inertia']:.2f}")
```

### 7.3.4 Clustering Results Visualization and Interpretation

```python
def visualize_clustering_results(feature_matrix, labels, true_labels, shm_kmeans):
    """
    Create comprehensive visualization of clustering results for SHM data.
    """
    from sklearn.decomposition import PCA
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(shm_kmeans.normalized_data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Clustering Results (PCA Space)', 'Convergence History',
                       'Cluster Size Distribution', 'Feature Importance'],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Color scheme for clusters
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Plot clustering results in PCA space
    for k in range(shm_kmeans.n_clusters):
        cluster_mask = labels == k
        if np.any(cluster_mask):
            fig.add_trace(
                go.Scatter(
                    x=features_2d[cluster_mask, 0],
                    y=features_2d[cluster_mask, 1],
                    mode='markers',
                    marker=dict(color=colors[k], size=8, opacity=0.7),
                    name=f'Cluster {k}',
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Plot centroids in PCA space
    centroids_2d = pca.transform(shm_kmeans.centroids)
    fig.add_trace(
        go.Scatter(
            x=centroids_2d[:, 0],
            y=centroids_2d[:, 1],
            mode='markers',
            marker=dict(color='black', size=15, symbol='x', line=dict(width=2)),
            name='Centroids',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Convergence history
    fig.add_trace(
        go.Scatter(
            x=list(range(len(shm_kmeans.cost_history))),
            y=shm_kmeans.cost_history,
            mode='lines+markers',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=6),
            name='Cost',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Cluster size distribution
    cluster_sizes = [shm_kmeans.cluster_stats[k]['size'] for k in range(shm_kmeans.n_clusters)]
    fig.add_trace(
        go.Bar(
            x=[f'Cluster {k}' for k in range(shm_kmeans.n_clusters)],
            y=cluster_sizes,
            marker_color=colors,
            name='Cluster Sizes',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Feature importance (PCA components)
    feature_names = list(feature_extractor.extract_all_features(signals[0]).keys())
    top_features_idx = np.argsort(np.abs(pca.components_[0]))[-5:]  # Top 5 most important
    top_features = [feature_names[i] for i in top_features_idx]
    importance_values = pca.components_[0][top_features_idx]
    
    fig.add_trace(
        go.Bar(
            x=top_features,
            y=importance_values,
            marker_color='#3498db',
            name='Feature Importance',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="k-means Clustering Analysis for Bridge Operational States",
        title_font_size=16,
        height=800,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="First Principal Component", row=1, col=1)
    fig.update_yaxes(title_text="Second Principal Component", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Cost (Within-cluster sum of squares)", row=1, col=2)
    fig.update_yaxes(title_text="Number of Samples", row=2, col=1)
    fig.update_yaxes(title_text="PCA Coefficient", row=2, col=2)
    
    return fig

# Create clustering visualization
clustering_fig = visualize_clustering_results(feature_matrix, shm_kmeans.labels, true_labels, shm_kmeans)
clustering_fig.show()

# Analyze clustering quality
def analyze_clustering_quality(true_labels, predicted_labels):
    """Analyze the quality of clustering results."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    
    # Calculate clustering metrics
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    silhouette_avg = silhouette_score(shm_kmeans.normalized_data, predicted_labels)
    
    print(f"\nClustering Quality Assessment:")
    print(f"Adjusted Rand Index: {ari_score:.3f} (1.0 = perfect match)")
    print(f"Silhouette Score: {silhouette_avg:.3f} (1.0 = well-separated clusters)")
    
    # Create confusion matrix-like analysis
    unique_true = list(set(true_labels))
    unique_pred = list(set(predicted_labels))
    
    print(f"\nCluster Assignment Analysis:")
    print("-" * 50)
    
    confusion_data = []
    for true_label in unique_true:
        true_mask = np.array(true_labels) == true_label
        predicted_for_true = np.array(predicted_labels)[true_mask]
        
        row_data = [true_label]
        for pred_label in range(max(predicted_labels) + 1):
            count = np.sum(predicted_for_true == pred_label)
            row_data.append(count)
            
        confusion_data.append(row_data)
    
    # Create table
    confusion_df = pd.DataFrame(confusion_data, 
                               columns=['True Scenario'] + [f'Cluster {i}' for i in range(max(predicted_labels) + 1)])
    print(confusion_df.to_string(index=False))
    
    return ari_score, silhouette_avg

ari, silhouette = analyze_clustering_quality(true_labels, shm_kmeans.labels)
```

### 7.3.5 Anomaly Detection Using Clustering

One of the most powerful applications of clustering in SHM is anomaly detection. By establishing clusters representing normal operational states, we can identify data points that fall outside these established patterns:

```python
# Demonstrate anomaly detection using clustering
def simulate_structural_damage(signal, damage_type='stiffness_reduction'):
    """
    Simulate different types of structural damage in bridge response.
    
    Args:
        signal (array): Original healthy signal
        damage_type (str): Type of damage to simulate
        
    Returns:
        array: Signal with simulated damage
    """
    damaged_signal = signal.copy()
    
    if damage_type == 'stiffness_reduction':
        # Simulate stiffness reduction by frequency shifting
        # This represents early-stage damage like crack initiation
        time = np.linspace(0, len(signal)/200, len(signal))
        frequency_shift = -0.05  # 5% frequency reduction
        
        # Apply frequency domain modification
        fft_signal = np.fft.fft(damaged_signal)
        freqs = np.fft.fftfreq(len(signal), 1/200)
        
        # Shift main structural frequencies
        for i, freq in enumerate(freqs):
            if 0.5 <= abs(freq) <= 5.0:  # Main structural frequency range
                shifted_freq = freq * (1 + frequency_shift)
                # This is a simplified representation - in practice, more sophisticated
                # frequency domain manipulation would be needed
                
        damaged_signal = damaged_signal * (1 + frequency_shift * 0.1)
        
    elif damage_type == 'connection_loosening':
        # Simulate loose connections by adding impulsive responses
        n_impulses = np.random.poisson(5)  # Random number of loose connection events
        for _ in range(n_impulses):
            impulse_location = np.random.randint(0, len(signal))
            impulse_magnitude = np.random.uniform(0.05, 0.15)
            
            # Add decaying impulse response
            decay_length = min(200, len(signal) - impulse_location)  # 1 second decay
            decay_signal = impulse_magnitude * np.exp(-np.arange(decay_length) / 50)
            
            damaged_signal[impulse_location:impulse_location + decay_length] += decay_signal
            
    elif damage_type == 'bearing_deterioration':
        # Simulate bearing problems with low-frequency content
        time = np.linspace(0, len(signal)/200, len(signal))
        bearing_frequency = 0.3  # Low frequency associated with bearing movement
        bearing_amplitude = 0.02
        
        bearing_component = bearing_amplitude * np.sin(2 * np.pi * bearing_frequency * time)
        damaged_signal += bearing_component
        
    return damaged_signal

# Generate damaged scenarios for anomaly detection testing
print("\nTesting Anomaly Detection with Simulated Damage...")
print("=" * 55)

# Generate healthy test data
test_healthy_features = []
for _ in range(10):
    time_test, signal_test = generate_realistic_bridge_data(duration=60, bridge_type='cable_stayed')
    features_test = feature_extractor.extract_all_features(signal_test)
    test_healthy_features.append(list(features_test.values()))

# Generate damaged test data
test_damaged_features = []
damage_types = ['stiffness_reduction', 'connection_loosening', 'bearing_deterioration']

for damage_type in damage_types:
    for _ in range(5):  # 5 samples per damage type
        time_test, signal_test = generate_realistic_bridge_data(duration=60, bridge_type='cable_stayed')
        damaged_signal = simulate_structural_damage(signal_test, damage_type)
        features_damaged = feature_extractor.extract_all_features(damaged_signal)
        test_damaged_features.append(list(features_damaged.values()))

# Combine test data
test_features = np.array(test_healthy_features + test_damaged_features)
test_labels = ['Healthy'] * 10 + ['Damaged'] * 15

# Perform anomaly detection
anomaly_flags, distances, threshold = shm_kmeans.detect_anomalies(test_features, threshold_percentile=95)

# Analyze results
print(f"Anomaly Detection Results:")
print(f"Detection threshold: {threshold:.4f}")
print(f"Anomalies detected: {np.sum(anomaly_flags)} out of {len(test_features)}")

# Calculate detection metrics
true_anomalies = np.array([label == 'Damaged' for label in test_labels])
true_positives = np.sum(anomaly_flags & true_anomalies)
false_positives = np.sum(anomaly_flags & ~true_anomalies)
false_negatives = np.sum(~anomaly_flags & true_anomalies)
true_negatives = np.sum(~anomaly_flags & ~true_anomalies)

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nDetection Performance Metrics:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1_score:.3f}")

# Create anomaly detection visualization
def visualize_anomaly_detection(distances, anomaly_flags, test_labels, threshold):
    """Visualize anomaly detection results."""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Distance to Nearest Cluster Centroid', 'Detection Performance'],
        vertical_spacing=0.12
    )
    
    # Distance plot
    colors = ['red' if flag else 'blue' for flag in anomaly_flags]
    symbols = ['triangle-up' if label == 'Damaged' else 'circle' for label in test_labels]
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(distances))),
            y=distances,
            mode='markers',
            marker=dict(
                color=colors,
                symbol=symbols,
                size=10,
                line=dict(width=1, color='black')
            ),
            name='Test Samples',
            text=[f'True: {label}, Detected: {"Anomaly" if flag else "Normal"}' 
                  for label, flag in zip(test_labels, anomaly_flags)],
            hovertemplate='Sample %{x}<br>Distance: %{y:.4f}<br>%{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Detection Threshold: {threshold:.4f}",
                  row=1, col=1)
    
    # Performance metrics bar chart
    metrics = ['True Positives', 'False Positives', 'False Negatives', 'True Negatives']
    values = [true_positives, false_positives, false_negatives, true_negatives]
    bar_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker_color=bar_colors,
            text=values,
            textposition='auto',
            name='Detection Results',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title_text="Anomaly Detection Results Using k-means Clustering",
        title_font_size=16,
        height=700,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Sample Index", row=1, col=1)
    fig.update_yaxes(title_text="Distance to Nearest Centroid", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    return fig

anomaly_fig = visualize_anomaly_detection(distances, anomaly_flags, test_labels, threshold)
anomaly_fig.show()
```

This k-means clustering implementation demonstrates how unsupervised learning can effectively identify different operational states in bridge structures and detect anomalies that may indicate damage or sensor malfunctions. The approach is particularly valuable because it doesn't require labeled damage data during training, making it practical for real-world deployment.

---

## 7.4 Deep Unsupervised Models: Autoencoders for Anomaly Detection

### 7.4.1 Motivation for Autoencoders in SHM

Structural health monitoring ensures the safety and longevity of structures like buildings and bridges. We present a novel "deploy-and-forget" approach for automated detection and localization of damage in structures using a mechanics-informed autoencoder. Autoencoders have emerged as one of the most powerful unsupervised learning techniques for SHM due to several key advantages:

**Learning Normal Behavior**: The current paradigm for evaluating data quality commences with the categorization of data anomalies based on expert experience, identifying types such as missing, minor, outlier, square, trend, and drift. Autoencoders can automatically learn the complex patterns that characterize normal structural behavior without requiring manual feature engineering or expert-defined anomaly categories.

**High-Dimensional Data Handling**: Bridge monitoring systems generate vast amounts of multi-sensor data. Jana et al. propose a two-step framework based on CNN and Convolutional Autoencoder (CAE) to leverage the spatio-temporal correlations in sensor data, demonstrating the effectiveness of autoencoders in handling complex, high-dimensional SHM datasets.

**Subtle Anomaly Detection**: Unsupervised methods have the potential to address these challenges, but improvements are required to deal with vast amounts of monitoring data. The generative adversarial networks are combined with a widely applied unsupervised method, that is, autoencoders, to improve the performance of existing unsupervised learning methods.

**Data Compression and Reconstruction**: Autoencoders can compress sensor data into lower-dimensional representations while preserving essential structural information, enabling efficient storage and transmission of monitoring data.

### 7.4.2 Mathematical Foundation of Autoencoders

An autoencoder consists of two main components: an encoder that compresses input data into a lower-dimensional latent representation, and a decoder that reconstructs the original data from this representation.

**Encoder Function**
The encoder maps input data $\mathbf{x} \in \mathbb{R}^n$ to a latent space representation $\mathbf{z} \in \mathbb{R}^m$ (where typically $m < n$):

$$\mathbf{z} = f_{\phi}(\mathbf{x}) = \sigma(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e) \tag{7.11}$$

where $\phi = \{\mathbf{W}_e, \mathbf{b}_e\}$ represents the encoder parameters, $\mathbf{W}_e$ is the weight matrix, $\mathbf{b}_e$ is the bias vector, and $\sigma$ is the activation function.

**Decoder Function**
The decoder reconstructs the input from the latent representation:

$$\hat{\mathbf{x}} = g_{\theta}(\mathbf{z}) = \sigma(\mathbf{W}_d \mathbf{z} + \mathbf{b}_d) \tag{7.12}$$

where $\theta = \{\mathbf{W}_d, \mathbf{b}_d\}$ represents the decoder parameters.

**Loss Function**
The autoencoder is trained to minimize the reconstruction error:

$$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = ||\mathbf{x} - \hat{\mathbf{x}}||^2 = ||\mathbf{x} - g_{\theta}(f_{\phi}(\mathbf{x}))||^2 \tag{7.13}$$

**Anomaly Detection Principle**
For anomaly detection, the reconstruction error serves as an anomaly score. Data points with high reconstruction error are considered anomalous:

$$\text{Anomaly Score} = ||\mathbf{x} - \hat{\mathbf{x}}||^2 \tag{7.14}$$

### 7.4.3 SHM-Specific Autoencoder Architecture

Let's implement a specialized autoencoder architecture designed for structural health monitoring data:

```python
class SHMAutoencoder(nn.Module):
    """
    Specialized autoencoder for structural health monitoring data.
    Designed to learn normal structural behavior patterns and detect anomalies.
    """
    
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[32, 16], dropout_rate=0.1):
        """
        Initialize SHM autoencoder.
        
        Args:
            input_dim (int): Number of input features
            latent_dim (int): Dimension of latent space
            hidden_dims (list): Hidden layer dimensions
            dropout_rate (float): Dropout rate for regularization
        """
        super(SHMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (mirror of encoder)
        decoder_layers = [nn.Linear(latent_dim, prev_dim), nn.ReLU()]
        
        for i in range(len(hidden_dims) - 1, -1, -1):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dims[i]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dims[i]
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
    
    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)

class SHMAutoencoderTrainer:
    """
    Training framework for SHM autoencoder with specialized loss functions and monitoring.
    """
    
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5):
        """
        Initialize trainer.
        
        Args:
            model: SHMAutoencoder model
            learning_rate: Learning rate for optimization
            weight_decay: L2 regularization parameter
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, latent = self.model(data)
            
            # Calculate loss (reconstruction error)
            reconstruction_loss = F.mse_loss(reconstructed, data)
            
            # Add regularization terms for SHM-specific objectives
            # Encourage smooth latent representations
            latent_smoothness = torch.mean(torch.abs(latent[:, 1:] - latent[:, :-1]))
            
            # Total loss
            total_loss_batch = reconstruction_loss + 0.001 * latent_smoothness
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                reconstructed, _ = self.model(data)
                loss = F.mse_loss(reconstructed, data)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=20):
        """
        Train the autoencoder with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Early stopping patience
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training SHM Autoencoder on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_shm_autoencoder.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}: Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_shm_autoencoder.pth'))
        print(f'Training completed. Best validation loss: {best_val_loss:.6f}')
    
    def get_reconstruction_errors(self, data_loader):
        """Calculate reconstruction errors for anomaly detection."""
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                reconstructed, _ = self.model(data)
                
                # Calculate per-sample reconstruction error
                errors = torch.mean((data - reconstructed) ** 2, dim=1)
                reconstruction_errors.extend(errors.cpu().numpy())
        
        return np.array(reconstruction_errors)

# Prepare data for autoencoder training
def prepare_autoencoder_data(feature_matrix, train_ratio=0.8, batch_size=32):
    """
    Prepare data for autoencoder training.
    
    Args:
        feature_matrix: Matrix of extracted features
        train_ratio: Ratio of data for training
        batch_size: Batch size for training
        
    Returns:
        tuple: (train_loader, val_loader, scaler)
    """
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    
    # Split data
    n_samples = len(normalized_features)
    n_train = int(n_samples * train_ratio)
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_data = normalized_features[train_indices]
    val_data = normalized_features[val_indices]
    
    # Convert to PyTorch tensors and create data loaders
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)
    
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    val_dataset = torch.utils.data.TensorDataset(val_tensor)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, scaler

# Train autoencoder on bridge monitoring data
print("\nTraining Autoencoder for Bridge Health Monitoring...")
print("=" * 55)

# Use the feature matrix from clustering section
train_loader, val_loader, autoencoder_scaler = prepare_autoencoder_data(
    feature_matrix, train_ratio=0.8, batch_size=16
)

# Initialize model and trainer
input_dim = feature_matrix.shape[1]
autoencoder = SHMAutoencoder(
    input_dim=input_dim,
    latent_dim=8,
    hidden_dims=[32, 16],
    dropout_rate=0.1
)

trainer = SHMAutoencoderTrainer(autoencoder, learning_rate=0.001)

# Train the model
trainer.train(train_loader, val_loader, epochs=100, early_stopping_patience=15)

print(f"\nAutoencoder training completed!")
print(f"Final training loss: {trainer.train_losses[-1]:.6f}")
print(f"Final validation loss: {trainer.val_losses[-1]:.6f}")
```

### 7.4.4 Autoencoder-Based Anomaly Detection

Now let's implement comprehensive anomaly detection using the trained autoencoder:

```python
def autoencoder_anomaly_detection(model, trainer, test_features, test_labels, scaler, threshold_percentile=95):
    """
    Perform anomaly detection using trained autoencoder.
    
    Args:
        model: Trained autoencoder model
        trainer: Trainer object with reconstruction error calculation
        test_features: Test feature matrix
        test_labels: True labels for test data
        scaler: Fitted StandardScaler
        threshold_percentile: Percentile for anomaly threshold
        
    Returns:
        dict: Anomaly detection results
    """
    # Normalize test features
    normalized_test = scaler.transform(test_features)
    
    # Create test data loader
    test_tensor = torch.FloatTensor(normalized_test)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Calculate reconstruction errors
    reconstruction_errors = trainer.get_reconstruction_errors(test_loader)
    
    # Determine threshold from training data (using validation data as proxy)
    val_errors = trainer.get_reconstruction_errors(val_loader)
    threshold = np.percentile(val_errors, threshold_percentile)
    
    # Detect anomalies
    anomaly_flags = reconstruction_errors > threshold
    
    # Calculate performance metrics
    true_anomalies = np.array([label == 'Damaged' for label in test_labels])
    
    true_positives = np.sum(anomaly_flags & true_anomalies)
    false_positives = np.sum(anomaly_flags & ~true_anomalies)
    false_negatives = np.sum(~anomaly_flags & true_anomalies)
    true_negatives = np.sum(~anomaly_flags & ~true_anomalies)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(test_labels)
    
    results = {
        'reconstruction_errors': reconstruction_errors,
        'threshold': threshold,
        'anomaly_flags': anomaly_flags,
        'true_anomalies': true_anomalies,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }
    
    return results

# Test autoencoder anomaly detection
print("\nTesting Autoencoder-Based Anomaly Detection...")
print("=" * 50)

# Use the same test data from clustering section
autoencoder_results = autoencoder_anomaly_detection(
    autoencoder, trainer, test_features, test_labels, autoencoder_scaler
)

print(f"Autoencoder Anomaly Detection Results:")
print(f"Detection threshold: {autoencoder_results['threshold']:.6f}")
print(f"Anomalies detected: {np.sum(autoencoder_results['anomaly_flags'])} out of {len(test_features)}")
print(f"\nPerformance Metrics:")
print(f"Precision: {autoencoder_results['precision']:.3f}")
print(f"Recall: {autoencoder_results['recall']:.3f}")
print(f"F1-Score: {autoencoder_results['f1_score']:.3f}")
print(f"Accuracy: {autoencoder_results['accuracy']:.3f}")
```

### 7.4.5 Comprehensive Autoencoder Visualization

Let's create comprehensive visualizations to understand the autoencoder's behavior:

```python
def visualize_autoencoder_results(trainer, autoencoder_results, test_features, test_labels):
    """
    Create comprehensive visualization of autoencoder training and anomaly detection results.
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Training History', 'Reconstruction Error Distribution',
                       'Anomaly Detection Results', 'Latent Space Visualization',
                       'Performance Metrics', 'Reconstruction Examples'],
        specs=[[{"colspan": 2}, None],
               [{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Training history
    epochs = range(1, len(trainer.train_losses) + 1)
    fig.add_trace(
        go.Scatter(x=epochs, y=trainer.train_losses, 
                  mode='lines', name='Training Loss',
                  line=dict(color='#e74c3c', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=trainer.val_losses,
                  mode='lines', name='Validation Loss',
                  line=dict(color='#3498db', width=2)),
        row=1, col=1
    )
    
    # Reconstruction error distribution
    healthy_errors = autoencoder_results['reconstruction_errors'][~autoencoder_results['true_anomalies']]
    damaged_errors = autoencoder_results['reconstruction_errors'][autoencoder_results['true_anomalies']]
    
    fig.add_trace(
        go.Histogram(x=healthy_errors, name='Healthy', 
                    opacity=0.7, nbinsx=20,
                    marker_color='#2ecc71'),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=damaged_errors, name='Damaged',
                    opacity=0.7, nbinsx=20,
                    marker_color='#e74c3c'),
        row=2, col=1
    )
    
    # Add threshold line
    fig.add_vline(x=autoencoder_results['threshold'], line_dash="dash", 
                  line_color="black", annotation_text="Threshold",
                  row=2, col=1)
    
    # Anomaly detection scatter plot
    colors = ['red' if flag else 'blue' for flag in autoencoder_results['anomaly_flags']]
    symbols = ['triangle-up' if label == 'Damaged' else 'circle' for label in test_labels]
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(autoencoder_results['reconstruction_errors']))),
            y=autoencoder_results['reconstruction_errors'],
            mode='markers',
            marker=dict(color=colors, symbol=symbols, size=8),
            name='Anomaly Detection',
            text=[f'True: {label}, Detected: {"Anomaly" if flag else "Normal"}' 
                  for label, flag in zip(test_labels, autoencoder_results['anomaly_flags'])],
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.add_hline(y=autoencoder_results['threshold'], line_dash="dash", 
                  line_color="red", row=2, col=2)
    
    # Performance metrics
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    values = [autoencoder_results['precision'], autoencoder_results['recall'],
              autoencoder_results['f1_score'], autoencoder_results['accuracy']]
    colors_metrics = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    fig.add_trace(
        go.Bar(x=metrics, y=values, marker_color=colors_metrics,
               text=[f'{v:.3f}' for v in values], textposition='auto',
               name='Performance', showlegend=False),
        row=3, col=1
    )
    
    # Latent space visualization (2D projection using PCA)
    with torch.no_grad():
        # Get latent representations
        test_tensor = torch.FloatTensor(autoencoder_scaler.transform(test_features))
        test_dataset = torch.utils.data.TensorDataset(test_tensor)
        test_loader_viz = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_features), shuffle=False)
        
        for data in test_loader_viz:
            data = data.to(trainer.device)
            _, latent = autoencoder(data)
            latent_np = latent.cpu().numpy()
            
            # Apply PCA for 2D visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_np)
            
            # Plot latent space
            healthy_mask = ~autoencoder_results['true_anomalies']
            
            fig.add_trace(
                go.Scatter(x=latent_2d[healthy_mask, 0], y=latent_2d[healthy_mask, 1],
                          mode='markers', marker=dict(color='blue', size=8),
                          name='Healthy (Latent)', showlegend=False),
                row=3, col=2
            )
            fig.add_trace(
                go.Scatter(x=latent_2d[~healthy_mask, 0], y=latent_2d[~healthy_mask, 1],
                          mode='markers', marker=dict(color='red', size=8),
                          name='Damaged (Latent)', showlegend=False),
                row=3, col=2
            )
            break
    
    # Update layout
    fig.update_layout(
        title_text="Autoencoder-Based Anomaly Detection for Bridge Health Monitoring",
        title_font_size=16,
        height=1000,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_xaxes(title_text="Reconstruction Error", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Sample Index", row=2, col=2)
    fig.update_yaxes(title_text="Reconstruction Error", row=2, col=2)
    fig.update_yaxes(title_text="Score", row=3, col=1)
    fig.update_xaxes(title_text="First Principal Component", row=3, col=2)
    fig.update_yaxes(title_text="Second Principal Component", row=3, col=2)
    
    return fig

# Create comprehensive visualization
autoencoder_viz = visualize_autoencoder_results(trainer, autoencoder_results, test_features, test_labels)
autoencoder_viz.show()

# Compare autoencoder vs k-means performance
print(f"\nComparison: Autoencoder vs k-means Clustering")
print("=" * 50)
print(f"{'Metric':<15} {'k-means':<10} {'Autoencoder':<12} {'Improvement':<12}")
print("-" * 50)

kmeans_precision = precision  # From k-means section
kmeans_recall = recall
kmeans_f1 = f1_score

ae_precision = autoencoder_results['precision']
ae_recall = autoencoder_results['recall']
ae_f1 = autoencoder_results['f1_score']

print(f"{'Precision':<15} {kmeans_precision:<10.3f} {ae_precision:<12.3f} {(ae_precision-kmeans_precision):<+12.3f}")
print(f"{'Recall':<15} {kmeans_recall:<10.3f} {ae_recall:<12.3f} {(ae_recall-kmeans_recall):<+12.3f}")
print(f"{'F1-Score':<15} {kmeans_f1:<10.3f} {ae_f1:<12.3f} {(ae_f1-kmeans_f1):<+12.3f}")
```

This autoencoder implementation demonstrates the power of deep unsupervised learning for structural health monitoring. The autoencoder learns to compress and reconstruct normal structural behavior patterns, making it highly effective at detecting anomalies that deviate from these learned patterns. The reconstruction error serves as a robust anomaly score that can identify subtle changes in structural behavior that might indicate damage or sensor malfunctions.

---

## 7.5 Generative Adversarial Networks (GANs) for Data Augmentation

### 7.5.1 Motivation for GANs in SHM

Structural Health Monitoring (SHM) of civil structures has been constantly evolving with novel methods, advancements in data science, and more accessible technology. One of the most significant challenges in applying machine learning to SHM is the scarcity of labeled damage data. Although supervised methods have been proven to be effective for detecting data anomalies, two unresolved challenges reduce the accuracy of anomaly detection: (1) the class imbalance and (2) incompleteness of anomalous patterns of training dataset.

Generative Adversarial Networks offer a revolutionary solution to this problem by generating synthetic data that closely resembles real structural monitoring data. In SHM applications, GANs serve several critical purposes:

**Data Augmentation for Rare Damage Events**: The authors concluded that the data augmentation achieved improvement in classification results when the classifier is trained with augmented data over when it is trained with fewer data points. GANs can generate synthetic damage scenarios that are too rare or dangerous to observe in real structures.

**Class Imbalance Mitigation**: Most SHM datasets are heavily skewed toward healthy structural states. GANs can generate synthetic damage data to balance datasets and improve classifier performance.

**Scenario Simulation**: GANs can generate realistic structural responses under various operational conditions, environmental effects, and damage scenarios that might not be present in the original dataset.

**Combined Anomaly Detection**: Mao et al. used Generative Adversarial Networks (GAN) combined with autoencoders to identify anomalies. The raw time series from the SHM system were transformed to Gramian Angular Field images, and two datasets from a full-scale bridge were utilized to validate the proposed methodology.

### 7.5.2 Mathematical Foundation of GANs

A Generative Adversarial Network consists of two neural networks competing in a minimax game: a generator $G$ that creates synthetic data, and a discriminator $D$ that distinguishes between real and synthetic data.

**Generator Network**
The generator maps random noise $\mathbf{z} \sim p_z(\mathbf{z})$ to synthetic data:

$$\mathbf{x}_{fake} = G(\mathbf{z}; \theta_g) \tag{7.15}$$

where $\theta_g$ represents the generator parameters.

**Discriminator Network**
The discriminator outputs the probability that input data is real:

$$D(\mathbf{x}; \theta_d) \in [0, 1] \tag{7.16}$$

where $\theta_d$ represents the discriminator parameters.

**Adversarial Loss Function**
The training objective is a minimax game:

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))] \tag{7.17}$$

**Generator Loss**
The generator aims to fool the discriminator:

$\mathcal{L}_G = -\mathbb{E}_{\mathbf{z} \sim p_z(\mathbf{z})}[\log D(G(\mathbf{z}))] \tag{7.18}$

**Discriminator Loss**
The discriminator aims to correctly classify real and fake data:

$\mathcal{L}_D = -\mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})}[\log D(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p_z(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))] \tag{7.19}$

### 7.5.3 SHM-Specific GAN Implementation

Let's implement a specialized GAN architecture designed for generating realistic structural health monitoring data:

```python
class SHMGenerator(nn.Module):
    """
    Generator network for creating synthetic SHM feature data.
    Designed to generate realistic bridge monitoring scenarios.
    """
    
    def __init__(self, noise_dim=100, feature_dim=15, hidden_dims=[128, 256, 128]):
        """
        Initialize SHM data generator.
        
        Args:
            noise_dim (int): Dimension of input noise vector
            feature_dim (int): Dimension of output feature vector
            hidden_dims (list): Hidden layer dimensions
        """
        super(SHMGenerator, self).__init__()
        
        layers = []
        input_dim = noise_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        # Output layer with Tanh activation (normalized features)
        layers.extend([
            nn.Linear(input_dim, feature_dim),
            nn.Tanh()  # Output in [-1, 1] range
        ])
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, noise):
        """Generate synthetic features from noise."""
        return self.model(noise)

class SHMDiscriminator(nn.Module):
    """
    Discriminator network for distinguishing real from synthetic SHM data.
    """
    
    def __init__(self, feature_dim=15, hidden_dims=[128, 64, 32]):
        """
        Initialize SHM data discriminator.
        
        Args:
            feature_dim (int): Dimension of input feature vector
            hidden_dims (list): Hidden layer dimensions
        """
        super(SHMDiscriminator, self).__init__()
        
        layers = []
        input_dim = feature_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, features):
        """Classify features as real or fake."""
        return self.model(features)

class SHMGANTrainer:
    """
    Training framework for SHM GAN with specialized objectives.
    """
    
    def __init__(self, generator, discriminator, lr_g=0.0002, lr_d=0.0002):
        """
        Initialize GAN trainer.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            lr_g: Learning rate for generator
            lr_d: Learning rate for discriminator
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.d_real_acc = []
        self.d_fake_acc = []
        
    def train_discriminator(self, real_data, batch_size):
        """Train discriminator for one step."""
        self.optimizer_d.zero_grad()
        
        # Train on real data
        real_labels = torch.ones(batch_size, 1).to(self.device)
        real_output = self.discriminator(real_data)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Train on fake data
        noise = torch.randn(batch_size, self.generator.model[0].in_features).to(self.device)
        fake_data = self.generator(noise).detach()  # Detach to avoid training generator
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        fake_output = self.discriminator(fake_data)
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        # Combined loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_d.step()
        
        # Calculate accuracies
        d_real_acc = (real_output > 0.5).float().mean().item()
        d_fake_acc = (fake_output < 0.5).float().mean().item()
        
        return d_loss.item(), d_real_acc, d_fake_acc
    
    def train_generator(self, batch_size):
        """Train generator for one step."""
        self.optimizer_g.zero_grad()
        
        # Generate fake data
        noise = torch.randn(batch_size, self.generator.model[0].in_features).to(self.device)
        fake_data = self.generator(noise)
        
        # Try to fool discriminator (use real labels)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(fake_output, real_labels)
        
        g_loss.backward()
        self.optimizer_g.step()
        
        return g_loss.item()
    
    def train(self, data_loader, epochs=100, d_steps=1, g_steps=1):
        """
        Train the GAN.
        
        Args:
            data_loader: Training data loader
            epochs: Number of training epochs
            d_steps: Discriminator training steps per iteration
            g_steps: Generator training steps per iteration
        """
        print(f"Training SHM GAN on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
        
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_d_real_acc = 0.0
            epoch_d_fake_acc = 0.0
            
            for i, (real_data,) in enumerate(data_loader):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Train discriminator
                for _ in range(d_steps):
                    d_loss, d_real_acc, d_fake_acc = self.train_discriminator(real_data, batch_size)
                    epoch_d_loss += d_loss
                    epoch_d_real_acc += d_real_acc
                    epoch_d_fake_acc += d_fake_acc
                
                # Train generator
                for _ in range(g_steps):
                    g_loss = self.train_generator(batch_size)
                    epoch_g_loss += g_loss
            
            # Average losses
            num_batches = len(data_loader)
            avg_g_loss = epoch_g_loss / (num_batches * g_steps)
            avg_d_loss = epoch_d_loss / (num_batches * d_steps)
            avg_d_real_acc = epoch_d_real_acc / (num_batches * d_steps)
            avg_d_fake_acc = epoch_d_fake_acc / (num_batches * d_steps)
            
            # Record history
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            self.d_real_acc.append(avg_d_real_acc)
            self.d_fake_acc.append(avg_d_fake_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1:3d}: G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}, '
                      f'D_Real_Acc: {avg_d_real_acc:.3f}, D_Fake_Acc: {avg_d_fake_acc:.3f}')
        
        print("GAN training completed!")
    
    def generate_samples(self, n_samples):
        """Generate synthetic samples."""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.generator.model[0].in_features).to(self.device)
            synthetic_data = self.generator(noise).cpu().numpy()
        return synthetic_data

# Prepare data for GAN training
def prepare_gan_data(feature_matrix, batch_size=32):
    """
    Prepare data for GAN training.
    Only use healthy data for generation.
    """
    # Use only first 80 samples (healthy operational states)
    healthy_features = feature_matrix[:80]  # Assuming first 80 are healthy
    
    # Normalize to [-1, 1] range (suitable for Tanh generator output)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(healthy_features)
    
    # Convert to tensors
    tensor_data = torch.FloatTensor(normalized_features)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return data_loader, scaler

# Train GAN on bridge monitoring data
print("\nTraining GAN for Bridge Data Augmentation...")
print("=" * 50)

# Prepare data
gan_data_loader, gan_scaler = prepare_gan_data(feature_matrix, batch_size=16)

# Initialize GAN
feature_dim = feature_matrix.shape[1]
generator = SHMGenerator(noise_dim=100, feature_dim=feature_dim, hidden_dims=[128, 256, 128])
discriminator = SHMDiscriminator(feature_dim=feature_dim, hidden_dims=[128, 64, 32])

# Initialize trainer
gan_trainer = SHMGANTrainer(generator, discriminator, lr_g=0.0002, lr_d=0.0002)

# Train GAN
gan_trainer.train(gan_data_loader, epochs=200, d_steps=1, g_steps=1)

# Generate synthetic samples
n_synthetic = 100
synthetic_features = gan_trainer.generate_samples(n_synthetic)

# Denormalize synthetic data
synthetic_features_original = gan_scaler.inverse_transform(synthetic_features)

print(f"\nGenerated {n_synthetic} synthetic bridge monitoring samples")
print(f"Synthetic data shape: {synthetic_features_original.shape}")
```

### 7.5.4 GAN Results Visualization and Analysis

```python
def visualize_gan_results(gan_trainer, real_features, synthetic_features, feature_names):
    """
    Create comprehensive visualization of GAN training and generation results.
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Training History (Losses)', 'Training History (Accuracies)',
                       'Real vs Synthetic Data Distribution', 'Feature Comparison',
                       'Principal Component Analysis', 'Generated Sample Quality'],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "box"}]]
    )
    
    # Training losses
    epochs = range(1, len(gan_trainer.g_losses) + 1)
    fig.add_trace(
        go.Scatter(x=epochs, y=gan_trainer.g_losses,
                  mode='lines', name='Generator Loss',
                  line=dict(color='#e74c3c', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=gan_trainer.d_losses,
                  mode='lines', name='Discriminator Loss',
                  line=dict(color='#3498db', width=2)),
        row=1, col=1
    )
    
    # Training accuracies
    fig.add_trace(
        go.Scatter(x=epochs, y=gan_trainer.d_real_acc,
                  mode='lines', name='D Real Accuracy',
                  line=dict(color='#2ecc71', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=gan_trainer.d_fake_acc,
                  mode='lines', name='D Fake Accuracy',
                  line=dict(color='#f39c12', width=2)),
        row=1, col=2
    )
    
    # Data distribution comparison (using first feature as example)
    fig.add_trace(
        go.Histogram(x=real_features[:, 0], name='Real Data',
                    opacity=0.7, nbinsx=20, marker_color='#3498db'),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(x=synthetic_features[:, 0], name='Synthetic Data',
                    opacity=0.7, nbinsx=20, marker_color='#e74c3c'),
        row=2, col=1
    )
    
    # Feature comparison (mean values)
    real_means = np.mean(real_features, axis=0)
    synthetic_means = np.mean(synthetic_features, axis=0)
    
    x_features = list(range(len(feature_names)))
    fig.add_trace(
        go.Bar(x=x_features, y=real_means, name='Real Mean',
               marker_color='#3498db', opacity=0.7),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=x_features, y=synthetic_means, name='Synthetic Mean',
               marker_color='#e74c3c', opacity=0.7),
        row=2, col=2
    )
    
    # PCA visualization
    from sklearn.decomposition import PCA
    
    # Combine real and synthetic data for PCA
    combined_data = np.vstack([real_features, synthetic_features])
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_data)
    
    n_real = len(real_features)
    real_pca = combined_pca[:n_real]
    synthetic_pca = combined_pca[n_real:]
    
    fig.add_trace(
        go.Scatter(x=real_pca[:, 0], y=real_pca[:, 1],
                  mode='markers', name='Real Data (PCA)',
                  marker=dict(color='#3498db', size=6, opacity=0.7)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=synthetic_pca[:, 0], y=synthetic_pca[:, 1],
                  mode='markers', name='Synthetic Data (PCA)',
                  marker=dict(color='#e74c3c', size=6, opacity=0.7)),
        row=3, col=1
    )
    
    # Quality metrics (box plots for feature distributions)
    # Select top 5 most important features for visualization
    feature_indices = [0, 1, 2, 3, 4]  # First 5 features
    
    for i, feat_idx in enumerate(feature_indices):
        fig.add_trace(
            go.Box(y=real_features[:, feat_idx], name=f'Real F{feat_idx}',
                  marker_color='#3498db', showlegend=False),
            row=3, col=2
        )
        fig.add_trace(
            go.Box(y=synthetic_features[:, feat_idx], name=f'Synth F{feat_idx}',
                  marker_color='#e74c3c', showlegend=False),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="GAN-Based Data Augmentation for Bridge Health Monitoring",
        title_font_size=16,
        height=1000,
        showlegend=True
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.update_xaxes(title_text="Feature Value", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Feature Index", row=2, col=2)
    fig.update_yaxes(title_text="Mean Value", row=2, col=2)
    fig.update_xaxes(title_text="First Principal Component", row=3, col=1)
    fig.update_yaxes(title_text="Second Principal Component", row=3, col=1)
    fig.update_yaxes(title_text="Feature Value", row=3, col=2)
    
    return fig

# Create feature names list
feature_names = list(feature_extractor.extract_all_features(signals[0]).keys())

# Use first 80 samples as "real" healthy data for comparison
real_features_for_comparison = feature_matrix[:80]

# Create GAN visualization
gan_viz = visualize_gan_results(gan_trainer, real_features_for_comparison, 
                               synthetic_features_original, feature_names)
gan_viz.show()

# Quantitative evaluation of synthetic data quality
def evaluate_synthetic_data_quality(real_data, synthetic_data, feature_names):
    """
    Quantitatively evaluate the quality of synthetic data.
    """
    print(f"\nSynthetic Data Quality Assessment:")
    print("=" * 45)
    
    # Statistical similarity metrics
    from scipy import stats
    
    print(f"{'Feature':<20} {'KS-Test p-value':<15} {'Mean Diff':<12} {'Std Diff':<12}")
    print("-" * 60)
    
    ks_pvalues = []
    mean_diffs = []
    std_diffs = []
    
    for i, feature_name in enumerate(feature_names):
        # Kolmogorov-Smirnov test for distribution similarity
        ks_stat, ks_pvalue = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
        
        # Mean and standard deviation differences
        mean_diff = abs(np.mean(real_data[:, i]) - np.mean(synthetic_data[:, i]))
        std_diff = abs(np.std(real_data[:, i]) - np.std(synthetic_data[:, i]))
        
        ks_pvalues.append(ks_pvalue)
        mean_diffs.append(mean_diff)
        std_diffs.append(std_diff)
        
        # Display top 10 features
        if i < 10:
            print(f"{feature_name[:19]:<20} {ks_pvalue:<15.4f} {mean_diff:<12.4f} {std_diff:<12.4f}")
    
    # Overall quality metrics
    avg_ks_pvalue = np.mean(ks_pvalues)
    avg_mean_diff = np.mean(mean_diffs)
    avg_std_diff = np.mean(std_diffs)
    
    print(f"\nOverall Quality Metrics:")
    print(f"Average KS-test p-value: {avg_ks_pvalue:.4f} (>0.05 indicates good similarity)")
    print(f"Average mean difference: {avg_mean_diff:.4f}")
    print(f"Average std difference: {avg_std_diff:.4f}")
    
    # Correlation analysis
    real_corr = np.corrcoef(real_data.T)
    synthetic_corr = np.corrcoef(synthetic_data.T)
    correlation_similarity = np.corrcoef(real_corr.flatten(), synthetic_corr.flatten())[0, 1]
    
    print(f"Correlation structure similarity: {correlation_similarity:.4f}")
    
    return {
        'ks_pvalues': ks_pvalues,
        'mean_diffs': mean_diffs,
        'std_diffs': std_diffs,
        'avg_ks_pvalue': avg_ks_pvalue,
        'correlation_similarity': correlation_similarity
    }

# Evaluate synthetic data quality
quality_metrics = evaluate_synthetic_data_quality(
    real_features_for_comparison, synthetic_features_original, feature_names
)
```

### 7.5.5 Data Augmentation Application

Now let's demonstrate how GAN-generated data can improve anomaly detection performance:

```python
def demonstrate_data_augmentation_benefit(original_data, synthetic_data, test_features, test_labels):
    """
    Demonstrate the benefit of GAN-based data augmentation for anomaly detection.
    """
    print(f"\nDemonstrating Data Augmentation Benefits:")
    print("=" * 50)
    
    # Scenario 1: Train with original data only
    print("Scenario 1: Training with original data only")
    original_kmeans = SHMKMeansCluster(n_clusters=5, random_state=42)
    original_kmeans.fit(original_data)
    
    # Test anomaly detection
    original_anomalies, original_distances, original_threshold = original_kmeans.detect_anomalies(
        test_features, threshold_percentile=95
    )
    
    # Calculate metrics
    true_anomalies = np.array([label == 'Damaged' for label in test_labels])
    original_tp = np.sum(original_anomalies & true_anomalies)
    original_fp = np.sum(original_anomalies & ~true_anomalies)
    original_fn = np.sum(~original_anomalies & true_anomalies)
    original_precision = original_tp / (original_tp + original_fp) if (original_tp + original_fp) > 0 else 0
    original_recall = original_tp / (original_tp + original_fn) if (original_tp + original_fn) > 0 else 0
    original_f1 = 2 * (original_precision * original_recall) / (original_precision + original_recall) if (original_precision + original_recall) > 0 else 0
    
    print(f"  Training samples: {len(original_data)}")
    print(f"  Precision: {original_precision:.3f}, Recall: {original_recall:.3f}, F1: {original_f1:.3f}")
    
    # Scenario 2: Train with augmented data (original + synthetic)
    print("\nScenario 2: Training with augmented data (original + synthetic)")
    augmented_data = np.vstack([original_data, synthetic_data])
    
    augmented_kmeans = SHMKMeansCluster(n_clusters=5, random_state=42)
    augmented_kmeans.fit(augmented_data)
    
    # Test anomaly detection
    augmented_anomalies, augmented_distances, augmented_threshold = augmented_kmeans.detect_anomalies(
        test_features, threshold_percentile=95
    )
    
    # Calculate metrics
    augmented_tp = np.sum(augmented_anomalies & true_anomalies)
    augmented_fp = np.sum(augmented_anomalies & ~true_anomalies)
    augmented_fn = np.sum(~augmented_anomalies & true_anomalies)
    augmented_precision = augmented_tp / (augmented_tp + augmented_fp) if (augmented_tp + augmented_fp) > 0 else 0
    augmented_recall = augmented_tp / (augmented_tp + augmented_fn) if (augmented_tp + augmented_fn) > 0 else 0
    augmented_f1 = 2 * (augmented_precision * augmented_recall) / (augmented_precision + augmented_recall) if (augmented_precision + augmented_recall) > 0 else 0
    
    print(f"  Training samples: {len(augmented_data)}")
    print(f"  Precision: {augmented_precision:.3f}, Recall: {augmented_recall:.3f}, F1: {augmented_f1:.3f}")
    
    # Improvement analysis
    print(f"\nImprovement Analysis:")
    precision_improvement = augmented_precision - original_precision
    recall_improvement = augmented_recall - original_recall
    f1_improvement = augmented_f1 - original_f1
    
    print(f"  Precision improvement: {precision_improvement:+.3f}")
    print(f"  Recall improvement: {recall_improvement:+.3f}")
    print(f"  F1-score improvement: {f1_improvement:+.3f}")
    
    return {
        'original': {'precision': original_precision, 'recall': original_recall, 'f1': original_f1},
        'augmented': {'precision': augmented_precision, 'recall': augmented_recall, 'f1': augmented_f1},
        'improvement': {'precision': precision_improvement, 'recall': recall_improvement, 'f1': f1_improvement}
    }

# Demonstrate augmentation benefits
augmentation_results = demonstrate_data_augmentation_benefit(
    real_features_for_comparison, synthetic_features_original, test_features, test_labels
)

# Create comparison visualization
def create_augmentation_comparison_plot(augmentation_results):
    """Create visualization comparing original vs augmented training results."""
    
    fig = go.Figure()
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    original_values = [augmentation_results['original']['precision'],
                      augmentation_results['original']['recall'],
                      augmentation_results['original']['f1']]
    augmented_values = [augmentation_results['augmented']['precision'],
                       augmentation_results['augmented']['recall'],
                       augmentation_results['augmented']['f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig.add_trace(go.Bar(
        x=[m + ' (Original)' for m in metrics],
        y=original_values,
        name='Original Data Only',
        marker_color='#3498db',
        text=[f'{v:.3f}' for v in original_values],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=[m + ' (Augmented)' for m in metrics],
        y=augmented_values,
        name='With GAN Augmentation',
        marker_color='#e74c3c',
        text=[f'{v:.3f}' for v in augmented_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Impact of GAN-Based Data Augmentation on Anomaly Detection Performance',
        title_font_size=16,
        xaxis_title='Metrics',
        yaxis_title='Score',
        barmode='group',
        height=500,
        showlegend=True
    )
    
    return fig

augmentation_comparison = create_augmentation_comparison_plot(augmentation_results)
augmentation_comparison.show()
```

---

## 7.6 Summary and Comparative Analysis

### 7.6.1 Method Comparison

Let's create a comprehensive comparison of all three machine learning approaches covered in this chapter:

```python
def comprehensive_method_comparison():
    """
    Create comprehensive comparison of k-means, autoencoders, and GANs for SHM.
    """
    # Summary table data
    methods = ['k-means Clustering', 'Autoencoders', 'GANs + Augmentation']
    
    # Performance metrics (from previous evaluations)
    precision_scores = [
        precision,  # k-means precision from earlier
        autoencoder_results['precision'],
        augmentation_results['augmented']['precision']
    ]
    
    recall_scores = [
        recall,  # k-means recall from earlier  
        autoencoder_results['recall'],
        augmentation_results['augmented']['recall']
    ]
    
    f1_scores = [
        f1_score,  # k-means f1 from earlier
        autoencoder_results['f1_score'],
        augmentation_results['augmented']['f1']
    ]
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Method': methods,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-Score': f1_scores,
        'Training Time': ['Fast', 'Moderate', 'Slow'],
        'Interpretability': ['High', 'Moderate', 'Low'],
        'Data Requirements': ['Low', 'Moderate', 'High'],
        'Anomaly Sensitivity': ['Moderate', 'High', 'High']
    })
    
    print("Comprehensive Method Comparison for SHM Applications")
    print("=" * 65)
    print(comparison_df.to_string(index=False, float_format='%.3f'))
    
    # Visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Performance Metrics Comparison', 'Method Characteristics',
                       'Computational Requirements', 'Application Suitability'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "radar"}, {"type": "table"}]]
    )
    
    # Performance metrics
    x_metrics = ['Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, method in enumerate(methods):
        values = [precision_scores[i], recall_scores[i], f1_scores[i]]
        fig.add_trace(
            go.Bar(x=x_metrics, y=values, name=method, 
                  marker_color=colors[i], opacity=0.8),
            row=1, col=1
        )
    
    # Method characteristics (normalized scores 1-5)
    characteristics = ['Speed', 'Interpretability', 'Simplicity', 'Accuracy']
    kmeans_chars = [5, 5, 5, 3]  # Fast, highly interpretable, simple, moderate accuracy
    ae_chars = [3, 3, 3, 4]      # Moderate speed, moderate interpretability, moderate complexity, good accuracy
    gan_chars = [1, 2, 1, 4]     # Slow, low interpretability, complex, good accuracy
    
    fig.add_trace(
        go.Bar(x=characteristics, y=kmeans_chars, name='k-means',
              marker_color='#3498db', opacity=0.8),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=characteristics, y=ae_chars, name='Autoencoders',
              marker_color='#e74c3c', opacity=0.8),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=characteristics, y=gan_chars, name='GANs',
              marker_color='#2ecc71', opacity=0.8),
        row=1, col=2
    )
    
    # Application suitability table
    suitability_data = [
        ['Real-time Monitoring', 'Excellent', 'Good', 'Poor'],
        ['Damage Localization', 'Good', 'Excellent', 'Good'],
        ['Rare Event Detection', 'Poor', 'Good', 'Excellent'],
        ['Multi-sensor Fusion', 'Good', 'Excellent', 'Good'],
        ['Long-term Monitoring', 'Excellent', 'Good', 'Moderate']
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=['Application', 'k-means', 'Autoencoders', 'GANs'],
                       fill_color='#34495e',
                       font=dict(color='white', size=12)),
            cells=dict(values=[[row[i] for row in suitability_data] for i in range(4)],
                      fill_color=[['#ecf0f1']*5, ['#e3f2fd']*5, ['#ffebee']*5, ['#e8f5e8']*5],
                      font=dict(size=11))
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Comprehensive Comparison of Machine Learning Methods for SHM",
        title_font_size=16,
        height=800,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Rating (1-5)", row=1, col=2)
    
    return fig, comparison_df

# Create comprehensive comparison
comparison_fig, comparison_table = comprehensive_method_comparison()
comparison_fig.show()
```

### 7.6.2 Practical Guidelines for Method Selection

Based on the analysis and results presented in this chapter, here are practical guidelines for selecting appropriate machine learning methods for different SHM scenarios:

**Choose k-means Clustering when:**
- Real-time processing is critical
- Interpretability is essential for decision-making
- Limited computational resources are available
- The goal is to understand operational patterns and behavioral states
- Multiple sensors need to be analyzed simultaneously

**Choose Autoencoders when:**
- High sensitivity to subtle anomalies is required
- Dealing with high-dimensional sensor data
- Need for automated feature learning
- Long-term continuous monitoring with gradual structural changes
- Want to combine anomaly detection with data compression

**Choose GANs when:**
- Severe class imbalance exists (rare damage events)
- Need to augment limited training datasets
- Developing robust classifiers for damage detection
- Simulating various damage scenarios for testing
- Research and development of new damage detection algorithms

---

## 7.7 Exercises

### Exercise 7.1: Feature Extraction Analysis
**Problem**: Using the provided bridge acceleration data, extract time-domain and frequency-domain features for different operational conditions (light traffic, heavy traffic, wind loading). Analyze which features are most sensitive to operational changes.

**Tasks**:
1. Generate bridge data for three different operational scenarios
2. Extract comprehensive features using the SHMFeatureExtractor class
3. Perform statistical analysis to identify the most discriminative features
4. Create visualizations comparing feature distributions across scenarios

**Solution**:
```python
# Solution for Exercise 7.1
def solve_feature_extraction_exercise():
    """Solution for feature extraction analysis exercise."""
    
    # Generate data for different scenarios
    scenarios = ['light_traffic', 'heavy_traffic', 'wind_loading']
    all_features = []
    scenario_labels = []
    
    feature_extractor = SHMFeatureExtractor(sampling_rate=200)
    
    for scenario in scenarios:
        print(f"Analyzing {scenario} scenario...")
        
        for i in range(20):  # 20 samples per scenario
            if scenario == 'light_traffic':
                time, signal = generate_realistic_bridge_data(duration=60, bridge_type='cable_stayed')
                signal *= 0.5  # Reduced amplitude
            elif scenario == 'heavy_traffic':
                time, signal = generate_realistic_bridge_data(duration=60, bridge_type='cable_stayed')
                signal *= 1.5  # Increased amplitude
            else:  # wind_loading
                time, signal = generate_realistic_bridge_data(duration=60, bridge_type='cable_stayed')
                # Add wind component
                wind_component = 0.03 * np.sin(2 * np.pi * 0.1 * time) * np.random.randn(len(time))
                signal += wind_component
            
            features = feature_extractor.extract_all_features(signal)
            all_features.append(list(features.values()))
            scenario_labels.append(scenario)
    
    # Convert to arrays
    feature_matrix = np.array(all_features)
    feature_names = list(features.keys())
    
    # Statistical analysis - ANOVA F-test for feature discrimination
    from scipy.stats import f_oneway
    
    f_scores = []
    p_values = []
    
    for i, feature_name in enumerate(feature_names):
        light_data = feature_matrix[:20, i]
        heavy_data = feature_matrix[20:40, i]
        wind_data = feature_matrix[40:60, i]
        
        f_stat, p_val = f_oneway(light_data, heavy_data, wind_data)
        f_scores.append(f_stat)
        p_values.append(p_val)
    
    # Find most discriminative features
    discriminative_indices = np.argsort(f_scores)[-5:]  # Top 5
    
    print(f"\nMost Discriminative Features:")
    print("-" * 40)
    for idx in discriminative_indices:
        print(f"{feature_names[idx]}: F-score = {f_scores[idx]:.3f}, p-value = {p_values[idx]:.6f}")
    
    return feature_matrix, scenario_labels, feature_names, discriminative_indices

# Execute solution
feature_matrix_ex1, scenario_labels_ex1, feature_names_ex1, discriminative_indices = solve_feature_extraction_exercise()
```

### Exercise 7.2: Clustering Optimization
**Problem**: Implement and compare different clustering algorithms for bridge operational state identification. Determine the optimal number of clusters using various metrics.

**Tasks**:
1. Implement k-means with different numbers of clusters (2-10)
2. Calculate clustering metrics (silhouette score, inertia, Davies-Bouldin index)
3. Use the elbow method to determine optimal cluster number
4. Compare with other clustering methods (DBSCAN, hierarchical clustering)

**Solution**:
```python
# Solution for Exercise 7.2
def solve_clustering_optimization_exercise(feature_matrix, true_labels):
    """Solution for clustering optimization exercise."""
    
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    
    # Test different numbers of clusters
    k_range = range(2, 11)
    silhouette_scores = []
    inertias = []
    davies_bouldin_scores = []
    
    print("Evaluating different numbers of clusters:")
    print("-" * 45)
    
    for k in k_range:
        # Fit k-means
        kmeans = SHMKMeansCluster(n_clusters=k, random_state=42)
        kmeans.fit(feature_matrix)
        
        # Calculate metrics
        sil_score = silhouette_score(kmeans.normalized_data, kmeans.labels)
        db_score = davies_bouldin_score(kmeans.normalized_data, kmeans.labels)
        inertia = sum([kmeans.cluster_stats[i]['inertia'] for i in range(k)])
        
        silhouette_scores.append(sil_score)
        davies_bouldin_scores.append(db_score)
        inertias.append(inertia)
        
        print(f"k={k}: Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}, Inertia={inertia:.2f}")
    
    # Find optimal k
    optimal_k_sil = k_range[np.argmax(silhouette_scores)]
    optimal_k_db = k_range[np.argmin(davies_bouldin_scores)]
    
    print(f"\nOptimal k by Silhouette Score: {optimal_k_sil}")
    print(f"Optimal k by Davies-Bouldin Score: {optimal_k_db}")
    
    # Compare with other clustering methods
    print(f"\nComparing different clustering algorithms:")
    print("-" * 45)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(StandardScaler().fit_transform(feature_matrix))
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    
    if n_clusters_dbscan > 1:
        dbscan_sil = silhouette_score(feature_matrix, dbscan_labels)
        print(f"DBSCAN: {n_clusters_dbscan} clusters, Silhouette={dbscan_sil:.3f}")
    
    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k_sil)
    hierarchical_labels = hierarchical.fit_predict(feature_matrix)
    hierarchical_sil = silhouette_score(feature_matrix, hierarchical_labels)
    print(f"Hierarchical: Silhouette={hierarchical_sil:.3f}")
    
    return optimal_k_sil, silhouette_scores, inertias

# Execute solution
optimal_k, sil_scores, inertias = solve_clustering_optimization_exercise(feature_matrix, true_labels)
```

### Exercise 7.3: Autoencoder Architecture Optimization
**Problem**: Design and compare different autoencoder architectures for bridge health monitoring data. Analyze the effect of latent dimension, network depth, and regularization on anomaly detection performance.

**Tasks**:
1. Implement autoencoders with different latent dimensions (4, 8, 16, 32)
2. Test different network depths (2, 3, 4, 5 hidden layers)
3. Compare regularization techniques (dropout, weight decay, batch normalization)
4. Evaluate reconstruction quality and anomaly detection performance

**Solution**:
```python
# Solution for Exercise 7.3
def solve_autoencoder_optimization_exercise(feature_matrix, test_features, test_labels):
    """Solution for autoencoder architecture optimization."""
    
    print("Optimizing Autoencoder Architecture:")
    print("=" * 40)
    
    # Test different latent dimensions
    latent_dims = [4, 8, 16, 32]
    results = []
    
    for latent_dim in latent_dims:
        print(f"\nTesting latent dimension: {latent_dim}")
        
        # Create model
        model = SHMAutoencoder(
            input_dim=feature_matrix.shape[1],
            latent_dim=latent_dim,
            hidden_dims=[64, 32],
            dropout_rate=0.1
        )
        
        # Train model
        train_loader, val_loader, scaler = prepare_autoencoder_data(feature_matrix)
        trainer = SHMAutoencoderTrainer(model, learning_rate=0.001)
        trainer.train(train_loader, val_loader, epochs=50, early_stopping_patience=10)
        
        # Evaluate
        ae_results = autoencoder_anomaly_detection(model, trainer, test_features, test_labels, scaler)
        
        results.append({
            'latent_dim': latent_dim,
            'precision': ae_results['precision'],
            'recall': ae_results['recall'],
            'f1_score': ae_results['f1_score'],
            'final_loss': trainer.val_losses[-1]
        })
        
        print(f"  Precision: {ae_results['precision']:.3f}")
        print(f"  Recall: {ae_results['recall']:.3f}")
        print(f"  F1-Score: {ae_results['f1_score']:.3f}")
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['f1_score'])
    print(f"\nBest Configuration:")
    print(f"Latent Dimension: {best_result['latent_dim']}")
    print(f"F1-Score: {best_result['f1_score']:.3f}")
    
    return results

# Execute solution
ae_optimization_results = solve_autoencoder_optimization_exercise(feature_matrix, test_features, test_labels)
```

### Exercise 7.4: GAN Training Stability
**Problem**: Analyze GAN training stability and implement techniques to improve convergence for structural health monitoring data generation.

**Tasks**:
1. Monitor GAN training metrics and identify mode collapse
2. Implement gradient penalty for improved stability
3. Compare different loss functions (Wasserstein, LSGAN)
4. Evaluate generated data quality using multiple metrics

**Solution**:
```python
# Solution for Exercise 7.4
class StabilizedSHMGAN:
    """Improved GAN with stability enhancements for SHM data."""
    
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Optimizers with different learning rates (stabilization technique)
        self.optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))
        
        # Training history for stability monitoring
        self.training_history = {
            'g_losses': [], 'd_losses': [], 
            'gradient_norms_g': [], 'gradient_norms_d': [],
            'discriminator_accuracy': []
        }
    
    def gradient_penalty(self, real_data, fake_data, lambda_gp=10):
        """Implement gradient penalty for improved training stability."""
        batch_size = real_data.size(0)
        
        # Random interpolation between real and fake data
        alpha = torch.rand(batch_size, 1).to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Calculate discriminator output for interpolated data
        d_interpolated = self.discriminator(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def detect_mode_collapse(self, generated_samples, threshold=0.1):
        """Detect mode collapse by analyzing diversity of generated samples."""
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist
        
        distances = pdist(generated_samples)
        mean_distance = np.mean(distances)
        
        # Mode collapse if samples are too similar
        return mean_distance < threshold, mean_distance
    
    def train_with_stability_monitoring(self, data_loader, epochs=100):
        """Train GAN with stability monitoring and early stopping."""
        
        print("Training Stabilized GAN with monitoring...")
        
        for epoch in range(epochs):
            for i, (real_data,) in enumerate(data_loader):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Train Discriminator
                self.optimizer_d.zero_grad()
                
                # Real data
                real_output = self.discriminator(real_data)
                d_loss_real = -torch.mean(real_output)  # Wasserstein loss
                
                # Fake data
                noise = torch.randn(batch_size, 100).to(self.device)
                fake_data = self.generator(noise).detach()
                fake_output = self.discriminator(fake_data)
                d_loss_fake = torch.mean(fake_output)
                
                # Gradient penalty
                gp = self.gradient_penalty(real_data, fake_data)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake + gp
                d_loss.backward()
                
                # Monitor gradient norms
                d_grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.optimizer_d.step()
                
                # Train Generator (less frequently for stability)
                if i % 5 == 0:  # Train generator every 5 discriminator updates
                    self.optimizer_g.zero_grad()
                    
                    noise = torch.randn(batch_size, 100).to(self.device)
                    fake_data = self.generator(noise)
                    fake_output = self.discriminator(fake_data)
                    
                    g_loss = -torch.mean(fake_output)  # Wasserstein loss
                    g_loss.backward()
                    
                    g_grad_norm = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    self.optimizer_g.step()
                    
                    # Record training metrics
                    self.training_history['g_losses'].append(g_loss.item())
                    self.training_history['gradient_norms_g'].append(g_grad_norm.item())
                
                self.training_history['d_losses'].append(d_loss.item())
                self.training_history['gradient_norms_d'].append(d_grad_norm.item())
                
                # Calculate discriminator accuracy
                d_acc = (torch.mean((real_output > 0).float()) + torch.mean((fake_output < 0).float())) / 2
                self.training_history['discriminator_accuracy'].append(d_acc.item())
            
            # Check for mode collapse every 10 epochs
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    test_noise = torch.randn(100, 100).to(self.device)
                    test_samples = self.generator(test_noise).cpu().numpy()
                    
                    mode_collapse, diversity = self.detect_mode_collapse(test_samples)
                    
                    print(f"Epoch {epoch+1}: G_Loss={np.mean(self.training_history['g_losses'][-10:]):.4f}, "
                          f"D_Loss={np.mean(self.training_history['d_losses'][-10:]):.4f}, "
                          f"Diversity={diversity:.4f}, Mode_Collapse={mode_collapse}")
                    
                    if mode_collapse:
                        print("Warning: Possible mode collapse detected!")

def solve_gan_stability_exercise():
    """Solution for GAN training stability exercise."""
    
    # Initialize improved GAN
    generator = SHMGenerator(noise_dim=100, feature_dim=feature_matrix.shape[1])
    discriminator = SHMDiscriminator(feature_dim=feature_matrix.shape[1])
    
    stabilized_gan = StabilizedSHMGAN(generator, discriminator)
    
    # Prepare data
    gan_data_loader, _ = prepare_gan_data(feature_matrix[:80])
    
    # Train with stability monitoring
    stabilized_gan.train_with_stability_monitoring(gan_data_loader, epochs=100)
    
    print("Stabilized GAN training completed!")
    return stabilized_gan

# Execute solution
stabilized_gan = solve_gan_stability_exercise()
```

### Exercise 7.5: Integrated SHM System
**Problem**: Design an integrated structural health monitoring system that combines all three machine learning approaches (clustering, autoencoders, GANs) for comprehensive bridge monitoring.

**Tasks**:
1. Create a hierarchical system: clustering for operational state identification, autoencoders for anomaly detection, GANs for data augmentation
2. Implement decision fusion from multiple algorithms
3. Design a real-time processing pipeline
4. Evaluate system performance under various scenarios

**Solution**:
```python
# Solution for Exercise 7.5
class IntegratedSHMSystem:
    """
    Integrated SHM system combining clustering, autoencoders, and GANs.
    """
    
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        
        # Initialize components
        self.clusterer = SHMKMeansCluster(n_clusters=5)
        self.autoencoder = SHMAutoencoder(input_dim=feature_dim, latent_dim=8)
        self.generator = SHMGenerator(feature_dim=feature_dim)
        
        # Training status
        self.is_trained = False
        
        # Decision thresholds
        self.cluster_threshold = None
        self.reconstruction_threshold = None
        
    def train_system(self, training_data, validation_data=None):
        """Train all components of the integrated system."""
        
        print("Training Integrated SHM System...")
        print("=" * 40)
        
        # 1. Train clustering for operational state identification
        print("Training clustering component...")
        self.clusterer.fit(training_data)
        
        # Set cluster-based anomaly threshold
        distances = np.sqrt(((self.clusterer.normalized_data - self.clusterer.centroids[:, np.newaxis])**2).sum(axis=2))
        min_distances = np.min(distances, axis=0)
        self.cluster_threshold = np.percentile(min_distances, 95)
        
        # 2. Train autoencoder for anomaly detection
        print("Training autoencoder component...")
        train_loader, val_loader, self.ae_scaler = prepare_autoencoder_data(training_data)
        ae_trainer = SHMAutoencoderTrainer(self.autoencoder)
        ae_trainer.train(train_loader, val_loader, epochs=100, early_stopping_patience=15)
        
        # Set reconstruction-based anomaly threshold
        reconstruction_errors = ae_trainer.get_reconstruction_errors(val_loader)
        self.reconstruction_threshold = np.percentile(reconstruction_errors, 95)
        
        # 3. Train GAN for data augmentation (if needed)
        print("Training GAN component...")
        gan_data_loader, self.gan_scaler = prepare_gan_data(training_data)
        
        discriminator = SHMDiscriminator(feature_dim=self.feature_dim)
        gan_trainer = SHMGANTrainer(self.generator, discriminator)
        gan_trainer.train(gan_data_loader, epochs=100)
        
        self.is_trained = True
        print("Integrated system training completed!")
        
    def process_real_time_data(self, new_data):
        """Process new data through the integrated system."""
        
        if not self.is_trained:
            raise ValueError("System must be trained before processing data")
        
        results = {
            'operational_state': None,
            'cluster_anomaly': False,
            'reconstruction_anomaly': False,
            'integrated_decision': None,
            'confidence_score': 0.0
        }
        
        # 1. Operational state identification using clustering
        cluster_prediction = self.clusterer.predict(new_data.reshape(1, -1))
        results['operational_state'] = cluster_prediction[0]
        
        # Check cluster-based anomaly
        distances = np.sqrt(((self.clusterer.scaler.transform(new_data.reshape(1, -1)) - 
                             self.clusterer.centroids[:, np.newaxis])**2).sum(axis=2))
        min_distance = np.min(distances)
        results['cluster_anomaly'] = min_distance > self.cluster_threshold
        
        # 2. Autoencoder-based anomaly detection
        normalized_data = self.ae_scaler.transform(new_data.reshape(1, -1))
        data_tensor = torch.FloatTensor(normalized_data)
        
        with torch.no_grad():
            reconstructed, _ = self.autoencoder(data_tensor)
            reconstruction_error = torch.mean((data_tensor - reconstructed) ** 2).item()
        
        results['reconstruction_anomaly'] = reconstruction_error > self.reconstruction_threshold
        
        # 3. Integrated decision making
        anomaly_votes = [results['cluster_anomaly'], results['reconstruction_anomaly']]
        anomaly_count = sum(anomaly_votes)
        
        if anomaly_count == 0:
            results['integrated_decision'] = 'Normal'
            results['confidence_score'] = 0.9
        elif anomaly_count == 1:
            results['integrated_decision'] = 'Suspicious'
            results['confidence_score'] = 0.6
        else:
            results['integrated_decision'] = 'Anomaly'
            results['confidence_score'] = 0.95
        
        return results
    
    def batch_process(self, data_batch):
        """Process a batch of data."""
        results = []
        for data_point in data_batch:
            result = self.process_real_time_data(data_point)
            results.append(result)
        return results
    
    def generate_augmented_data(self, n_samples=100):
        """Generate synthetic data for system testing."""
        if not self.is_trained:
            raise ValueError("System must be trained before generating data")
        
        # Generate using trained GAN
        noise = torch.randn(n_samples, 100)
        with torch.no_grad():
            synthetic_data = self.generator(noise).numpy()
        
        # Denormalize
        synthetic_data = self.gan_scaler.inverse_transform(synthetic_data)
        return synthetic_data

def solve_integrated_system_exercise():
    """Solution for integrated SHM system exercise."""
    
    print("Building Integrated SHM System")
    print("=" * 35)
    
    # Initialize system
    integrated_system = IntegratedSHMSystem(feature_dim=feature_matrix.shape[1])
    
    # Train system
    integrated_system.train_system(feature_matrix[:80])  # Use healthy data for training
    
    # Test system with various scenarios
    print("\nTesting Integrated System:")
    print("-" * 30)
    
    # Test with healthy data
    healthy_results = integrated_system.batch_process(test_features[:10])
    healthy_anomaly_rate = sum([1 for r in healthy_results if r['integrated_decision'] == 'Anomaly']) / len(healthy_results)
    
    # Test with damaged data
    damaged_results = integrated_system.batch_process(test_features[10:])
    damaged_detection_rate = sum([1 for r in damaged_results if r['integrated_decision'] == 'Anomaly']) / len(damaged_results)
    
    print(f"Healthy data anomaly rate: {healthy_anomaly_rate:.2%}")
    print(f"Damaged data detection rate: {damaged_detection_rate:.2%}")
    
    # Generate synthetic data for testing
    synthetic_data = integrated_system.generate_augmented_data(50)
    synthetic_results = integrated_system.batch_process(synthetic_data)
    synthetic_anomaly_rate = sum([1 for r in synthetic_results if r['integrated_decision'] == 'Anomaly']) / len(synthetic_results)
    
    print(f"Synthetic data anomaly rate: {synthetic_anomaly_rate:.2%}")
    
    return integrated_system

# Execute solution
integrated_system = solve_integrated_system_exercise()
```

---

## 7.8 References and Further Reading

The development of machine learning techniques for structural health monitoring builds upon extensive research across multiple disciplines. The following references provide comprehensive coverage of the theoretical foundations and practical applications discussed in this chapter:

**Foundational Machine Learning References:**

[1] Malekloo, A., Ozer, E., AlHamaydeh, M., & Girolami, M. (2022). Machine learning and structural health monitoring overview with emerging technology and high-dimensional data source highlights. *Structural Health Monitoring*, 21(4), 1906-1955.

[2] Sun, L., Shang, Z., Xia, Y., Bhowmick, S., & Nagarajaiah, S. (2020). Review of bridge structural health monitoring aided by big data and artificial intelligence: from condition assessment to damage detection. *Journal of Structural Engineering*, 146(5), 04020073.

[3] Bao, Y., Tang, Z., Li, H., & Zhang, Y. (2019). Computer vision and deep learning–based data anomaly detection method for structural health monitoring. *Structural Health Monitoring*, 18(2), 401-421.

**Clustering Applications in SHM:**

[4] Diez, A., Khoa, N. L. D., Alamdari, M. M., Wang, Y., & Chen, F. (2016). A clustering approach for structural health monitoring on bridges. *Journal of Civil Structural Health Monitoring*, 6(3), 429-445.

[5] Alamdari, M. M., Rakotoarivelo, T., & Khoa, N. L. D. (2017). A spectral-based clustering for structural health monitoring of the Sydney Harbour Bridge. *Mechanical Systems and Signal Processing*, 87, 384-400.

[6] Burrello, A., Fiori, F., Brunelli, D., & Benini, L. (2021). K-means clustering for structural health monitoring of bridges using IoT devices. *Smart Cities*, 4(1), 327-350.

**Autoencoder-Based Anomaly Detection:**

[7] Jana, D., Patil, J., Herkal, S., Nagarajaiah, S., & Duenas-Osorio, L. (2022). CNN and convolutional autoencoder (CAE) based real-time sensor fault detection, localization, and correction. *Mechanical Systems and Signal Processing*, 169, 108723.

[8] Ni, F., Zhang, J., & Noori, M. N. (2020). Deep learning for data anomaly detection and data compression of a long-span suspension bridge. *Computer-Aided Civil and Infrastructure Engineering*, 35(7), 685-700.

[9] Xiao, F., Hulsey, J. L., Chen, G. S., & Xiang, Y. (2017). Optimal static strain sensor placement for truss bridges. *International Journal of Distributed Sensor Networks*, 13(5), 1550147717707929.

**Generative Adversarial Networks for SHM:**

[10] Mao, J., Wang, H., & Spencer Jr, B. F. (2021). Toward data anomaly detection for automated structural health monitoring: Exploiting generative adversarial nets and autoencoders. *Structural Health Monitoring*, 20(4), 1609-1626.

[11] Liu, G., Niu, Y., Zhao, W., Duan, Y., & Shu, J. (2022). Data anomaly detection for structural health monitoring using a combination network of GANomaly and CNN. *Smart Structures and Systems*, 29(1), 53-62.

[12] Wang, Z., & Cha, Y. J. (2021). Unsupervised deep learning approach using a deep auto-encoder with a one-class support vector machine to detect structural damage. *Structural Health Monitoring*, 20(1), 406-425.

**Bridge-Specific Applications:**

[13] Chen, Z., Zhou, X., Wang, X., Dong, L., & Qian, Y. (2017). Deployment of a smart structural health monitoring system for long-span arch bridges: A review and a case study. *Sensors*, 17(9), 2151.

[14] Neves, A. C., González, I., Leander, J., & Karoumi, R. (2017). Structural health monitoring of bridges: a model-free ANN-based approach to damage detection. *Journal of Civil Structural Health Monitoring*, 7(5), 689-702.

[15] Li, H., & Ou, J. (2016). The state-of-the-art in structural health monitoring of cable-stayed bridges. *Journal of Civil Structural Health Monitoring*, 6(1), 43-67.

**Deep Learning Frameworks and Implementation:**

[16] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32, 8026-8037.

[17] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

**Statistical Methods and Feature Engineering:**

[18] Farrar, C. R., & Worden, K. (2012). *Structural Health Monitoring: A Machine Learning Perspective*. John Wiley & Sons.

[19] Doebling, S. W., Farrar, C. R., Prime, M. B., & Shevitz, D. W. (1996). Damage identification and health monitoring of structural and mechanical systems from changes in their vibration characteristics: a literature review. Los Alamos National Lab.(LANL), Los Alamos, NM (United States).

**Data Quality and Preprocessing:**

[20] Tang, Z., Chen, Z., Bao, Y., & Li, H. (2019). Convolutional neural network‐based data anomaly detection method using multiple information for structural health monitoring. *Structural Control and Health Monitoring*, 26(1), e2296.

[21] Bao, Y., Tang, Z., Li, H., & Zhang, Y. (2019). Computer vision and deep learning–based data anomaly detection method for structural health monitoring. *Structural Health Monitoring*, 18(2), 401-421.

**Advanced Topics and Future Directions:**

[22] Sony, S., Laventure, S., & Sadhu, A. (2019). A literature review of next‐generation smart sensing technology in structural health monitoring. *Structural Control and Health Monitoring*, 26(3), e2321.

[23] Flah, M., Nunez, I., Ben Chaabene, W., & Nehdi, M. L. (2021). Machine learning algorithms in civil structural health monitoring: a systematic review. *Archives of Computational Methods in Engineering*, 28(4), 2621-2643.

[24] Azimi, M., Eslamlou, A. D., & Pekcan, G. (2020). Data-driven structural health monitoring and damage detection through deep learning: State-of-the-art review. *Sensors*, 20(10), 2778.

These references provide comprehensive coverage of the theoretical foundations, practical implementations, and cutting-edge research in machine learning applications for structural health monitoring. Students are encouraged to explore these sources for deeper understanding of specific topics and to stay current with the rapidly evolving field of intelligent infrastructure monitoring.

---

*This chapter has provided a comprehensive introduction to machine learning techniques for structural health monitoring, with particular emphasis on bridge applications. The combination of clustering for operational state identification, autoencoders for anomaly detection, and GANs for data augmentation creates a powerful toolkit for developing intelligent monitoring systems. As the field continues to evolve, the integration of these techniques with emerging technologies such as edge computing, 5G communications, and advanced sensor networks will enable even more sophisticated and effective structural health monitoring solutions.*