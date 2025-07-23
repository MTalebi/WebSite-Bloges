# Chapter 3: Time-Domain Signal Processing for Structural Health Monitoring

*Instructor: Mohammad Talebi-Kalaleh – University of Alberta*

---

## Chapter Overview

Time-domain signal processing forms the foundation of modern structural health monitoring (SHM) systems. Unlike frequency-domain analysis, which examines signals in terms of their spectral content, time-domain processing focuses on the temporal characteristics of structural responses—amplitude variations, trends, correlations, and statistical properties that change over time. For bridge monitoring applications, time-domain analysis provides immediate insights into structural behavior under loading, enables real-time damage detection, and serves as the preprocessing foundation for all subsequent analysis techniques.

Recent systematic literature reviews have shown that time-domain feature extraction methods, combined with proper signal preprocessing, are among the most frequently deployed techniques in bridge SHM applications. This chapter emphasizes practical implementation of time-domain techniques using realistic bridge acceleration and strain data, ensuring students develop both theoretical understanding and hands-on expertise.

**Learning Objectives:**
- Master fundamental time-series analysis concepts for SHM applications
- Understand sampling theory and anti-aliasing principles for sensor data acquisition
- Implement robust data preprocessing and filtering techniques
- Apply correlation and convolution methods for structural response analysis
- Extract meaningful statistical features from bridge monitoring data
- Develop complete signal processing pipelines for real-world SHM systems

---

## 3.1 Time-Series Fundamentals for Structural Monitoring

### 3.1.1 Motivation: Why Time-Domain Analysis Matters

In structural health monitoring, we continuously measure responses such as accelerations, strains, and displacements as structures react to environmental forces (wind, traffic, seismic activity). These measurements create time-series data—sequences of values indexed by time. Understanding how these signals behave in the time domain provides critical insights:

- **Immediate damage detection**: Sudden changes in signal amplitude or pattern often indicate structural damage
- **Load identification**: The time-domain response reveals how structures react to specific loading events
- **Baseline establishment**: Statistical properties in the time domain define "normal" structural behavior
- **Real-time monitoring**: Time-domain features can be computed rapidly for continuous monitoring systems

### 3.1.2 Mathematical Foundation

A discrete-time signal $x[n]$ represents our measured structural response at sample $n$, where $n = 0, 1, 2, \ldots, N-1$ for a signal of length $N$. For bridge SHM, this could be acceleration data from accelerometers or strain measurements from strain gauges.

The fundamental relationship between continuous and discrete signals follows:

$$x[n] = x_c(nT_s)$$ 
$$(3.1)$$

where $x_c(t)$ is the continuous-time signal, $T_s$ is the sampling period, and $f_s = 1/T_s$ is the sampling frequency.

### 3.1.3 Stationarity and Non-stationarity

**Stationarity** is a crucial concept for time-series analysis. A signal is stationary if its statistical properties (mean, variance, autocorrelation) don't change over time.

For a stationary process:
- **Mean stationarity**: $E[x[n]] = \mu$ (constant)
- **Variance stationarity**: $\text{Var}[x[n]] = \sigma^2$ (constant)  
- **Autocorrelation stationarity**: $R_{xx}[k] = E[x[n]x[n+k]]$ depends only on lag $k$

$$R_{xx}[k] = \frac{1}{N-k} \sum_{n=0}^{N-k-1} x[n] x[n+k]$$
$$(3.2)$$

Most bridge response signals are **non-stationary** due to:
- Varying traffic loads
- Environmental condition changes (temperature, wind)
- Potential damage progression over time

### 3.1.4 Statistical Descriptors

Key statistical measures characterize signal behavior:

**Mean (First Moment):**
$$\mu_x = \frac{1}{N} \sum_{n=0}^{N-1} x[n]$$
$$(3.3)$$

**Variance (Second Central Moment):**
$$\sigma_x^2 = \frac{1}{N-1} \sum_{n=0}^{N-1} (x[n] - \mu_x)^2$$
$$(3.4)$$

**Skewness (Third Standardized Moment):**
$$\gamma_x = \frac{1}{N} \sum_{n=0}^{N-1} \left(\frac{x[n] - \mu_x}{\sigma_x}\right)^3$$
$$(3.5)$$

**Kurtosis (Fourth Standardized Moment):**
$$\kappa_x = \frac{1}{N} \sum_{n=0}^{N-1} \left(\frac{x[n] - \mu_x}{\sigma_x}\right)^4$$
$$(3.6)$$

**Root Mean Square (RMS):**
$$x_{\text{RMS}} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2}$$
$$(3.7)$$

These descriptors help identify changes in structural behavior, as damage often manifests as changes in statistical properties.

### 3.1.5 Visual Analysis Framework

The following flowchart illustrates the systematic approach to time-domain signal analysis in SHM:

```svg
<svg width="800" height="600" viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .box { fill: #E8F4FD; stroke: #2E86AB; stroke-width: 2; rx: 8; }
      .process { fill: #A8E6CF; stroke: #2E8B57; stroke-width: 2; rx: 8; }
      .decision { fill: #FFD3BA; stroke: #FF8C42; stroke-width: 2; rx: 8; }
      .output { fill: #F4E4C1; stroke: #E07A5F; stroke-width: 2; rx: 8; }
      .text { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 11px; text-anchor: middle; dominant-baseline: middle; }
      .arrow { stroke: #333; stroke-width: 2; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
    </marker>
  </defs>
  
  <!-- Raw Data Input -->
  <rect class="box" x="20" y="20" width="160" height="60"/>
  <text class="text" x="100" y="50">Raw Sensor Data</text>
  <text class="text" x="100" y="65" style="font-size: 9px;">(Acceleration/Strain)</text>
  
  <!-- Visual Inspection -->
  <rect class="process" x="20" y="120" width="160" height="50"/>
  <text class="text" x="100" y="145">Visual Inspection</text>
  
  <!-- Statistical Analysis -->
  <rect class="process" x="220" y="120" width="160" height="50"/>
  <text class="text" x="300" y="145">Statistical Analysis</text>
  
  <!-- Stationarity Check -->
  <rect class="decision" x="420" y="120" width="160" height="50"/>
  <text class="text" x="500" y="145">Stationarity Check</text>
  
  <!-- Statistical Features -->
  <rect class="output" x="120" y="220" width="140" height="80"/>
  <text class="text" x="190" y="240" style="font-weight: bold;">Statistical Features:</text>
  <text class="text" x="190" y="255" style="font-size: 9px;">• Mean, Variance</text>
  <text class="text" x="190" y="270" style="font-size: 9px;">• Skewness, Kurtosis</text>
  <text class="text" x="190" y="285" style="font-size: 9px;">• RMS, Peak Values</text>
  
  <!-- Trend Analysis -->
  <rect class="output" x="320" y="220" width="140" height="80"/>
  <text class="text" x="390" y="240" style="font-weight: bold;">Trend Analysis:</text>
  <text class="text" x="390" y="255" style="font-size: 9px;">• Linear/Nonlinear Trends</text>
  <text class="text" x="390" y="270" style="font-size: 9px;">• Seasonality</text>
  <text class="text" x="390" y="285" style="font-size: 9px;">• Change Points</text>
  
  <!-- Time-Local Analysis -->
  <rect class="output" x="520" y="220" width="140" height="80"/>
  <text class="text" x="590" y="240" style="font-weight: bold;">Time-Local Analysis:</text>
  <text class="text" x="590" y="255" style="font-size: 9px;">• Windowed Statistics</text>
  <text class="text" x="590" y="270" style="font-size: 9px;">• Moving Averages</text>
  <text class="text" x="590" y="285" style="font-size: 9px;">• Event Detection</text>
  
  <!-- Correlation Analysis -->
  <rect class="process" x="220" y="340" width="160" height="50"/>
  <text class="text" x="300" y="365">Correlation Analysis</text>
  
  <!-- Feature Extraction -->
  <rect class="output" x="220" y="440" width="160" height="60"/>
  <text class="text" x="300" y="465">Damage-Sensitive</text>
  <text class="text" x="300" y="480">Features</text>
  
  <!-- Arrows -->
  <line class="arrow" x1="100" y1="80" x2="100" y2="120"/>
  <line class="arrow" x1="180" y1="145" x2="220" y2="145"/>
  <line class="arrow" x1="380" y1="145" x2="420" y2="145"/>
  
  <line class="arrow" x1="190" y1="170" x2="190" y2="220"/>
  <line class="arrow" x1="390" y1="170" x2="390" y2="220"/>
  <line class="arrow" x1="590" y1="170" x2="590" y2="220"/>
  
  <line class="arrow" x1="300" y1="300" x2="300" y2="340"/>
  <line class="arrow" x1="300" y1="390" x2="300" y2="440"/>
  
</svg>
```

**Figure 3.1:** *Time-domain signal analysis framework for structural health monitoring applications*

### 3.1.6 Python Implementation: Basic Time-Series Analysis

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def generate_bridge_acceleration_data(duration=60, fs=1000, bridge_freq=2.3, 
                                    traffic_loading=True, noise_level=0.05):
    """
    Generate realistic bridge acceleration data for SHM analysis.
    
    Parameters:
    -----------
    duration : float
        Signal duration in seconds
    fs : int
        Sampling frequency in Hz
    bridge_freq : float
        Primary bridge frequency in Hz
    traffic_loading : bool
        Include traffic-induced loading
    noise_level : float
        Noise level relative to signal amplitude
    
    Returns:
    --------
    t : array
        Time vector
    acceleration : array
        Simulated acceleration signal in m/s²
    """
    
    # Time vector
    t = np.linspace(0, duration, int(fs * duration))
    N = len(t)
    
    # Primary structural response (first mode)
    primary_response = 0.02 * np.sin(2 * np.pi * bridge_freq * t) * np.exp(-0.05 * t)
    
    # Higher modes
    mode2 = 0.01 * np.sin(2 * np.pi * 7.1 * t) * np.exp(-0.08 * t)
    mode3 = 0.005 * np.sin(2 * np.pi * 12.4 * t) * np.exp(-0.12 * t)
    
    # Traffic loading simulation
    if traffic_loading:
        # Vehicle crossing events (realistic truck frequencies 0.1-5 Hz)
        vehicle_times = np.random.exponential(8, int(duration/5))  # Vehicle arrivals
        vehicle_times = np.cumsum(vehicle_times)
        vehicle_times = vehicle_times[vehicle_times < duration]
        
        traffic_response = np.zeros_like(t)
        for vehicle_time in vehicle_times:
            # Vehicle weight and speed variation
            weight_factor = np.random.uniform(0.5, 2.0)  # 0.5x to 2x typical loading
            vehicle_freq = np.random.uniform(1.5, 8.0)   # Vehicle-bridge interaction freq
            
            # Exponentially decaying response after vehicle passage
            vehicle_response = weight_factor * 0.03 * np.exp(-2 * (t - vehicle_time)) * \
                             np.sin(2 * np.pi * vehicle_freq * (t - vehicle_time))
            vehicle_response[t < vehicle_time] = 0  # Causal response
            traffic_response += vehicle_response
    else:
        traffic_response = 0
    
    # Environmental loading (wind, temperature effects)
    environmental = 0.002 * np.sin(2 * np.pi * 0.05 * t)  # Low frequency environmental
    
    # Measurement noise (realistic sensor noise)
    noise = noise_level * np.random.normal(0, 1, N)
    
    # Total acceleration signal
    acceleration = primary_response + mode2 + mode3 + traffic_response + environmental + noise
    
    return t, acceleration

def compute_statistical_features(signal, feature_names=None):
    """
    Compute comprehensive statistical features for time-domain analysis.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    feature_names : list, optional
        Custom feature names
        
    Returns:
    --------
    features_df : DataFrame
        Statistical features with descriptions
    """
    
    # Remove DC component for AC analysis
    signal_ac = signal - np.mean(signal)
    
    features = {
        'Mean': np.mean(signal),
        'Standard Deviation': np.std(signal, ddof=1),
        'Variance': np.var(signal, ddof=1),
        'RMS': np.sqrt(np.mean(signal**2)),
        'RMS (AC)': np.sqrt(np.mean(signal_ac**2)),
        'Skewness': stats.skew(signal),
        'Kurtosis': stats.kurtosis(signal),
        'Peak Value': np.max(np.abs(signal)),
        'Crest Factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
        'Shape Factor': np.sqrt(np.mean(signal**2)) / np.mean(np.abs(signal)),
        'Impulse Factor': np.max(np.abs(signal)) / np.mean(np.abs(signal)),
        'Clearance Factor': np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal))))**2,
        'Energy': np.sum(signal**2),
        'Signal Power': np.mean(signal**2)
    }
    
    # Create DataFrame with descriptions
    descriptions = {
        'Mean': 'Average value (DC component)',
        'Standard Deviation': 'Measure of signal variability',
        'Variance': 'Square of standard deviation',
        'RMS': 'Root mean square value',
        'RMS (AC)': 'AC component RMS',
        'Skewness': 'Asymmetry of distribution',
        'Kurtosis': 'Tail heaviness of distribution',
        'Peak Value': 'Maximum absolute amplitude',
        'Crest Factor': 'Peak to RMS ratio',
        'Shape Factor': 'RMS to mean ratio',
        'Impulse Factor': 'Peak to mean ratio',
        'Clearance Factor': 'Peak to square mean of square roots',
        'Energy': 'Total signal energy',
        'Signal Power': 'Average signal power'
    }
    
    features_df = pd.DataFrame({
        'Feature': list(features.keys()),
        'Value': list(features.values()),
        'Description': [descriptions[key] for key in features.keys()]
    })
    
    return features_df

# Generate realistic bridge acceleration data
t, acceleration = generate_bridge_acceleration_data(duration=60, fs=1000, 
                                                   traffic_loading=True, noise_level=0.03)

# Compute statistical features
features_df = compute_statistical_features(acceleration)

# Create comprehensive visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Bridge Acceleration Time History', 'Statistical Distribution',
                   'Autocorrelation Function', 'Rolling Statistics',
                   'Spectrogram View', 'Statistical Features'),
    specs=[[{"colspan": 2}, None],
           [{"type": "xy"}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "table"}]],
    vertical_spacing=0.08
)

# Time history plot
fig.add_trace(
    go.Scatter(x=t, y=acceleration*1000, mode='lines', name='Acceleration',
               line=dict(color='#2E86AB', width=1.5)),
    row=1, col=1
)

# Statistical distribution
fig.add_trace(
    go.Histogram(x=acceleration*1000, nbinsx=50, name='Distribution',
                marker_color='#A8E6CF', opacity=0.7),
    row=2, col=1
)

# Autocorrelation function
lags = np.arange(0, 1000)  # First 1000 lags (1 second)
autocorr = np.correlate(acceleration, acceleration, mode='full')
autocorr = autocorr[autocorr.size // 2:][:1000]
autocorr = autocorr / autocorr[0]  # Normalize

fig.add_trace(
    go.Scatter(x=lags/1000, y=autocorr, mode='lines', name='Autocorrelation',
               line=dict(color='#FF8C42', width=2)),
    row=2, col=2
)

# Rolling statistics (1-second windows)
window_size = 1000  # 1 second windows at 1000 Hz
t_windows = t[::window_size]
rolling_rms = []
rolling_std = []

for i in range(0, len(acceleration) - window_size, window_size):
    window_data = acceleration[i:i+window_size]
    rolling_rms.append(np.sqrt(np.mean(window_data**2)))
    rolling_std.append(np.std(window_data))

fig.add_trace(
    go.Scatter(x=t_windows[:len(rolling_rms)], y=np.array(rolling_rms)*1000, 
               mode='lines+markers', name='Rolling RMS',
               line=dict(color='#E07A5F', width=2)),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=t_windows[:len(rolling_std)], y=np.array(rolling_std)*1000, 
               mode='lines+markers', name='Rolling Std',
               line=dict(color='#F4A261', width=2)),
    row=3, col=1
)

# Statistical features table
fig.add_trace(
    go.Table(
        header=dict(values=['Feature', 'Value', 'Description'],
                   fill_color='#E8F4FD',
                   font=dict(color='#2E86AB', size=10),
                   align='left'),
        cells=dict(values=[features_df['Feature'], 
                          [f"{val:.6f}" for val in features_df['Value']],
                          features_df['Description']],
                  fill_color='white',
                  font=dict(color='black', size=9),
                  align='left')
    ),
    row=3, col=2
)

# Update layout
fig.update_layout(
    height=900,
    title_text="Bridge Acceleration Signal: Comprehensive Time-Domain Analysis",
    title_font_size=16,
    showlegend=True
)

# Update axes labels
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Acceleration (mm/s²)", row=1, col=1)

fig.update_xaxes(title_text="Acceleration (mm/s²)", row=2, col=1)
fig.update_yaxes(title_text="Frequency", row=2, col=1)

fig.update_xaxes(title_text="Lag (s)", row=2, col=2)
fig.update_yaxes(title_text="Autocorrelation", row=2, col=2)

fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Rolling Statistics (mm/s²)", row=3, col=1)

fig.show()

# Display statistical summary
print("\n=== BRIDGE ACCELERATION SIGNAL ANALYSIS ===")
print(f"Signal Duration: {t[-1]:.1f} seconds")
print(f"Sampling Frequency: {1/(t[1]-t[0]):.0f} Hz")
print(f"Number of Samples: {len(acceleration):,}")
print(f"\nKey Statistical Measures:")
print(f"RMS Acceleration: {np.sqrt(np.mean(acceleration**2))*1000:.3f} mm/s²")
print(f"Peak Acceleration: {np.max(np.abs(acceleration))*1000:.3f} mm/s²")
print(f"Crest Factor: {np.max(np.abs(acceleration))/np.sqrt(np.mean(acceleration**2)):.2f}")
```

This code generates realistic bridge acceleration data and performs comprehensive statistical analysis, providing the foundation for understanding time-domain signal characteristics in structural health monitoring applications.

---

## 3.2 Sampling Theory and Anti-Aliasing

### 3.2.1 The Nyquist-Shannon Sampling Theorem

The Nyquist-Shannon sampling theorem provides the fundamental principle for converting continuous analog signals to discrete digital representations without information loss. For structural monitoring, this theorem determines the minimum sampling rate required to accurately capture structural dynamics.

**Theorem Statement:** A continuous-time signal $x_c(t)$ that is band-limited to frequencies below $f_{\max}$ can be perfectly reconstructed from its samples if the sampling frequency $f_s$ satisfies:

$$f_s \geq 2f_{\max}$$
$$(3.8)$$

The frequency $f_N = f_s/2$ is called the **Nyquist frequency**, and $2f_{\max}$ is called the **Nyquist rate**.

### 3.2.2 Aliasing: The Fundamental Sampling Problem

When the sampling theorem is violated ($f_s < 2f_{\max}$), **aliasing** occurs. High-frequency components in the signal appear as false low-frequency components in the sampled data.

Recent research in SHM has shown that aliasing can severely compromise modal identification and damage detection capabilities, particularly when using low-cost wireless sensor networks.

The aliasing relationship is given by:
$$f_{\text{alias}} = |f_{\text{true}} - k \cdot f_s|$$
$$(3.9)$$

where $k$ is the integer that minimizes $f_{\text{alias}}$, and $f_{\text{true}}$ is the actual frequency component.

### 3.2.3 Anti-Aliasing Filter Design

To prevent aliasing, an **anti-aliasing filter** (low-pass filter) must be applied before sampling. The filter should attenuate all frequencies above $f_s/2$.

**Ideal Anti-Aliasing Filter:**
$$H_{\text{ideal}}(f) = \begin{cases}
1 & |f| \leq f_s/2 \\
0 & |f| > f_s/2
\end{cases}$$
$$(3.10)$$

**Practical Butterworth Anti-Aliasing Filter:**
$$|H(f)|^2 = \frac{1}{1 + \left(\frac{f}{f_c}\right)^{2n}}$$
$$(3.11)$$

where $f_c$ is the cutoff frequency and $n$ is the filter order.

### 3.2.4 Practical Considerations for SHM Applications

**Typical Frequency Ranges in Bridge Monitoring:**
- **Primary structural modes**: 0.5 - 10 Hz
- **Higher modes**: 10 - 50 Hz  
- **Traffic-induced vibrations**: 1 - 20 Hz
- **Wind-induced responses**: 0.1 - 5 Hz
- **Measurement noise**: Up to several hundred Hz

**Recommended Sampling Frequencies:**
- **General purpose SHM**: 200 - 500 Hz
- **Detailed modal analysis**: 500 - 1000 Hz
- **High-frequency damage detection**: 1000 - 5000 Hz

### 3.2.5 Implementation: Sampling and Anti-Aliasing

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import plotly.express as px

def demonstrate_aliasing_effects():
    """
    Demonstrate aliasing effects in bridge acceleration measurement.
    """
    
    # Create a complex bridge response signal
    t_continuous = np.linspace(0, 4, 10000)  # High resolution time
    fs_high = len(t_continuous) / 4  # High sampling rate
    
    # Bridge response components
    f1, f2, f3 = 2.3, 7.8, 15.6  # Hz - typical bridge frequencies
    f_noise = 45  # Hz - high frequency noise
    
    # Construct signal
    signal_clean = (0.5 * np.sin(2*np.pi*f1*t_continuous) + 
                   0.3 * np.sin(2*np.pi*f2*t_continuous) + 
                   0.2 * np.sin(2*np.pi*f3*t_continuous))
    
    # Add high-frequency noise that will cause aliasing
    signal_with_noise = signal_clean + 0.15 * np.sin(2*np.pi*f_noise*t_continuous)
    
    # Define different sampling rates
    fs_adequate = 100    # Adequate sampling rate
    fs_inadequate = 25   # Inadequate sampling rate (will cause aliasing)
    
    # Sample the signals
    # Adequate sampling
    n_adequate = int(4 * fs_adequate)
    t_adequate = np.linspace(0, 4, n_adequate)
    dt_adequate = 4 / (n_adequate - 1)
    indices_adequate = np.round(t_adequate * fs_high / 4 * (len(t_continuous) - 1)).astype(int)
    indices_adequate = np.clip(indices_adequate, 0, len(t_continuous)-1)
    
    signal_sampled_adequate = signal_with_noise[indices_adequate]
    
    # Inadequate sampling (aliasing occurs)
    n_inadequate = int(4 * fs_inadequate)
    t_inadequate = np.linspace(0, 4, n_inadequate)
    indices_inadequate = np.round(t_inadequate * fs_high / 4 * (len(t_continuous) - 1)).astype(int)
    indices_inadequate = np.clip(indices_inadequate, 0, len(t_continuous)-1)
    
    signal_sampled_inadequate = signal_with_noise[indices_inadequate]
    
    # Apply anti-aliasing filter before inadequate sampling
    # Design Butterworth low-pass filter
    fc = fs_inadequate / 2 * 0.8  # Cutoff at 80% of Nyquist frequency
    sos = signal.butter(6, fc, btype='low', fs=fs_high, output='sos')
    signal_filtered = signal.sosfilt(sos, signal_with_noise)
    signal_sampled_filtered = signal_filtered[indices_inadequate]
    
    # Create visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Continuous Signal (True Response)', 'Frequency Spectrum (Continuous)',
            'Adequate Sampling (100 Hz)', 'Inadequate Sampling (25 Hz) - Aliased',
            'Anti-Aliasing Filtered', 'Comparison of Sampling Effects'
        ),
        specs=[[{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}]]
    )
    
    # Continuous signal
    fig.add_trace(
        go.Scatter(x=t_continuous, y=signal_with_noise, mode='lines',
                  name='Continuous Signal', line=dict(color='#2E86AB', width=1)),
        row=1, col=1
    )
    
    # Frequency spectrum of continuous signal
    frequencies = np.fft.fftfreq(len(t_continuous), t_continuous[1] - t_continuous[0])
    fft_continuous = np.fft.fft(signal_with_noise)
    pos_freq_mask = frequencies >= 0
    
    fig.add_trace(
        go.Scatter(x=frequencies[pos_freq_mask], y=np.abs(fft_continuous[pos_freq_mask]),
                  mode='lines', name='Original Spectrum', 
                  line=dict(color='#2E86AB', width=2)),
        row=1, col=2
    )
    
    # Mark Nyquist frequencies
    fig.add_vline(x=fs_adequate/2, line_dash="dash", line_color="green", 
                 annotation_text=f"Nyquist (100Hz): {fs_adequate/2}Hz", row=1, col=2)
    fig.add_vline(x=fs_inadequate/2, line_dash="dash", line_color="red",
                 annotation_text=f"Nyquist (25Hz): {fs_inadequate/2}Hz", row=1, col=2)
    
    # Adequate sampling
    fig.add_trace(
        go.Scatter(x=t_adequate, y=signal_sampled_adequate, mode='lines+markers',
                  name='Adequate Sampling', line=dict(color='#A8E6CF', width=2),
                  marker=dict(size=4)),
        row=2, col=1
    )
    
    # Inadequate sampling (aliased)
    fig.add_trace(
        go.Scatter(x=t_inadequate, y=signal_sampled_inadequate, mode='lines+markers',
                  name='Inadequate Sampling', line=dict(color='#FF6B6B', width=2),
                  marker=dict(size=6, color='red')),
        row=2, col=2
    )
    
    # Anti-aliasing filtered
    fig.add_trace(
        go.Scatter(x=t_inadequate, y=signal_sampled_filtered, mode='lines+markers',
                  name='Anti-Aliasing Filtered', line=dict(color='#4ECDC4', width=2),
                  marker=dict(size=4)),
        row=3, col=1
    )
    
    # Comparison
    fig.add_trace(
        go.Scatter(x=t_continuous, y=signal_clean, mode='lines',
                  name='True Signal (Clean)', line=dict(color='#2E86AB', width=2)),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=t_inadequate, y=signal_sampled_inadequate, mode='markers',
                  name='Aliased Samples', marker=dict(size=6, color='red')),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=t_inadequate, y=signal_sampled_filtered, mode='markers',
                  name='Filtered Samples', marker=dict(size=6, color='#4ECDC4')),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text="Aliasing Effects in Bridge Monitoring: The Importance of Proper Sampling",
        title_font_size=16,
        showlegend=True
    )
    
    # Update axes
    for i in range(1, 4):
        for j in range(1, 3):
            if j == 1:
                fig.update_xaxes(title_text="Time (s)", row=i, col=j)
                fig.update_yaxes(title_text="Acceleration (m/s²)", row=i, col=j)
    
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude", row=1, col=2)
    
    fig.show()
    
    # Calculate aliased frequencies
    print("=== ALIASING ANALYSIS ===")
    print(f"Original frequency component: {f_noise} Hz")
    print(f"Inadequate sampling rate: {fs_inadequate} Hz")
    print(f"Nyquist frequency: {fs_inadequate/2} Hz")
    
    # Calculate where the 45 Hz component aliases to
    k = round(f_noise / fs_inadequate)
    f_aliased = abs(f_noise - k * fs_inadequate)
    print(f"45 Hz component aliases to: {f_aliased} Hz")
    print(f"This creates a false low-frequency component!")
    
    return fig

# Demonstrate anti-aliasing filter design
def design_antialiasing_filter():
    """
    Design and visualize anti-aliasing filter for bridge monitoring.
    """
    
    # Sampling parameters for bridge monitoring
    fs = 200  # Sampling frequency (Hz)
    f_nyquist = fs / 2
    f_cutoff = f_nyquist * 0.8  # Conservative cutoff
    
    # Design different filter orders
    orders = [2, 4, 6, 8]
    
    # Frequency range for analysis
    frequencies = np.logspace(-1, 2, 1000)  # 0.1 to 100 Hz
    
    fig = go.Figure()
    
    for order in orders:
        # Design Butterworth filter
        sos = signal.butter(order, f_cutoff, btype='low', fs=fs, output='sos')
        w, h = signal.sosfreqz(sos, worN=frequencies, fs=fs)
        
        # Plot magnitude response
        fig.add_trace(
            go.Scatter(x=frequencies, y=20*np.log10(np.abs(h)),
                      mode='lines', name=f'Order {order}',
                      line=dict(width=2))
        )
    
    # Add reference lines
    fig.add_vline(x=f_cutoff, line_dash="dash", line_color="green",
                 annotation_text=f"Cutoff: {f_cutoff:.1f} Hz")
    fig.add_vline(x=f_nyquist, line_dash="dash", line_color="red",
                 annotation_text=f"Nyquist: {f_nyquist:.1f} Hz")
    fig.add_hline(y=-3, line_dash="dot", line_color="gray",
                 annotation_text="-3 dB")
    fig.add_hline(y=-40, line_dash="dot", line_color="gray",
                 annotation_text="-40 dB")
    
    fig.update_layout(
        title="Anti-Aliasing Filter Design for Bridge Monitoring (fs = 200 Hz)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        xaxis_type="log",
        height=500,
        showlegend=True
    )
    
    fig.show()
    
    print("=== ANTI-ALIASING FILTER DESIGN ===")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Nyquist frequency: {f_nyquist} Hz")
    print(f"Recommended cutoff: {f_cutoff} Hz")
    print(f"Transition band: {f_cutoff} - {f_nyquist} Hz")

# Run demonstrations
aliasing_demo = demonstrate_aliasing_effects()
design_antialiasing_filter()
```

### 3.2.6 Practical Guidelines for SHM Sensor Networks

**Sensor Selection Criteria:**
1. **Maximum structural frequency of interest** (typically 20-50 Hz for bridges)
2. **Required anti-aliasing filter performance** (>40 dB attenuation above Nyquist)
3. **Data transmission and storage constraints**
4. **Power consumption limitations** (especially for wireless sensors)

Recent advances in compressed sensing have shown promise for sub-Nyquist sampling in SHM applications, though these require sophisticated reconstruction algorithms.

---

## 3.3 Data Preprocessing and Cleaning

### 3.3.1 Motivation for Robust Preprocessing

Real-world bridge monitoring data often contains various artifacts that can severely impact analysis results: sensor malfunctions, electromagnetic interference, data transmission errors, and environmental effects. Robust preprocessing ensures that subsequent analysis operates on clean, reliable data.

Common data quality issues in SHM include:
- **Outliers**: Sudden spikes due to sensor malfunction or electromagnetic interference
- **Missing data**: Transmission errors or sensor downtime
- **Trends**: Long-term drifts due to temperature or aging effects
- **Offsets**: DC bias changes in sensor electronics
- **Noise**: Random measurement uncertainty

### 3.3.2 Outlier Detection and Handling

**Statistical Outlier Detection:**

Z-score method identifies outliers as points with standardized values exceeding a threshold:
$$z_i = \frac{x[i] - \mu_x}{\sigma_x}$$
$$(3.12)$$

Outliers are defined as $|z_i| > z_{\text{threshold}}$ (typically $z_{\text{threshold}} = 3$).

**Robust Outlier Detection - Modified Z-Score:**
$$M_i = \frac{0.6745(x[i] - \text{median}(x))}{\text{MAD}(x)}$$
$$(3.13)$$

where MAD is the Median Absolute Deviation:
$$\text{MAD}(x) = \text{median}(|x[i] - \text{median}(x)|)$$
$$(3.14)$$

**Interquartile Range (IQR) Method:**
$$\text{Lower bound} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Upper bound} = Q_3 + 1.5 \times \text{IQR}$$
$$(3.15)$$

where $\text{IQR} = Q_3 - Q_1$ (75th percentile - 25th percentile).

### 3.3.3 Missing Data Interpolation

**Linear Interpolation:**
For missing samples between indices $n_1$ and $n_2$:
$$x[n] = x[n_1] + \frac{x[n_2] - x[n_1]}{n_2 - n_1}(n - n_1)$$
$$(3.16)$$

**Spline Interpolation:**
Uses piecewise polynomial functions for smoother interpolation.

**Autoregressive (AR) Model Interpolation:**
For structural signals with known dynamics:
$$x[n] = \sum_{k=1}^{p} a_k x[n-k] + e[n]$$
$$(3.17)$$

### 3.3.4 Detrending Techniques

**Linear Detrending:**
Remove linear trend by fitting and subtracting a straight line:
$$\hat{x}[n] = x[n] - (an + b)$$
$$(3.18)$$

where $a$ and $b$ are determined by least squares fitting.

**Polynomial Detrending:**
Remove higher-order trends using polynomial fitting:
$$\text{trend}[n] = \sum_{k=0}^{p} c_k n^k$$
$$(3.19)$$

**High-Pass Filtering for Detrending:**
Moving average detrending subtracts a rolling window average from the original signal:
$$x_{\text{detrended}}[n] = x[n] - \frac{1}{W} \sum_{k=-W/2}^{W/2} x[n+k]$$
$$(3.20)$$

### 3.3.5 Implementation: Robust Data Preprocessing Pipeline

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal, interpolate, stats
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class SHMDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for structural health monitoring.
    """
    
    def __init__(self, fs):
        """
        Initialize preprocessor with sampling frequency.
        
        Parameters:
        -----------
        fs : float
            Sampling frequency in Hz
        """
        self.fs = fs
        self.dt = 1.0 / fs
        self.preprocessing_log = []
    
    def detect_outliers(self, data, method='modified_zscore', threshold=3.5):
        """
        Detect outliers using multiple methods.
        
        Parameters:
        -----------
        data : array-like
            Input signal
        method : str
            Outlier detection method ('zscore', 'modified_zscore', 'iqr')
        threshold : float
            Threshold for outlier detection
            
        Returns:
        --------
        outlier_mask : array
            Boolean array indicating outliers
        """
        
        data = np.asarray(data)
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
            
        elif method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            
        else:
            raise ValueError("Method must be 'zscore', 'modified_zscore', or 'iqr'")
        
        n_outliers = np.sum(outlier_mask)
        outlier_percentage = 100 * n_outliers / len(data)
        
        self.preprocessing_log.append({
            'step': 'outlier_detection',
            'method': method,
            'threshold': threshold,
            'outliers_found': n_outliers,
            'outlier_percentage': outlier_percentage
        })
        
        return outlier_mask
    
    def handle_outliers(self, data, outlier_mask, method='interpolate'):
        """
        Handle detected outliers.
        
        Parameters:
        -----------
        data : array-like
            Input signal
        outlier_mask : array
            Boolean array indicating outliers
        method : str
            Handling method ('remove', 'interpolate', 'clip')
            
        Returns:
        --------
        data_clean : array
            Signal with outliers handled
        """
        
        data = np.asarray(data).copy()
        
        if method == 'remove':
            data_clean = data[~outlier_mask]
            
        elif method == 'interpolate':
            data_clean = data.copy()
            if np.any(outlier_mask):
                # Create time index
                t = np.arange(len(data))
                t_clean = t[~outlier_mask]
                data_clean_vals = data[~outlier_mask]
                
                # Interpolate missing values
                if len(t_clean) > 1:  # Need at least 2 points for interpolation
                    interp_func = interpolate.interp1d(
                        t_clean, data_clean_vals, 
                        kind='linear', 
                        bounds_error=False, 
                        fill_value='extrapolate'
                    )
                    data_clean[outlier_mask] = interp_func(t[outlier_mask])
        
        elif method == 'clip':
            # Clip to reasonable bounds (e.g., 5th and 95th percentiles)
            lower_bound = np.percentile(data[~outlier_mask], 5)
            upper_bound = np.percentile(data[~outlier_mask], 95)
            data_clean = np.clip(data, lower_bound, upper_bound)
            
        else:
            raise ValueError("Method must be 'remove', 'interpolate', or 'clip'")
        
        self.preprocessing_log.append({
            'step': 'outlier_handling',
            'method': method,
            'outliers_handled': np.sum(outlier_mask)
        })
        
        return data_clean
    
    def handle_missing_data(self, data, missing_mask=None, method='linear'):
        """
        Handle missing data points.
        
        Parameters:
        -----------
        data : array-like
            Input signal (NaN values indicate missing data)
        missing_mask : array, optional
            Boolean array indicating missing data
        method : str
            Interpolation method ('linear', 'cubic', 'autoregressive')
            
        Returns:
        --------
        data_filled : array
            Signal with missing data filled
        """
        
        data = np.asarray(data).copy()
        
        if missing_mask is None:
            missing_mask = np.isnan(data)
        
        if not np.any(missing_mask):
            return data  # No missing data
        
        # Create time index
        t = np.arange(len(data))
        t_valid = t[~missing_mask]
        data_valid = data[~missing_mask]
        
        if len(t_valid) < 2:
            raise ValueError("Insufficient valid data for interpolation")
        
        if method == 'linear':
            interp_func = interpolate.interp1d(
                t_valid, data_valid, 
                kind='linear', 
                bounds_error=False, 
                fill_value='extrapolate'
            )
            data[missing_mask] = interp_func(t[missing_mask])
            
        elif method == 'cubic':
            if len(t_valid) >= 4:  # Cubic requires at least 4 points
                interp_func = interpolate.interp1d(
                    t_valid, data_valid, 
                    kind='cubic', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                data[missing_mask] = interp_func(t[missing_mask])
            else:
                # Fall back to linear interpolation
                interp_func = interpolate.interp1d(
                    t_valid, data_valid, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value='extrapolate'
                )
                data[missing_mask] = interp_func(t[missing_mask])
        
        elif method == 'autoregressive':
            # Simple AR(2) model for demonstration
            if len(data_valid) > 10:  # Need sufficient data for AR fitting
                # Fit AR model to valid data
                from scipy.signal import lfilter
                
                # Use backward and forward prediction
                # This is a simplified implementation
                data[missing_mask] = np.interp(t[missing_mask], t_valid, data_valid)
        
        self.preprocessing_log.append({
            'step': 'missing_data_handling',
            'method': method,
            'missing_points': np.sum(missing_mask),
            'missing_percentage': 100 * np.sum(missing_mask) / len(data)
        })
        
        return data
    
    def detrend_signal(self, data, method='linear', poly_order=1, filter_cutoff=None):
        """
        Remove trends from signal.
        
        Parameters:
        -----------
        data : array-like
            Input signal
        method : str
            Detrending method ('linear', 'polynomial', 'highpass', 'moving_average')
        poly_order : int
            Polynomial order for polynomial detrending
        filter_cutoff : float
            Cutoff frequency for high-pass filtering (Hz)
            
        Returns:
        --------
        data_detrended : array
            Detrended signal
        trend : array
            Removed trend component
        """
        
        data = np.asarray(data)
        n = len(data)
        t = np.arange(n)
        
        if method == 'linear':
            # Fit linear trend
            coeffs = np.polyfit(t, data, 1)
            trend = np.polyval(coeffs, t)
            data_detrended = data - trend
            
        elif method == 'polynomial':
            # Fit polynomial trend
            coeffs = np.polyfit(t, data, poly_order)
            trend = np.polyval(coeffs, t)
            data_detrended = data - trend
            
        elif method == 'highpass':
            # High-pass filter to remove low-frequency trends
            if filter_cutoff is None:
                filter_cutoff = 0.1  # Default 0.1 Hz cutoff
            
            sos = signal.butter(4, filter_cutoff, btype='high', fs=self.fs, output='sos')
            data_detrended = signal.sosfilt(sos, data)
            trend = data - data_detrended
            
        elif method == 'moving_average':
            # Moving average detrending
            window_size = min(int(self.fs * 10), n // 4)  # 10 second window or 1/4 signal length
            if window_size % 2 == 0:
                window_size += 1  # Make odd for symmetric window
            
            # Compute moving average
            trend = signal.savgol_filter(data, window_size, 1)  # Order 1 (linear) smoothing
            data_detrended = data - trend
            
        else:
            raise ValueError("Method must be 'linear', 'polynomial', 'highpass', or 'moving_average'")
        
        self.preprocessing_log.append({
            'step': 'detrending',
            'method': method,
            'trend_rms': np.sqrt(np.mean(trend**2)),
            'detrended_rms': np.sqrt(np.mean(data_detrended**2))
        })
        
        return data_detrended, trend
    
    def filter_signal(self, data, filter_type='bandpass', low_freq=0.5, high_freq=50, order=4):
        """
        Apply digital filter to signal.
        
        Parameters:
        -----------
        data : array-like
            Input signal
        filter_type : str
            Filter type ('lowpass', 'highpass', 'bandpass', 'bandstop')
        low_freq : float
            Low cutoff frequency (Hz)
        high_freq : float
            High cutoff frequency (Hz)
        order : int
            Filter order
            
        Returns:
        --------
        data_filtered : array
            Filtered signal
        """
        
        data = np.asarray(data)
        
        if filter_type == 'lowpass':
            sos = signal.butter(order, high_freq, btype='low', fs=self.fs, output='sos')
        elif filter_type == 'highpass':
            sos = signal.butter(order, low_freq, btype='high', fs=self.fs, output='sos')
        elif filter_type == 'bandpass':
            sos = signal.butter(order, [low_freq, high_freq], btype='band', fs=self.fs, output='sos')
        elif filter_type == 'bandstop':
            sos = signal.butter(order, [low_freq, high_freq], btype='bandstop', fs=self.fs, output='sos')
        else:
            raise ValueError("Filter type must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'")
        
        data_filtered = signal.sosfilt(sos, data)
        
        self.preprocessing_log.append({
            'step': 'filtering',
            'filter_type': filter_type,
            'low_freq': low_freq,
            'high_freq': high_freq,
            'order': order
        })
        
        return data_filtered
    
    def preprocess_pipeline(self, data, outlier_detection=True, missing_data_handling=True, 
                          detrending=True, filtering=True, **kwargs):
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        data : array-like
            Raw input signal
        outlier_detection : bool
            Enable outlier detection and handling
        missing_data_handling : bool
            Enable missing data interpolation
        detrending : bool
            Enable detrending
        filtering : bool
            Enable filtering
        **kwargs : dict
            Parameters for individual processing steps
            
        Returns:
        --------
        processed_data : array
            Fully processed signal
        processing_info : dict
            Information about processing steps applied
        """
        
        self.preprocessing_log = []  # Reset log
        processed_data = np.asarray(data).copy()
        
        print("Starting SHM Data Preprocessing Pipeline...")
        print(f"Input signal: {len(processed_data)} samples, {len(processed_data)/self.fs:.1f} seconds")
        
        # Step 1: Outlier detection and handling
        if outlier_detection:
            outlier_method = kwargs.get('outlier_method', 'modified_zscore')
            outlier_threshold = kwargs.get('outlier_threshold', 3.5)
            outlier_handling = kwargs.get('outlier_handling', 'interpolate')
            
            outlier_mask = self.detect_outliers(processed_data, outlier_method, outlier_threshold)
            processed_data = self.handle_outliers(processed_data, outlier_mask, outlier_handling)
            
            print(f"✓ Outlier detection: {np.sum(outlier_mask)} outliers found and handled")
        
        # Step 2: Missing data handling
        if missing_data_handling:
            missing_method = kwargs.get('missing_method', 'linear')
            missing_mask = np.isnan(processed_data)
            
            if np.any(missing_mask):
                processed_data = self.handle_missing_data(processed_data, missing_mask, missing_method)
                print(f"✓ Missing data: {np.sum(missing_mask)} points interpolated")
            else:
                print("✓ Missing data: No missing points found")
        
        # Step 3: Detrending
        if detrending:
            detrend_method = kwargs.get('detrend_method', 'linear')
            poly_order = kwargs.get('poly_order', 1)
            filter_cutoff = kwargs.get('filter_cutoff', None)
            
            processed_data, trend = self.detrend_signal(
                processed_data, detrend_method, poly_order, filter_cutoff
            )
            print(f"✓ Detrending: {detrend_method} method applied")
        
        # Step 4: Filtering
        if filtering:
            filter_type = kwargs.get('filter_type', 'bandpass')
            low_freq = kwargs.get('low_freq', 0.5)
            high_freq = kwargs.get('high_freq', 50)
            filter_order = kwargs.get('filter_order', 4)
            
            processed_data = self.filter_signal(
                processed_data, filter_type, low_freq, high_freq, filter_order
            )
            print(f"✓ Filtering: {filter_type} filter applied ({low_freq}-{high_freq} Hz)")
        
        processing_info = {
            'original_length': len(data),
            'processed_length': len(processed_data),
            'sampling_frequency': self.fs,
            'steps_applied': [step['step'] for step in self.preprocessing_log],
            'detailed_log': self.preprocessing_log
        }
        
        print("✓ Preprocessing pipeline completed successfully!")
        
        return processed_data, processing_info

# Generate corrupted bridge data for demonstration
def create_corrupted_bridge_data():
    """Create realistic corrupted bridge acceleration data."""
    
    # Generate clean signal
    fs = 200  # Hz
    duration = 30  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Clean bridge response
    f1, f2 = 2.1, 6.8  # Bridge frequencies
    clean_signal = (0.5 * np.sin(2*np.pi*f1*t) * np.exp(-0.02*t) + 
                   0.3 * np.sin(2*np.pi*f2*t) * np.exp(-0.05*t))
    
    # Add realistic corruptions
    corrupted_signal = clean_signal.copy()
    
    # 1. Add linear trend (temperature effect)
    trend = 0.001 * t
    corrupted_signal += trend
    
    # 2. Add outliers (electromagnetic interference)
    n_outliers = 50
    outlier_indices = np.random.choice(len(t), n_outliers, replace=False)
    outlier_magnitude = np.random.uniform(5, 15, n_outliers)
    outlier_signs = np.random.choice([-1, 1], n_outliers)
    corrupted_signal[outlier_indices] += outlier_magnitude * outlier_signs
    
    # 3. Add missing data (communication dropouts)
    missing_indices = np.random.choice(len(t), int(0.02 * len(t)), replace=False)
    corrupted_signal[missing_indices] = np.nan
    
    # 4. Add measurement noise
    noise = 0.05 * np.random.normal(0, 1, len(t))
    corrupted_signal += noise
    
    return t, clean_signal, corrupted_signal

# Demonstrate preprocessing pipeline
def demonstrate_preprocessing():
    """Demonstrate the complete preprocessing pipeline."""
    
    # Create corrupted data
    t, clean_signal, corrupted_signal = create_corrupted_bridge_data()
    fs = 200  # Hz
    
    # Initialize preprocessor
    preprocessor = SHMDataPreprocessor(fs)
    
    # Apply preprocessing pipeline
    processed_signal, processing_info = preprocessor.preprocess_pipeline(
        corrupted_signal,
        outlier_detection=True,
        missing_data_handling=True,
        detrending=True,
        filtering=True,
        outlier_method='modified_zscore',
        outlier_threshold=3.0,
        outlier_handling='interpolate',
        missing_method='cubic',
        detrend_method='linear',
        filter_type='bandpass',
        low_freq=0.5,
        high_freq=50,
        filter_order=4
    )
    
    # Create visualization
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Original Clean Signal',
            'Corrupted Signal (Outliers + Missing + Trend + Noise)',
            'After Preprocessing Pipeline',
            'Comparison: Clean vs Processed'
        ),
        vertical_spacing=0.08
    )
    
    # Original clean signal
    fig.add_trace(
        go.Scatter(x=t, y=clean_signal, mode='lines', name='Clean Signal',
                  line=dict(color='#2E86AB', width=2)),
        row=1, col=1
    )
    
    # Corrupted signal
    fig.add_trace(
        go.Scatter(x=t, y=corrupted_signal, mode='lines', name='Corrupted Signal',
                  line=dict(color='#FF6B6B', width=1)),
        row=2, col=1
    )
    
    # Processed signal
    fig.add_trace(
        go.Scatter(x=t, y=processed_signal, mode='lines', name='Processed Signal',
                  line=dict(color='#4ECDC4', width=2)),
        row=3, col=1
    )
    
    # Comparison
    fig.add_trace(
        go.Scatter(x=t, y=clean_signal, mode='lines', name='Original Clean',
                  line=dict(color='#2E86AB', width=2)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=processed_signal, mode='lines', name='Processed',
                  line=dict(color='#4ECDC4', width=2, dash='dash')),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        title_text="SHM Data Preprocessing Pipeline Demonstration",
        title_font_size=16,
        showlegend=True
    )
    
    # Update axes
    for i in range(1, 5):
        fig.update_xaxes(title_text="Time (s)", row=i, col=1)
        fig.update_yaxes(title_text="Acceleration (m/s²)", row=i, col=1)
    
    fig.show()
    
    # Compute error metrics
    # Align signals (processed might be shorter due to filtering)
    min_length = min(len(clean_signal), len(processed_signal))
    clean_aligned = clean_signal[:min_length]
    processed_aligned = processed_signal[:min_length]
    
    rmse = np.sqrt(np.mean((clean_aligned - processed_aligned)**2))
    correlation = np.corrcoef(clean_aligned, processed_aligned)[0, 1]
    
    print(f"\n=== PREPROCESSING PERFORMANCE ===")
    print(f"RMSE (Clean vs Processed): {rmse:.6f}")
    print(f"Correlation (Clean vs Processed): {correlation:.4f}")
    print(f"Signal recovery quality: {'Excellent' if correlation > 0.95 else 'Good' if correlation > 0.85 else 'Fair'}")
    
    return fig, processing_info

# Run demonstration
preprocessing_demo, info = demonstrate_preprocessing()
```

This comprehensive preprocessing pipeline addresses the most common data quality issues encountered in structural health monitoring applications, providing a robust foundation for subsequent analysis.

---

## 3.4 Digital Filtering Fundamentals

### 3.4.1 Filter Classification and Design Principles

Digital filters are essential tools in SHM for separating signal components of interest from noise and unwanted frequency content. For bridge monitoring, filters serve multiple purposes: anti-aliasing, noise reduction, frequency band isolation, and trend removal.

**Filter Categories by Frequency Response:**
- **Low-pass**: Passes frequencies below cutoff $f_c$
- **High-pass**: Passes frequencies above cutoff $f_c$  
- **Band-pass**: Passes frequencies between $f_1$ and $f_2$
- **Band-stop (Notch)**: Rejects frequencies between $f_1$ and $f_2$

**Filter Categories by Impulse Response:**
- **Finite Impulse Response (FIR)**: $h[n] = 0$ for $n < 0$ and $n \geq N$
- **Infinite Impulse Response (IIR)**: $h[n] \neq 0$ for all $n \geq 0$

### 3.4.2 Mathematical Representation

**Difference Equation (Time Domain):**
$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$
$$(3.21)$$

where $\{b_k\}$ are feedforward coefficients, $\{a_k\}$ are feedback coefficients, and $a_0 = 1$.

**Transfer Function (Z-Domain):**
$$H(z) = \frac{Y(z)}{X(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$$
$$(3.22)$$

**Frequency Response:**
$$H(e^{j\omega}) = H(z)|_{z=e^{j\omega}} = |H(e^{j\omega})| e^{j\phi(\omega)}$$
$$(3.23)$$

where $|H(e^{j\omega})|$ is the magnitude response and $\phi(\omega)$ is the phase response.

### 3.4.3 FIR vs IIR Filter Comparison

| Characteristic | FIR Filters | IIR Filters |
|---|---|---|
| **Stability** | Always stable | Can be unstable |
| **Phase Response** | Can be exactly linear | Generally nonlinear |
| **Computational Cost** | Higher (more coefficients) | Lower (fewer coefficients) |
| **Design Complexity** | Straightforward | More complex |
| **Memory Requirements** | Higher | Lower |
| **Applications in SHM** | Precision filtering, matched filtering | Real-time processing, anti-aliasing |

### 3.4.4 Butterworth Filter Design

The Butterworth filter provides maximally flat frequency response in the passband. The magnitude-squared response is:

$$|H(\omega)|^2 = \frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2n}}$$
$$(3.24)$$

where $\omega_c$ is the cutoff frequency and $n$ is the filter order.

**Key Properties:**
- **3 dB cutoff**: $|H(\omega_c)| = 1/\sqrt{2} = -3$ dB
- **Roll-off rate**: $20n$ dB/decade beyond cutoff
- **Group delay**: Non-constant (phase distortion)

### 3.4.5 Implementation: Comprehensive Filtering for Bridge SHM

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import plotly.express as px

class SHMFilterDesigner:
    """
    Comprehensive digital filter design and analysis for SHM applications.
    """
    
    def __init__(self, fs):
        """
        Initialize filter designer.
        
        Parameters:
        -----------
        fs : float
            Sampling frequency (Hz)
        """
        self.fs = fs
        self.nyquist = fs / 2
    
    def design_butterworth_filter(self, cutoff, filter_type='low', order=4):
        """
        Design Butterworth filter for SHM applications.
        
        Parameters:
        -----------
        cutoff : float or list
            Cutoff frequency/frequencies (Hz)
        filter_type : str
            Filter type ('low', 'high', 'band', 'bandstop')
        order : int
            Filter order
            
        Returns:
        --------
        sos : array
            Second-order sections representation
        """
        
        # Normalize cutoff frequencies
        if isinstance(cutoff, (list, tuple)):
            wn = [f / self.nyquist for f in cutoff]
        else:
            wn = cutoff / self.nyquist
        
        # Design filter
        sos = signal.butter(order, wn, btype=filter_type, output='sos')
        
        return sos
    
    def design_chebyshev_filter(self, cutoff, filter_type='low', order=4, ripple=0.5):
        """
        Design Chebyshev Type I filter.
        
        Parameters:
        -----------
        cutoff : float or list
            Cutoff frequency/frequencies (Hz)
        filter_type : str
            Filter type ('low', 'high', 'band', 'bandstop')
        order : int
            Filter order
        ripple : float
            Maximum ripple in passband (dB)
            
        Returns:
        --------
        sos : array
            Second-order sections representation
        """
        
        # Normalize cutoff frequencies
        if isinstance(cutoff, (list, tuple)):
            wn = [f / self.nyquist for f in cutoff]
        else:
            wn = cutoff / self.nyquist
        
        # Design filter
        sos = signal.cheby1(order, ripple, wn, btype=filter_type, output='sos')
        
        return sos
    
    def design_elliptic_filter(self, cutoff, filter_type='low', order=4, 
                              passband_ripple=0.5, stopband_attenuation=40):
        """
        Design elliptic (Cauer) filter.
        
        Parameters:
        -----------
        cutoff : float or list
            Cutoff frequency/frequencies (Hz)
        filter_type : str
            Filter type ('low', 'high', 'band', 'bandstop')
        order : int
            Filter order
        passband_ripple : float
            Maximum ripple in passband (dB)
        stopband_attenuation : float
            Minimum attenuation in stopband (dB)
            
        Returns:
        --------
        sos : array
            Second-order sections representation
        """
        
        # Normalize cutoff frequencies
        if isinstance(cutoff, (list, tuple)):
            wn = [f / self.nyquist for f in cutoff]
        else:
            wn = cutoff / self.nyquist
        
        # Design filter
        sos = signal.ellip(order, passband_ripple, stopband_attenuation, 
                          wn, btype=filter_type, output='sos')
        
        return sos
    
    def analyze_filter_response(self, sos, freq_range=None):
        """
        Analyze filter frequency and phase response.
        
        Parameters:
        -----------
        sos : array
            Second-order sections representation
        freq_range : tuple, optional
            Frequency range for analysis (Hz)
            
        Returns:
        --------
        frequencies : array
            Frequency points (Hz)
        magnitude : array
            Magnitude response
        phase : array
            Phase response (radians)
        group_delay : array
            Group delay (samples)
        """
        
        if freq_range is None:
            freq_range = (0.01, self.nyquist)
        
        # Create frequency vector
        frequencies = np.logspace(np.log10(freq_range[0]), 
                                 np.log10(freq_range[1]), 1000)
        
        # Compute frequency response
        w, h = signal.sosfreqz(sos, worN=frequencies, fs=self.fs)
        magnitude = np.abs(h)
        phase = np.angle(h)
        
        # Compute group delay
        w_gd, group_delay = signal.group_delay((sos), w=frequencies, fs=self.fs)
        
        return frequencies, magnitude, phase, group_delay
    
    def apply_filter(self, data, sos, method='sosfilt'):
        """
        Apply filter to data.
        
        Parameters:
        -----------
        data : array
            Input signal
        sos : array
            Second-order sections representation
        method : str
            Filtering method ('sosfilt', 'sosfilt_zi', 'filtfilt')
            
        Returns:
        --------
        filtered_data : array
            Filtered signal
        """
        
        if method == 'sosfilt':
            # Standard filtering (causal)
            filtered_data = signal.sosfilt(sos, data)
            
        elif method == 'sosfilt_zi':
            # Filtering with initial conditions
            zi = signal.sosfilt_zi(sos)
            filtered_data, _ = signal.sosfilt(sos, data, zi=zi*data[0])
            
        elif method == 'filtfilt':
            # Zero-phase filtering (non-causal)
            filtered_data = signal.sosfiltfilt(sos, data)
            
        else:
            raise ValueError("Method must be 'sosfilt', 'sosfilt_zi', or 'filtfilt'")
        
        return filtered_data
    
    def design_bridge_monitoring_filters(self):
        """
        Design standard filter bank for bridge monitoring applications.
        
        Returns:
        --------
        filter_bank : dict
            Dictionary of pre-designed filters for common SHM applications
        """
        
        filter_bank = {}
        
        # 1. Anti-aliasing filter (for 200 Hz sampling)
        filter_bank['anti_aliasing'] = {
            'sos': self.design_butterworth_filter(80, 'low', order=6),
            'description': 'Anti-aliasing filter for 200 Hz sampling',
            'application': 'Pre-sampling conditioning'
        }
        
        # 2. Structural dynamics filter (0.1 - 50 Hz)
        filter_bank['structural'] = {
            'sos': self.design_butterworth_filter([0.1, 50], 'band', order=4),
            'description': 'Structural dynamics frequency range',
            'application': 'Modal analysis, vibration monitoring'
        }
        
        # 3. First mode isolation (1.5 - 3.5 Hz typical for bridges)
        filter_bank['first_mode'] = {
            'sos': self.design_butterworth_filter([1.5, 3.5], 'band', order=6),
            'description': 'First mode frequency band',
            'application': 'Primary mode tracking, fundamental frequency monitoring'
        }
        
        # 4. Traffic frequency filter (1 - 20 Hz)
        filter_bank['traffic'] = {
            'sos': self.design_butterworth_filter([1, 20], 'band', order=4),
            'description': 'Traffic-induced vibration frequencies',
            'application': 'Traffic load monitoring, vehicle detection'
        }
        
        # 5. High-frequency noise removal (>100 Hz)
        filter_bank['noise_removal'] = {
            'sos': self.design_butterworth_filter(100, 'low', order=8),
            'description': 'High-frequency noise suppression',
            'application': 'Signal cleaning, measurement noise reduction'
        }
        
        # 6. Low-frequency trend removal (>0.05 Hz)
        filter_bank['detrend'] = {
            'sos': self.design_butterworth_filter(0.05, 'high', order=2),
            'description': 'Low-frequency trend and drift removal',
            'application': 'Baseline correction, temperature effect removal'
        }
        
        # 7. Power line interference notch (50/60 Hz)
        filter_bank['notch_50hz'] = {
            'sos': self.design_elliptic_filter([49, 51], 'bandstop', order=4, 
                                            passband_ripple=0.1, stopband_attenuation=40),
            'description': '50 Hz power line interference removal',
            'application': 'Electromagnetic interference suppression'
        }
        
        filter_bank['notch_60hz'] = {
            'sos': self.design_elliptic_filter([59, 61], 'bandstop', order=4, 
                                            passband_ripple=0.1, stopband_attenuation=40),
            'description': '60 Hz power line interference removal',
            'application': 'Electromagnetic interference suppression'
        }
        
        return filter_bank

def demonstrate_shm_filtering():
    """
    Comprehensive demonstration of filtering for bridge SHM.
    """
    
    # Initialize filter designer
    fs = 200  # Hz
    designer = SHMFilterDesigner(fs)
    
    # Design filter bank
    filter_bank = designer.design_bridge_monitoring_filters()
    
    # Generate realistic bridge acceleration signal with multiple components
    duration = 30  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Signal components
    # 1. Structural modes
    mode1 = 0.02 * np.sin(2*np.pi*2.3*t) * np.exp(-0.01*t)  # 1st mode
    mode2 = 0.01 * np.sin(2*np.pi*6.8*t) * np.exp(-0.02*t)  # 2nd mode
    mode3 = 0.005 * np.sin(2*np.pi*12.4*t) * np.exp(-0.03*t)  # 3rd mode
    
    # 2. Traffic loading (random vehicle passages)
    traffic = np.zeros_like(t)
    vehicle_times = [5, 12, 18, 25]  # Vehicle passage times
    for vt in vehicle_times:
        vehicle_response = 0.03 * np.exp(-2*(t-vt)) * np.sin(2*np.pi*8*(t-vt))
        vehicle_response[t < vt] = 0
        traffic += vehicle_response
    
    # 3. Environmental effects (temperature, wind)
    environmental = 0.003 * np.sin(2*np.pi*0.02*t)  # Very low frequency
    
    # 4. Power line interference
    power_line = 0.002 * np.sin(2*np.pi*50*t)
    
    # 5. High-frequency noise
    noise_hf = 0.001 * np.random.normal(0, 1, len(t))
    noise_hf_filtered = signal.sosfilt(
        signal.butter(4, [150, 200], btype='band', fs=fs, output='sos'), 
        np.random.normal(0, 1, len(t))
    ) * 0.005
    
    # 6. Measurement noise
    measurement_noise = 0.002 * np.random.normal(0, 1, len(t))
    
    # Combine all components
    raw_signal = (mode1 + mode2 + mode3 + traffic + environmental + 
                  power_line + noise_hf_filtered + measurement_noise)
    
    # Apply different filters
    filtered_signals = {}
    for filter_name, filter_info in filter_bank.items():
        if filter_name in ['structural', 'first_mode', 'traffic', 'noise_removal', 'notch_50hz']:
            filtered_signals[filter_name] = designer.apply_filter(
                raw_signal, filter_info['sos'], method='filtfilt'
            )
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Raw Bridge Acceleration Signal', 'Structural Dynamics Filter (0.1-50 Hz)',
            'First Mode Filter (1.5-3.5 Hz)', 'Traffic Filter (1-20 Hz)', 
            'Noise Removal Filter (<100 Hz)', 'Power Line Notch Filter (50 Hz)'
        ),
        vertical_spacing=0.1
    )
    
    # Raw signal
    fig.add_trace(
        go.Scatter(x=t, y=raw_signal*1000, mode='lines', name='Raw Signal',
                  line=dict(color='#FF6B6B', width=1)),
        row=1, col=1
    )
    
    # Filtered signals
    colors = ['#2E86AB', '#A8E6CF', '#FFD93D', '#6BCF7F', '#4ECDC4']
    filter_names = ['structural', 'first_mode', 'traffic', 'noise_removal', 'notch_50hz']
    
    positions = [(1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
    
    for i, (filter_name, (row, col)) in enumerate(zip(filter_names, positions)):
        fig.add_trace(
            go.Scatter(x=t, y=filtered_signals[filter_name]*1000, mode='lines',
                      name=f'{filter_name.title()} Filtered',
                      line=dict(color=colors[i], width=2)),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text="Bridge SHM Filtering Applications",
        title_font_size=16,
        showlegend=True
    )
    
    # Update axes
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Time (s)", row=i, col=j)
            fig.update_yaxes(title_text="Acceleration (mm/s²)", row=i, col=j)
    
    fig.show()
    
    # Frequency response analysis
    fig_freq = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Filter Magnitude Responses', 'Filter Phase Responses',
            'Group Delay Responses', 'Filter Comparison'
        )
    )
    
    # Analyze key filters
    key_filters = ['structural', 'first_mode', 'traffic', 'noise_removal']
    colors_freq = ['#2E86AB', '#A8E6CF', '#FFD93D', '#6BCF7F']
    
    for filter_name, color in zip(key_filters, colors_freq):
        sos = filter_bank[filter_name]['sos']
        frequencies, magnitude, phase, group_delay = designer.analyze_filter_response(sos)
        
        # Magnitude response
        fig_freq.add_trace(
            go.Scatter(x=frequencies, y=20*np.log10(magnitude), mode='lines',
                      name=filter_name.title(), line=dict(color=color, width=2)),
            row=1, col=1
        )
        
        # Phase response
        fig_freq.add_trace(
            go.Scatter(x=frequencies, y=np.unwrap(phase), mode='lines',
                      name=filter_name.title(), line=dict(color=color, width=2),
                      showlegend=False),
            row=1, col=2
        )
        
        # Group delay
        fig_freq.add_trace(
            go.Scatter(x=frequencies, y=group_delay, mode='lines',
                      name=filter_name.title(), line=dict(color=color, width=2),
                      showlegend=False),
            row=2, col=1
        )
    
    # Passband comparison
    passband_freqs = np.linspace(0.1, 20, 1000)
    for filter_name, color in zip(['structural', 'first_mode', 'traffic'], colors_freq[:3]):
        sos = filter_bank[filter_name]['sos']
        w, h = signal.sosfreqz(sos, worN=passband_freqs, fs=fs)
        
        fig_freq.add_trace(
            go.Scatter(x=passband_freqs, y=20*np.log10(np.abs(h)), mode='lines',
                      name=filter_name.title(), line=dict(color=color, width=2),
                      showlegend=False),
            row=2, col=2
        )
    
    # Update frequency plot layout
    fig_freq.update_layout(
        height=800,
        title_text="SHM Filter Bank: Frequency Response Analysis",
        title_font_size=16
    )
    
    # Update axes for frequency plots
    fig_freq.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=1)
    fig_freq.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    
    fig_freq.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
    fig_freq.update_yaxes(title_text="Phase (radians)", row=1, col=2)
    
    fig_freq.update_xaxes(title_text="Frequency (Hz)", type="log", row=2, col=1)
    fig_freq.update_yaxes(title_text="Group Delay (samples)", row=2, col=1)
    
    fig_freq.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig_freq.update_yaxes(title_text="Magnitude (dB)", row=2, col=2)
    
    fig_freq.show()
    
    # Performance analysis
    print("=== SHM FILTER BANK PERFORMANCE ===")
    for filter_name in key_filters:
        original_energy = np.sum(raw_signal**2)
        filtered_energy = np.sum(filtered_signals[filter_name]**2)
        energy_ratio = filtered_energy / original_energy
        
        print(f"{filter_name.upper()} FILTER:")
        print(f"  Energy retention: {energy_ratio:.3f} ({100*energy_ratio:.1f}%)")
        print(f"  RMS reduction: {np.sqrt(energy_ratio):.3f}")
        print(f"  Application: {filter_bank[filter_name]['description']}")
        print()

# Run filtering demonstration
demonstrate_shm_filtering()
```

This comprehensive filtering implementation provides the essential tools for processing bridge monitoring data, addressing the specific frequency content and noise characteristics typical in structural health monitoring applications.

---

## 3.5 Correlation and Convolution Analysis

### 3.5.1 Mathematical Foundations

Correlation and convolution are fundamental operations in structural health monitoring for analyzing relationships between signals, identifying system responses, and extracting structural characteristics from ambient vibration data.

**Cross-Correlation:**
The cross-correlation between two signals $x[n]$ and $y[n]$ quantifies their similarity as a function of lag:

$$R_{xy}[k] = \sum_{n=-\infty}^{\infty} x[n] y[n-k]$$
$$(3.25)$$

For finite-length discrete signals:
$$R_{xy}[k] = \sum_{n=0}^{N-1-k} x[n] y[n+k], \quad k \geq 0$$
$$(3.26)$$

**Auto-Correlation:**
Special case where $y[n] = x[n]$:
$$R_{xx}[k] = \sum_{n=0}^{N-1-k} x[n] x[n+k]$$
$$(3.27)$$

**Normalized Cross-Correlation:**
$$\rho_{xy}[k] = \frac{R_{xy}[k]}{\sqrt{R_{xx}[0] R_{yy}[0]}}$$
$$(3.28)$$

**Convolution:**
The convolution of signal $x[n]$ with impulse response $h[n]$ produces the system output:
$$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-k]$$
$$(3.29)$$

### 3.5.2 Physical Interpretation in SHM

**Auto-Correlation Applications:**
- **Periodicity detection**: Identifies repeating patterns in structural response
- **Natural frequency estimation**: Peaks in autocorrelation indicate dominant frequencies
- **Signal stationarity assessment**: Changes in autocorrelation reveal non-stationary behavior

**Cross-Correlation Applications:**
- **System identification**: Relationship between input forces and structural response
- **Travel time estimation**: Delay between responses at different structural locations
- **Coherence analysis**: Degree of linear relationship between measurement points

**Convolution Applications:**
- **System modeling**: Structural response prediction from known input and transfer function
- **Matched filtering**: Optimal detection of known signal patterns in noise
- **Impulse response analysis**: Characterization of structural dynamics

### 3.5.3 Impulse Response in Structural Systems

For a linear time-invariant (LTI) structural system, the impulse response $h[n]$ completely characterizes the system dynamics. The relationship between input force $F[n]$ and structural response $x[n]$ is:

$$x[n] = F[n] * h[n]$$
$$(3.30)$$

**Free Vibration Response:**
For an underdamped single-degree-of-freedom system, the impulse response is:
$$h(t) = \frac{1}{m\omega_d} e^{-\zeta\omega_n t} \sin(\omega_d t) u(t)$$
$$(3.31)$$

where:
- $m$ = mass
- $\omega_n$ = natural frequency
- $\zeta$ = damping ratio  
- $\omega_d = \omega_n\sqrt{1-\zeta^2}$ = damped frequency
- $u(t)$ = unit step function

### 3.5.4 Implementation: Correlation and Convolution for Bridge SHM

```python
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal, optimize
import plotly.express as px

class SHMCorrelationAnalyzer:
    """
    Correlation and convolution analysis tools for structural health monitoring.
    """
    
    def __init__(self, fs):
        """
        Initialize analyzer with sampling frequency.
        
        Parameters:
        -----------
        fs : float
            Sampling frequency (Hz)
        """
        self.fs = fs
        self.dt = 1.0 / fs
    
    def compute_autocorrelation(self, signal, max_lag=None, normalize=True):
        """
        Compute autocorrelation function of a signal.
        
        Parameters:
        -----------
        signal : array-like
            Input signal
        max_lag : int, optional
            Maximum lag in samples
        normalize : bool
            Normalize autocorrelation
            
        Returns:
        --------
        lags : array
            Lag values in samples
        autocorr : array
            Autocorrelation values
        """
        
        signal = np.asarray(signal)
        N = len(signal)
        
        if max_lag is None:
            max_lag = N - 1
        
        # Compute full autocorrelation using numpy correlate
        autocorr_full = np.correlate(signal, signal, mode='full')
        
        # Extract positive lags
        center = len(autocorr_full) // 2
        autocorr = autocorr_full[center:center + max_lag + 1]
        lags = np.arange(len(autocorr))
        
        if normalize:
            autocorr = autocorr / autocorr[0]  # Normalize by zero-lag value
        
        return lags, autocorr
    
    def compute_crosscorrelation(self, signal1, signal2, max_lag=None, normalize=True):
        """
        Compute cross-correlation between two signals.
        
        Parameters:
        -----------
        signal1, signal2 : array-like
            Input signals
        max_lag : int, optional
            Maximum lag in samples
        normalize : bool
            Normalize cross-correlation
            
        Returns:
        --------
        lags : array
            Lag values in samples (negative = signal2 leads signal1)
        crosscorr : array
            Cross-correlation values
        """
        
        signal1 = np.asarray(signal1)
        signal2 = np.asarray(signal2)
        
        # Ensure same length
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]
        
        if max_lag is None:
            max_lag = min_len - 1
        
        # Compute full cross-correlation
        crosscorr_full = np.correlate(signal1, signal2, mode='full')
        
        # Extract relevant lags
        center = len(crosscorr_full) // 2
        start_idx = max(0, center - max_lag)
        end_idx = min(len(crosscorr_full), center + max_lag + 1)
        
        crosscorr = crosscorr_full[start_idx:end_idx]
        lags = np.arange(start_idx - center, end_idx - center)
        
        if normalize:
            # Normalize by geometric mean of auto-correlations at zero lag
            norm_factor = np.sqrt(np.sum(signal1**2) * np.sum(signal2**2))
            crosscorr = crosscorr / norm_factor
        
        return lags, crosscorr
    
    def estimate_impulse_response(self, input_signal, output_signal, method='deconvolution'):
        """
        Estimate system impulse response from input-output data.
        
        Parameters:
        -----------
        input_signal : array-like
            System input (e.g., force)
        output_signal : array-like
            System output (e.g., acceleration)
        method : str
            Estimation method ('deconvolution', 'correlation', 'least_squares')
            
        Returns:
        --------
        impulse_response : array
            Estimated impulse response
        """
        
        input_signal = np.asarray(input_signal)
        output_signal = np.asarray(output_signal)
        
        # Ensure same length
        min_len = min(len(input_signal), len(output_signal))
        input_signal = input_signal[:min_len]
        output_signal = output_signal[:min_len]
        
        if method == 'deconvolution':
            # Frequency domain deconvolution with regularization
            # Add small regularization to prevent division by zero
            epsilon = 1e-6 * np.max(np.abs(np.fft.fft(input_signal)))
            
            Input_fft = np.fft.fft(input_signal)
            Output_fft = np.fft.fft(output_signal)
            
            # Regularized deconvolution
            H_fft = Output_fft * np.conj(Input_fft) / (np.abs(Input_fft)**2 + epsilon)
            impulse_response = np.real(np.fft.ifft(H_fft))
            
        elif method == 'correlation':
            # Cross-correlation based estimation
            lags, crosscorr = self.compute_crosscorrelation(input_signal, output_signal, 
                                                          normalize=False)
            _, autocorr_input = self.compute_autocorrelation(input_signal, normalize=False)
            
            # Impulse response is cross-correlation divided by input autocorrelation
            # This is an approximation for white noise input
            if autocorr_input[0] > 0:
                impulse_response = crosscorr[lags >= 0] / autocorr_input[0]
            else:
                impulse_response = crosscorr[lags >= 0]
        
        elif method == 'least_squares':
            # Least squares estimation using Toeplitz matrix
            from scipy.linalg import toeplitz, solve
            
            # Length of impulse response to estimate
            ir_length = min(100, len(input_signal) // 4)
            
            # Create Toeplitz matrix
            if len(input_signal) > ir_length:
                input_padded = np.concatenate([input_signal, np.zeros(ir_length-1)])
                toeplitz_matrix = toeplitz(input_padded[:len(output_signal)], 
                                         input_signal[:ir_length])
                
                # Solve least squares problem
                impulse_response = solve(toeplitz_matrix.T @ toeplitz_matrix + 
                                       1e-6 * np.eye(ir_length),
                                       toeplitz_matrix.T @ output_signal)
            else:
                # Fallback to correlation method
                lags, crosscorr = self.compute_crosscorrelation(input_signal, output_signal)
                impulse_response = crosscorr[lags >= 0]
        
        else:
            raise ValueError("Method must be 'deconvolution', 'correlation', or 'least_squares'")
        
        return impulse_response
    
    def identify_structural_parameters(self, impulse_response, plot_results=False):
        """
        Extract structural parameters from impulse response.
        
        Parameters:
        -----------
        impulse_response : array
            System impulse response
        plot_results : bool
            Plot fitting results
            
        Returns:
        --------
        parameters : dict
            Identified structural parameters
        """
        
        h = np.asarray(impulse_response)
        t = np.arange(len(h)) * self.dt
        
        # Find peaks to estimate frequency
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(np.abs(h), height=0.1*np.max(np.abs(h)))
        
        if len(peaks) > 1:
            # Estimate frequency from peak spacing
            peak_spacing = np.mean(np.diff(peaks)) * self.dt
            freq_estimate = 1 / (2 * peak_spacing)  # For damped oscillation
        else:
            freq_estimate = 1.0  # Default estimate
        
        # Fit exponentially decaying sinusoid
        def damped_sinusoid(t, A, freq, damping, phase):
            return A * np.exp(-damping * t) * np.sin(2 * np.pi * freq * t + phase)
        
        # Initial parameter estimates
        A_init = np.max(np.abs(h))
        freq_init = freq_estimate
        damping_init = 0.05  # 5% damping ratio initial guess
        phase_init = 0
        
        try:
            # Fit the model
            popt, pcov = optimize.curve_fit(
                damped_sinusoid, t, h,
                p0=[A_init, freq_init, damping_init, phase_init],
                bounds=([0, 0.1, 0, -np.pi], [10*A_init, 0.5*self.fs, 1.0, np.pi]),
                maxfev=5000
            )
            
            A_fit, freq_fit, damping_fit, phase_fit = popt
            
            # Calculate damping ratio
            omega_n = 2 * np.pi * freq_fit
            zeta = damping_fit / omega_n
            
            # Calculate quality factor
            Q_factor = 1 / (2 * zeta) if zeta > 0 else np.inf
            
            parameters = {
                'natural_frequency': freq_fit,
                'damping_coefficient': damping_fit,
                'damping_ratio': zeta,
                'Q_factor': Q_factor,
                'amplitude': A_fit,
                'phase': phase_fit,
                'fit_quality': np.corrcoef(h, damped_sinusoid(t, *popt))[0, 1]**2
            }
            
            if plot_results:
                h_fit = damped_sinusoid(t, *popt)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t, y=h, mode='lines', name='Measured',
                                       line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=t, y=h_fit, mode='lines', name='Fitted',
                                       line=dict(color='red', dash='dash')))
                fig.update_layout(title='Impulse Response Parameter Identification',
                                xaxis_title='Time (s)', yaxis_title='Amplitude')
                fig.show()
                
        except:
            # If fitting fails, return basic estimates
            parameters = {
                'natural_frequency': freq_estimate,
                'damping_coefficient': np.nan,
                'damping_ratio': np.nan,
                'Q_factor': np.nan,
                'amplitude': A_init,
                'phase': 0,
                'fit_quality': 0
            }
        
        return parameters
    
    def detect_periodicity(self, signal, min_period=None, max_period=None):
        """
        Detect periodic components in signal using autocorrelation.
        
        Parameters:
        -----------
        signal : array-like
            Input signal
        min_period : float, optional
            Minimum period to search (seconds)
        max_period : float, optional
            Maximum period to search (seconds)
            
        Returns:
        --------
        periods : array
            Detected periods (seconds)
        strengths : array
            Periodicity strengths (0-1)
        """
        
        signal = np.asarray(signal)
        
        if min_period is None:
            min_period = 2 / self.fs  # At least 2 samples
        if max_period is None:
            max_period = len(signal) / (4 * self.fs)  # Max 1/4 signal length
        
        min_lag = int(min_period * self.fs)
        max_lag = int(max_period * self.fs)
        
        # Compute autocorrelation
        lags, autocorr = self.compute_autocorrelation(signal, max_lag=max_lag)
        
        # Find peaks in autocorrelation (excluding zero lag)
        from scipy.signal import find_peaks
        peak_indices, peak_properties = find_peaks(
            autocorr[min_lag:], 
            height=0.1,  # Minimum 10% of zero-lag correlation
            distance=min_lag  # Minimum separation between peaks
        )
        
        # Adjust indices to account for offset
        peak_indices += min_lag
        
        # Convert to periods and strengths
        periods = lags[peak_indices] / self.fs
        strengths = autocorr[peak_indices]
        
        # Sort by strength
        sort_indices = np.argsort(strengths)[::-1]
        periods = periods[sort_indices]
        strengths = strengths[sort_indices]
        
        return periods, strengths

def demonstrate_bridge_correlation_analysis():
    """
    Comprehensive demonstration of correlation analysis for bridge SHM.
    """
    
    # Initialize analyzer
    fs = 200  # Hz
    analyzer = SHMCorrelationAnalyzer(fs)
    
    # Generate realistic bridge scenario
    duration = 60  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Scenario: Bridge with accelerometers at midspan and quarter-span
    # Simulate vehicle crossing events
    
    # Bridge structural parameters
    bridge_freq = 2.5  # Hz - first mode frequency
    damping_ratio = 0.02  # 2% damping
    
    # Vehicle crossing events
    vehicle_times = [10, 25, 40, 50]  # Vehicle crossing times
    vehicle_speeds = [60, 80, 70, 90]  # km/h
    vehicle_weights = [1.0, 1.5, 1.2, 0.8]  # Relative weights
    
    # Generate responses at two locations
    response_midspan = np.zeros_like(t)
    response_quarterspan = np.zeros_like(t)
    
    # Input force signal (for system identification)
    input_force = np.zeros_like(t)
    
    for i, (vt, speed, weight) in enumerate(zip(vehicle_times, vehicle_speeds, vehicle_weights)):
        # Vehicle crossing duration (based on speed and bridge length)
        bridge_length = 30  # meters
        crossing_duration = bridge_length / (speed / 3.6)  # Convert km/h to m/s
        
        # Vehicle reaches quarter-span first, then midspan
        delay_to_midspan = bridge_length / 4 / (speed / 3.6)  # Quarter bridge length
        
        # Create vehicle load time history
        vehicle_mask = (t >= vt) & (t <= vt + crossing_duration)
        load_intensity = weight * 10000  # Newtons
        
        # Gaussian-shaped load as vehicle passes
        t_vehicle = t[vehicle_mask] - vt
        vehicle_load = load_intensity * np.exp(-(t_vehicle - crossing_duration/2)**2 / 
                                             (2 * (crossing_duration/6)**2))
        
        # Add to input force
        input_force[vehicle_mask] += vehicle_load
        
        # Bridge response (simplified single mode)
        omega_n = 2 * np.pi * bridge_freq
        omega_d = omega_n * np.sqrt(1 - damping_ratio**2)
        
        # Response at quarter-span (arrives first)
        for j, load_val in enumerate(vehicle_load):
            t_resp = t[t >= vt + j/fs] - (vt + j/fs)
            if len(t_resp) > 0:
                impulse_resp = (load_val/fs) * np.exp(-damping_ratio * omega_n * t_resp) * \
                              np.sin(omega_d * t_resp) / omega_d
                end_idx = min(len(impulse_resp), len(response_quarterspan) - int((vt + j/fs) * fs))
                start_idx = int((vt + j/fs) * fs)
                response_quarterspan[start_idx:start_idx + end_idx] += impulse_resp[:end_idx]
        
        # Response at midspan (with delay and scaling)
        delay_samples = int(delay_to_midspan * fs)
        for j, load_val in enumerate(vehicle_load):
            t_resp = t[t >= vt + j/fs + delay_to_midspan] - (vt + j/fs + delay_to_midspan)
            if len(t_resp) > 0:
                # Midspan has higher response (mode shape effect)
                impulse_resp = 1.4 * (load_val/fs) * np.exp(-damping_ratio * omega_n * t_resp) * \
                              np.sin(omega_d * t_resp) / omega_d
                end_idx = min(len(impulse_resp), 
                            len(response_midspan) - int((vt + j/fs + delay_to_midspan) * fs))
                start_idx = int((vt + j/fs + delay_to_midspan) * fs)
                if start_idx >= 0:
                    response_midspan[start_idx:start_idx + end_idx] += impulse_resp[:end_idx]
    
    # Add ambient vibration and noise
    ambient = 0.001 * np.random.normal(0, 1, len(t))
    noise_quarter = 0.0005 * np.random.normal(0, 1, len(t))
    noise_mid = 0.0005 * np.random.normal(0, 1, len(t))
    
    response_quarterspan += ambient + noise_quarter
    response_midspan += ambient + noise_mid
    
    # Perform correlation analysis
    print("=== BRIDGE CORRELATION ANALYSIS ===")
    
    # 1. Auto-correlation analysis
    lags_quarter, autocorr_quarter = analyzer.compute_autocorrelation(
        response_quarterspan, max_lag=int(2*fs)  # 2 seconds
    )
    lags_mid, autocorr_mid = analyzer.compute_autocorrelation(
        response_midspan, max_lag=int(2*fs)
    )
    
    # 2. Cross-correlation between locations
    lags_cross, crosscorr = analyzer.compute_crosscorrelation(
        response_quarterspan, response_midspan, max_lag=int(0.5*fs)  # 0.5 seconds
    )
    
    # 3. Periodicity detection
    periods_quarter, strengths_quarter = analyzer.detect_periodicity(
        response_quarterspan, min_period=0.1, max_period=10
    )
    periods_mid, strengths_mid = analyzer.detect_periodicity(
        response_midspan, min_period=0.1, max_period=10
    )
    
    # 4. System identification
    # Use a segment with strong response for system ID
    start_id = int(9 * fs)  # Just before first vehicle
    end_id = int(15 * fs)   # After first vehicle response
    
    force_segment = input_force[start_id:end_id]
    response_segment = response_midspan[start_id:end_id]
    
    if np.std(force_segment) > 1e-6:  # Ensure non-zero input
        impulse_response = analyzer.estimate_impulse_response(
            force_segment, response_segment, method='correlation'
        )
        
        # Identify parameters
        ir_length = min(int(2*fs), len(impulse_response))  # 2 seconds max
        parameters = analyzer.identify_structural_parameters(
            impulse_response[:ir_length]
        )
    else:
        impulse_response = None
        parameters = None
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Bridge Response Signals', 'Auto-Correlation Functions',
            'Cross-Correlation Analysis', 'Periodicity Detection',
            'Input Force and Response', 'System Identification Results'
        ),
        specs=[[{"colspan": 2}, None],
               [{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}]],
        vertical_spacing=0.08
    )
    
    # Response signals
    fig.add_trace(
        go.Scatter(x=t, y=response_quarterspan*1000, mode='lines', 
                  name='Quarter-span', line=dict(color='#2E86AB', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=response_midspan*1000, mode='lines',
                  name='Midspan', line=dict(color='#E07A5F', width=1.5)),
        row=1, col=1
    )
    
    # Mark vehicle crossing times
    for vt in vehicle_times:
        fig.add_vline(x=vt, line_dash="dash", line_color="gray", 
                     annotation_text=f"Vehicle {vehicle_times.index(vt)+1}", row=1, col=1)
    
    # Auto-correlation functions
    fig.add_trace(
        go.Scatter(x=lags_quarter/fs, y=autocorr_quarter, mode='lines',
                  name='Quarter-span ACF', line=dict(color='#2E86AB', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=lags_mid/fs, y=autocorr_mid, mode='lines',
                  name='Midspan ACF', line=dict(color='#E07A5F', width=2)),
        row=2, col=1
    )
    
    # Cross-correlation
    fig.add_trace(
        go.Scatter(x=lags_cross/fs, y=crosscorr, mode='lines',
                  name='Cross-Correlation', line=dict(color='#4ECDC4', width=2)),
        row=2, col=2
    )
    
    # Find peak in cross-correlation for delay estimation
    peak_idx = np.argmax(np.abs(crosscorr))
    delay_estimate = lags_cross[peak_idx] / fs
    fig.add_vline(x=delay_estimate, line_color="red", line_dash="dash",
                 annotation_text=f"Delay: {delay_estimate:.3f}s", row=2, col=2)
    
    # Periodicity detection
    if len(periods_quarter) > 0:
        fig.add_trace(
            go.Bar(x=1/periods_quarter[:5], y=strengths_quarter[:5], 
                  name='Quarter-span Freq', marker_color='#2E86AB', opacity=0.7),
            row=3, col=1
        )
    
    if len(periods_mid) > 0:
        fig.add_trace(
            go.Bar(x=1/periods_mid[:5], y=strengths_mid[:5],
                  name='Midspan Freq', marker_color='#E07A5F', opacity=0.7),
            row=3, col=2
        )
    
    # Input and response for system ID
    t_segment = t[start_id:end_id]
    fig.add_trace(
        go.Scatter(x=t_segment, y=force_segment, mode='lines',
                  name='Input Force', line=dict(color='green', width=2)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_segment, y=response_segment*1000, mode='lines',
                  name='Response', line=dict(color='blue', width=2), yaxis='y2'),
        row=4, col=1
    )
    
    # Impulse response
    if impulse_response is not None:
        t_ir = np.arange(len(impulse_response[:ir_length])) / fs
        fig.add_trace(
            go.Scatter(x=t_ir, y=impulse_response[:ir_length], mode='lines',
                      name='Impulse Response', line=dict(color='purple', width=2)),
            row=4, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="Bridge SHM: Comprehensive Correlation and Convolution Analysis",
        title_font_size=16,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration (mm/s²)", row=1, col=1)
    
    fig.update_xaxes(title_text="Lag (s)", row=2, col=1)
    fig.update_yaxes(title_text="Autocorrelation", row=2, col=1)
    
    fig.update_xaxes(title_text="Lag (s)", row=2, col=2)
    fig.update_yaxes(title_text="Cross-correlation", row=2, col=2)
    
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
    fig.update_yaxes(title_text="Periodicity Strength", row=3, col=1)
    
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=2)
    fig.update_yaxes(title_text="Periodicity Strength", row=3, col=2)
    
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Force (N)", row=4, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=4, col=2)
    fig.update_yaxes(title_text="Impulse Response", row=4, col=2)
    
    fig.show()
    
    # Results summary
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Cross-correlation peak delay: {delay_estimate:.3f} seconds")
    print(f"Maximum cross-correlation: {np.max(np.abs(crosscorr)):.3f}")
    
    if len(periods_quarter) > 0:
        print(f"\nDominant periods (Quarter-span):")
        for i, (period, strength) in enumerate(zip(periods_quarter[:3], strengths_quarter[:3])):
            print(f"  {i+1}. Period: {period:.2f}s, Frequency: {1/period:.2f}Hz, Strength: {strength:.3f}")
    
    if len(periods_mid) > 0:
        print(f"\nDominant periods (Midspan):")
        for i, (period, strength) in enumerate(zip(periods_mid[:3], strengths_mid[:3])):
            print(f"  {i+1}. Period: {period:.2f}s, Frequency: {1/period:.2f}Hz, Strength: {strength:.3f}")
    
    if parameters is not None:
        print(f"\n=== IDENTIFIED SYSTEM PARAMETERS ===")
        print(f"Natural frequency: {parameters['natural_frequency']:.2f} Hz")
        print(f"Damping ratio: {parameters['damping_ratio']:.4f}")
        print(f"Q factor: {parameters['Q_factor']:.1f}")
        print(f"Fit quality (R²): {parameters['fit_quality']:.3f}")
    
    return fig, parameters

# Run correlation analysis demonstration
correlation_demo, identified_params = demonstrate_bridge_correlation_analysis()
```

### 3.5.5 Advanced Applications in Bridge SHM

**Operational Modal Analysis (OMA):**
Cross-correlation functions between response measurements can reveal modal parameters under ambient excitation conditions. The cross-correlation function for two responses under white noise excitation approaches the impulse response function between the measurement points.

**Damage Detection via Correlation Changes:**
Structural damage alters the correlation patterns between measurement points. Changes in cross-correlation peak values, time delays, or overall correlation coefficients can indicate damage progression.

**Travel Time Estimation:**
For wave-based SHM methods, cross-correlation analysis determines wave travel times between sensors, enabling damage localization through changes in wave propagation characteristics.

---

## 3.6 Chapter Summary and Key Takeaways

This chapter has established the fundamental time-domain signal processing techniques essential for structural health monitoring applications. The key concepts and practical implementations covered include:

**Core Concepts Mastered:**
1. **Time-series fundamentals**: Statistical descriptors, stationarity analysis, and signal characterization
2. **Sampling theory**: Nyquist criteria, aliasing prevention, and anti-aliasing filter design
3. **Data preprocessing**: Robust outlier detection, missing data handling, and detrending techniques
4. **Digital filtering**: FIR/IIR filter design, frequency response analysis, and SHM-specific filter banks
5. **Correlation analysis**: Auto-correlation, cross-correlation, and system identification methods

**Practical Skills Developed:**
- Implementation of robust preprocessing pipelines for real-world SHM data
- Design and application of appropriate filters for bridge monitoring applications  
- Extraction of structural parameters from correlation analysis
- Quality assessment and validation of time-domain processing results

**Bridge SHM Applications:**
- Traffic loading detection and characterization
- Structural parameter identification from ambient vibration
- Multi-sensor data fusion and correlation analysis
- Real-time monitoring system development

The time-domain techniques presented form the foundation for frequency-domain analysis (Chapter 5) and advanced signal processing methods covered in subsequent chapters.

---

## 3.7 Exercises

### Exercise 3.1: Statistical Analysis of Bridge Vibration Data
A bridge monitoring system records 10 minutes of acceleration data at 100 Hz sampling rate. The data contains the typical characteristics of bridge response under traffic loading.

**Tasks:**
1. Generate realistic bridge acceleration data with the following components:
   - Primary structural mode at 1.8 Hz with 2% damping
   - Traffic loading events at irregular intervals
   - Measurement noise (SNR = 40 dB)
   - Linear trend due to temperature effects

2. Compute and interpret the following statistical features:
   - Mean, variance, skewness, and kurtosis
   - RMS value and crest factor
   - Rolling statistics with 30-second windows

3. Assess signal stationarity using:
   - Visual inspection of rolling statistics
   - Statistical tests for trend and change points

4. Discuss how each statistical measure relates to structural health assessment.

### Exercise 3.2: Sampling Rate Optimization for Bridge SHM
Design an optimal sampling strategy for a cable-stayed bridge monitoring system considering the following constraints:

**Given Information:**
- Bridge span: 200 meters
- Expected frequency range of interest: 0.2 - 25 Hz
- Available data transmission bandwidth: 1 Mbps
- Number of sensors: 50 accelerometers
- Data resolution: 24 bits per sample

**Tasks:**
1. Determine the minimum theoretical sampling rate based on Nyquist criteria
2. Design an anti-aliasing filter considering practical transition band requirements
3. Calculate the actual sampling rate needed considering filter roll-off characteristics
4. Evaluate the data transmission requirements and optimize the sampling rate
5. Simulate aliasing effects for inadequate sampling rates and demonstrate the importance of proper anti-aliasing

### Exercise 3.3: Robust Data Preprocessing Pipeline
Develop and validate a comprehensive preprocessing pipeline for corrupted bridge monitoring data.

**Scenario:**
You receive 1 hour of bridge acceleration data that contains:
- 15% missing data points (communication dropouts)
- Outliers due to electromagnetic interference (2% of data)
- Linear and quadratic trends from temperature effects
- 50 Hz power line interference
- High-frequency sensor noise above 80 Hz

**Tasks:**
1. Implement a complete preprocessing pipeline that addresses all data quality issues
2. Compare different outlier detection methods (Z-score, modified Z-score, IQR)
3. Evaluate missing data interpolation techniques (linear, cubic spline, autoregressive)
4. Design appropriate filters for trend removal and noise suppression
5. Validate preprocessing effectiveness using synthetic data with known ground truth
6. Quantify the improvement in signal quality using appropriate metrics

### Exercise 3.4: Digital Filter Design for SHM Applications
Design a comprehensive filter bank for a multi-purpose bridge monitoring system.

**Requirements:**
- Sampling frequency: 500 Hz
- Filter types needed: anti-aliasing, structural dynamics, mode isolation, noise removal
- Performance criteria: <0.1 dB ripple in passband, >40 dB stopband attenuation
- Computational efficiency: real-time processing capability

**Tasks:**
1. Design filters using different approaches (Butterworth, Chebyshev, Elliptic)
2. Compare filter characteristics (magnitude response, phase response, group delay)
3. Analyze computational requirements (number of operations per sample)
4. Implement zero-phase filtering for offline analysis
5. Demonstrate filter performance on realistic bridge response signals
6. Develop guidelines for filter selection based on application requirements

### Exercise 3.5: Correlation-Based System Identification
Use correlation analysis to identify structural parameters from bridge response data during a controlled loading test.

**Scenario:**
A bridge undergoes load testing with a known force applied at midspan. Acceleration responses are measured at multiple locations along the bridge deck.

**Given Data:**
- Applied force time history (input)
- Acceleration responses at 5 locations (outputs)  
- Sampling rate: 200 Hz
- Test duration: 30 minutes including free vibration periods

**Tasks:**
1. Implement cross-correlation based system identification
2. Estimate impulse response functions between force and responses
3. Extract modal parameters (frequency, damping) from impulse responses
4. Compare results from different measurement locations
5. Validate identified parameters using frequency-domain methods
6. Assess the influence of noise on identification accuracy
7. Propose methods to improve identification robustness

---

## 3.8 Exercise Solutions

### Solution 3.1: Statistical Analysis of Bridge Vibration Data

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal, stats

# Generate realistic bridge acceleration data
def generate_exercise_data():
    fs = 100  # Hz
    duration = 600  # 10 minutes
    t = np.linspace(0, duration, int(fs * duration))
    
    # 1. Primary structural mode (1.8 Hz, 2% damping)
    omega_n = 2 * np.pi * 1.8
    zeta = 0.02
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    
    # Ambient excitation response
    white_noise = np.random.normal(0, 1, len(t))
    structural_response = signal.lfilter([omega_n**2], [1, 2*zeta*omega_n, omega_n**2], white_noise)
    structural_response = structural_response * 0.01  # Scale to reasonable amplitude
    
    # 2. Traffic loading events (irregular intervals)
    traffic_times = np.random.exponential(45, 20)  # Average 45 seconds between vehicles
    traffic_times = np.cumsum(traffic_times)
    traffic_times = traffic_times[traffic_times < duration]
    
    traffic_response = np.zeros_like(t)
    for vehicle_time in traffic_times:
        # Vehicle response - decaying oscillation
        vehicle_mask = t >= vehicle_time
        t_vehicle = t[vehicle_mask] - vehicle_time
        vehicle_amplitude = np.random.uniform(0.02, 0.08)  # Variable vehicle loading
        vehicle_freq = np.random.uniform(1.5, 2.5)  # Slight frequency variation
        
        vehicle_contribution = (vehicle_amplitude * np.exp(-0.1 * t_vehicle) * 
                              np.sin(2 * np.pi * vehicle_freq * t_vehicle))
        traffic_response[vehicle_mask] += vehicle_contribution
    
    # 3. Linear trend (temperature effect)
    linear_trend = 0.00005 * t  # Gradual drift
    
    # 4. Measurement noise (SNR = 40 dB)
    signal_power = np.mean((structural_response + traffic_response)**2)
    noise_power = signal_power / (10**(40/10))  # 40 dB SNR
    measurement_noise = np.random.normal(0, np.sqrt(noise_power), len(t))
    
    # Combine all components
    acceleration = structural_response + traffic_response + linear_trend + measurement_noise
    
    return t, acceleration, {
        'structural': structural_response,
        'traffic': traffic_response,
        'trend': linear_trend,
        'noise': measurement_noise
    }

# Statistical analysis functions
def compute_comprehensive_statistics(signal):
    """Compute comprehensive statistical features."""
    
    stats_dict = {
        'Mean': np.mean(signal),
        'Variance': np.var(signal, ddof=1),
        'Standard Deviation': np.std(signal, ddof=1),
        'Skewness': stats.skew(signal),
        'Kurtosis': stats.kurtosis(signal),
        'RMS': np.sqrt(np.mean(signal**2)),
        'Peak Value': np.max(np.abs(signal)),
        'Crest Factor': np.max(np.abs(signal)) / np.sqrt(np.mean(signal**2)),
        'Shape Factor': np.sqrt(np.mean(signal**2)) / np.mean(np.abs(signal)),
        'Impulse Factor': np.max(np.abs(signal)) / np.mean(np.abs(signal)),
        'Peak-to-Peak': np.ptp(signal),
        'Energy': np.sum(signal**2),
        'Power': np.mean(signal**2)
    }
    
    return stats_dict

def rolling_statistics(signal, window_size, fs):
    """Compute rolling statistics."""
    
    n_windows = len(signal) // window_size
    window_centers = []
    rolling_stats = {'mean': [], 'std': [], 'rms': [], 'peak': []}
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        window_data = signal[start_idx:end_idx]
        
        window_centers.append((start_idx + end_idx) / 2 / fs)
        rolling_stats['mean'].append(np.mean(window_data))
        rolling_stats['std'].append(np.std(window_data))
        rolling_stats['rms'].append(np.sqrt(np.mean(window_data**2)))
        rolling_stats['peak'].append(np.max(np.abs(window_data)))
    
    return np.array(window_centers), rolling_stats

# Generate and analyze data
t, acceleration, components = generate_exercise_data()
fs = 100

# Compute statistics
overall_stats = compute_comprehensive_statistics(acceleration)

# Rolling statistics (30-second windows)
window_duration = 30  # seconds
window_size = window_duration * fs
window_centers, rolling_stats = rolling_statistics(acceleration, window_size, fs)

# Create visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Bridge Acceleration Time Series', 'Statistical Distribution',
        'Signal Components', 'Rolling Statistics',
        'Stationarity Assessment', 'Correlation Analysis'
    ),
    specs=[[{"colspan": 2}, None],
           [{"colspan": 1}, {"colspan": 1}],
           [{"colspan": 1}, {"colspan": 1}]],
    vertical_spacing=0.1
)

# Time series
fig.add_trace(
    go.Scatter(x=t, y=acceleration*1000, mode='lines', name='Total Signal',
              line=dict(color='#2E86AB', width=1)),
    row=1, col=1
)

# Statistical distribution
fig.add_trace(
    go.Histogram(x=acceleration*1000, nbinsx=50, name='Distribution',
                marker_color='#A8E6CF', opacity=0.7, histnorm='probability density'),
    row=1, col=1
)

# Add normal distribution overlay
x_norm = np.linspace(np.min(acceleration*1000), np.max(acceleration*1000), 100)
y_norm = stats.norm.pdf(x_norm, overall_stats['Mean']*1000, overall_stats['Standard Deviation']*1000)
fig.add_trace(
    go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Fit',
              line=dict(color='red', dash='dash')),
    row=1, col=1
)

# Signal components
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
component_names = ['Structural', 'Traffic', 'Trend', 'Noise']
for i, (comp_name, color) in enumerate(zip(['structural', 'traffic', 'trend', 'noise'], colors)):
    fig.add_trace(
        go.Scatter(x=t[:5000], y=components[comp_name][:5000]*1000, mode='lines',
                  name=comp_name.title(), line=dict(color=color, width=1.5)),
        row=2, col=1
    )

# Rolling statistics
fig.add_trace(
    go.Scatter(x=window_centers, y=np.array(rolling_stats['mean'])*1000, mode='lines+markers',
              name='Rolling Mean', line=dict(color='#2E86AB', width=2)),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=window_centers, y=np.array(rolling_stats['std'])*1000, mode='lines+markers',
              name='Rolling Std', line=dict(color='#E07A5F', width=2)),
    row=2, col=2
)

# Stationarity tests - trend analysis
detrended_signal = signal.detrend(acceleration)
fig.add_trace(
    go.Scatter(x=t, y=acceleration*1000, mode='lines', name='Original',
              line=dict(color='#2E86AB', width=1.5)),
    row=3, col=1
)
fig.add_trace(
    go.Scatter(x=t, y=detrended_signal*1000, mode='lines', name='Detrended',
              line=dict(color='#E07A5F', width=1.5)),
    row=3, col=1
)

# Autocorrelation for stationarity assessment
lags = np.arange(0, 500)  # 5 seconds at 100 Hz
autocorr = np.correlate(acceleration, acceleration, mode='full')
autocorr = autocorr[autocorr.size // 2:][:500]
autocorr = autocorr / autocorr[0]

fig.add_trace(
    go.Scatter(x=lags/fs, y=autocorr, mode='lines', name='Autocorrelation',
              line=dict(color='#4ECDC4', width=2)),
    row=3, col=2
)

fig.update_layout(
    height=1000,
    title_text="Exercise 3.1 Solution: Comprehensive Statistical Analysis",
    showlegend=True
)

fig.show()

# Print statistical results
print("=== EXERCISE 3.1 SOLUTION ===")
print("\n1. OVERALL STATISTICAL FEATURES:")
for feature, value in overall_stats.items():
    if 'Factor' in feature:
        print(f"   {feature}: {value:.3f}")
    else:
        print(f"   {feature}: {value:.6f}")

print(f"\n2. STATIONARITY ASSESSMENT:")
print(f"   Mean of rolling means: {np.mean(rolling_stats['mean']):.6f}")
print(f"   Std of rolling means: {np.std(rolling_stats['mean']):.6f}")
print(f"   Coefficient of variation: {np.std(rolling_stats['mean'])/np.mean(rolling_stats['mean']):.3f}")

if np.std(rolling_stats['mean'])/np.mean(rolling_stats['mean']) < 0.1:
    print("   Signal appears relatively stationary in mean")
else:
    print("   Signal shows non-stationary behavior in mean")

print(f"\n3. INTERPRETATION FOR STRUCTURAL HEALTH:")
print(f"   - RMS value ({overall_stats['RMS']*1000:.3f} mm/s²) indicates overall vibration level")
print(f"   - Crest factor ({overall_stats['Crest Factor']:.2f}) suggests {'impulsive' if overall_stats['Crest Factor'] > 4 else 'continuous'} loading")
print(f"   - Skewness ({overall_stats['Skewness']:.3f}) indicates {'right-skewed' if overall_stats['Skewness'] > 0 else 'left-skewed' if overall_stats['Skewness'] < 0 else 'symmetric'} distribution")
print(f"   - Kurtosis ({overall_stats['Kurtosis']:.3f}) suggests {'heavy-tailed' if overall_stats['Kurtosis'] > 0 else 'light-tailed'} behavior")
```

### Solution 3.2: Sampling Rate Optimization

```python
def design_sampling_strategy():
    """Design optimal sampling strategy for cable-stayed bridge."""
    
    # Given constraints
    f_max_interest = 25  # Hz
    n_sensors = 50
    bits_per_sample = 24
    bandwidth_mbps = 1  # Mbps
    
    # 1. Theoretical minimum sampling rate
    f_s_min_theoretical = 2 * f_max_interest  # 50 Hz
    
    # 2. Practical considerations for anti-aliasing filter
    # Butterworth filter design - need transition band
    transition_bandwidth = 0.2 * f_max_interest  # 20% of max frequency
    f_cutoff = f_max_interest  # Filter cutoff
    f_s_practical = 2 * (f_cutoff + transition_bandwidth)  # Include transition band
    
    # 3. Filter design for different sampling rates
    sampling_rates = [60, 80, 100, 125, 150, 200]
    
    print("=== EXERCISE 3.2 SOLUTION ===")
    print(f"Frequency range of interest: 0.2 - {f_max_interest} Hz")
    print(f"Theoretical minimum sampling rate: {f_s_min_theoretical} Hz")
    print(f"Practical minimum sampling rate: {f_s_practical} Hz")
    
    print(f"\nSAMPLING RATE ANALYSIS:")
    
    for fs in sampling_rates:
        # Data rate calculation
        data_rate_per_sensor = fs * bits_per_sample  # bits per second
        total_data_rate = n_sensors * data_rate_per_sensor  # bits per second
        total_data_rate_mbps = total_data_rate / 1e6  # Mbps
        
        # Filter requirements
        nyquist = fs / 2
        transition_ratio = transition_bandwidth / nyquist
        
        print(f"\nSampling Rate: {fs} Hz")
        print(f"  Nyquist frequency: {nyquist} Hz")
        print(f"  Safety margin: {(nyquist - f_max_interest)/f_max_interest*100:.1f}%")
        print(f"  Data rate per sensor: {data_rate_per_sensor/1000:.1f} kbps")
        print(f"  Total data rate: {total_data_rate_mbps:.2f} Mbps")
        print(f"  Bandwidth feasible: {'Yes' if total_data_rate_mbps <= bandwidth_mbps else 'No'}")
        
        # Design anti-aliasing filter
        if nyquist > f_max_interest:
            filter_order = 6  # Typical choice
            cutoff_freq = f_max_interest / nyquist  # Normalized frequency
            
            # Create frequency response
            sos = signal.butter(filter_order, cutoff_freq, output='sos')
            frequencies = np.logspace(-2, 0, 1000) * nyquist
            w, h = signal.sosfreqz(sos, worN=frequencies, fs=fs)
            
            # Find attenuation at Nyquist frequency
            nyquist_idx = np.argmin(np.abs(frequencies - nyquist))
            attenuation_at_nyquist = -20 * np.log10(np.abs(h[nyquist_idx]))
            
            print(f"  Filter attenuation at Nyquist: {attenuation_at_nyquist:.1f} dB")
    
    # Recommended sampling rate
    recommended_fs = 100  # Hz - good balance of all factors
    
    print(f"\nRECOMMENDED SAMPLING RATE: {recommended_fs} Hz")
    print(f"Justification:")
    print(f"  - Provides {recommended_fs/2 - f_max_interest} Hz safety margin")
    print(f"  - Total data rate: {n_sensors * recommended_fs * bits_per_sample / 1e6:.2f} Mbps (within bandwidth)")
    print(f"  - Allows practical anti-aliasing filter design")
    print(f"  - Standard sampling rate for structural monitoring")

design_sampling_strategy()
```

### Solution 3.3: Robust Data Preprocessing Pipeline

```python
def comprehensive_preprocessing_validation():
    """Validate preprocessing pipeline with synthetic corrupted data."""
    
    # Generate clean ground truth signal
    fs = 200
    duration = 3600  # 1 hour
    t = np.linspace(0, duration, int(fs * duration))
    
    # Clean bridge response - multiple modes
    modes = [
        {'freq': 1.2, 'damping': 0.02, 'amplitude': 0.01},
        {'freq': 3.8, 'damping': 0.025, 'amplitude': 0.006},
        {'freq': 7.2, 'damping': 0.03, 'amplitude': 0.003}
    ]
    
    clean_signal = np.zeros_like(t)
    for mode in modes:
        omega_n = 2 * np.pi * mode['freq']
        zeta = mode['damping']
        
        # Generate mode response to ambient excitation
        white_noise = np.random.normal(0, 1, len(t))
        mode_response = signal.lfilter([omega_n**2], [1, 2*zeta*omega_n, omega_n**2], white_noise)
        clean_signal += mode['amplitude'] * mode_response / np.std(mode_response)
    
    # Add realistic corruptions
    corrupted_signal = clean_signal.copy()
    
    # 1. Missing data (15%)
    n_missing = int(0.15 * len(t))
    missing_indices = np.random.choice(len(t), n_missing, replace=False)
    corrupted_signal[missing_indices] = np.nan
    
    # 2. Outliers (2%)
    n_outliers = int(0.02 * len(t))
    outlier_indices = np.random.choice(len(t), n_outliers, replace=False)
    outlier_magnitudes = np.random.uniform(10, 50, n_outliers) * np.std(clean_signal)
    outlier_signs = np.random.choice([-1, 1], n_outliers)
    corrupted_signal[outlier_indices] += outlier_magnitudes * outlier_signs
    
    # 3. Linear and quadratic trends
    linear_trend = 0.00001 * t
    quadratic_trend = 1e-8 * t**2
    corrupted_signal += linear_trend + quadratic_trend
    
    # 4. 50 Hz interference
    power_line = 0.002 * np.sin(2 * np.pi * 50 * t)
    corrupted_signal += power_line
    
    # 5. High-frequency noise
    hf_noise = signal.sosfilt(
        signal.butter(4, [80, 100], btype='band', fs=fs, output='sos'),
        np.random.normal(0, 1, len(t))
    ) * 0.001
    corrupted_signal += hf_noise
    
    # Apply preprocessing pipeline
    preprocessor = SHMDataPreprocessor(fs)
    
    processed_signal, processing_info = preprocessor.preprocess_pipeline(
        corrupted_signal,
        outlier_detection=True,
        missing_data_handling=True,
        detrending=True,
        filtering=True,
        outlier_method='modified_zscore',
        outlier_threshold=3.0,
        missing_method='cubic',
        detrend_method='polynomial',
        poly_order=2,
        filter_type='bandpass',
        low_freq=0.05,
        high_freq=75,
        filter_order=4
    )
    
    # Performance evaluation
    # Align signals (account for potential length differences)
    min_length = min(len(clean_signal), len(processed_signal))
    clean_aligned = clean_signal[:min_length]
    processed_aligned = processed_signal[:min_length]
    
    # Remove trend from clean signal for fair comparison
    clean_detrended = signal.detrend(clean_aligned, type='linear')
    
    # Metrics
    rmse = np.sqrt(np.mean((clean_detrended - processed_aligned)**2))
    correlation = np.corrcoef(clean_detrended, processed_aligned)[0, 1]
    snr_improvement = 10 * np.log10(np.var(clean_detrended) / np.var(clean_detrended - processed_aligned))
    
    print("=== EXERCISE 3.3 SOLUTION ===")
    print(f"PREPROCESSING PERFORMANCE:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  SNR improvement: {snr_improvement:.2f} dB")
    print(f"  Signal recovery quality: {'Excellent' if correlation > 0.95 else 'Good' if correlation > 0.85 else 'Fair'}")
    
    return clean_signal, corrupted_signal, processed_signal, processing_info

# Run preprocessing validation
clean, corrupted, processed, info = comprehensive_preprocessing_validation()
```

### Solutions 3.4 and 3.5

*[Due to length constraints, Solutions 3.4 and 3.5 would follow similar comprehensive implementation patterns, focusing on filter design comparison and correlation-based system identification respectively.]*

---

## 3.9 References and Further Reading

1. **Farrar, C. R., & Worden, K.** (2013). *Structural Health Monitoring: A Machine Learning Perspective*. John Wiley & Sons. [Comprehensive reference on SHM fundamentals]

2. **Maia, N. M. M., et al.** (1997). *Theoretical and Experimental Modal Analysis*. Research Studies Press. [Modal analysis theory and practice]

3. **Proakis, J. G., & Manolakis, D. K.** (2007). *Digital Signal Processing: Principles, Algorithms, and Applications* (4th ed.). Pearson Prentice Hall. [Digital signal processing fundamentals]

4. **Oppenheim, A. V., & Schafer, R. W.** (2010). *Discrete-Time Signal Processing* (3rd ed.). Pearson. [Advanced signal processing theory]

5. **Brownjohn, J. M. W.** (2007). Structural health monitoring of civil infrastructure. *Philosophical Transactions of the Royal Society A*, 365(1851), 589-622. [Bridge SHM overview]

6. **Rainieri, C., & Fabbrocino, G.** (2014). *Operational Modal Analysis of Civil Engineering Structures*. Springer. [Operational modal analysis techniques]

7. **Worden, K., & Tomlinson, G. R.** (2001). *Nonlinearity in Structural Dynamics: Detection, Identification and Modelling*. Institute of Physics Publishing. [Nonlinear system identification]

8. **Sony, S., et al.** (2019). A literature review of next-generation smart sensing technology in structural health monitoring. *Structural Control and Health Monitoring*, 26(3), e2321. [Modern SHM sensing technologies]

9. **Bao, Y., et al.** (2019). The state of the art of data science and engineering in structural health monitoring. *Engineering*, 5(2), 234-242. [Data science approaches in SHM]

10. **Entezami, A., et al.** (2020). Big data analytics and structural health monitoring: A statistical pattern recognition-based approach. *Sensors*, 20(8), 2328. [Statistical pattern recognition in SHM]

**Key Journal Sources:**
- *Structural Health Monitoring* (SAGE Publications)
- *Smart Materials and Structures* (IOP Publishing)  
- *Journal of Sound and Vibration* (Elsevier)
- *Mechanical Systems and Signal Processing* (Elsevier)
- *Computer-Aided Civil and Infrastructure Engineering* (Wiley)

**Online Resources:**
- MATLAB Signal Processing Toolbox Documentation
- SciPy Signal Processing Reference
- Los Alamos National Laboratory SHM Literature
- International Association for Structural Control and Monitoring (IASCM)

---

*This concludes Chapter 3: Time-Domain Signal Processing for Structural Health Monitoring. The next chapter will build upon these fundamentals to explore frequency-domain analysis techniques essential for modal identification and spectral analysis in bridge monitoring applications.*
    
    