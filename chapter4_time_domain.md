# Chapter 4: Time-Domain Signal Processing

**Mohammad Talebi-Kalaleh – University of Alberta**

---

## 4.1 Introduction and Motivation

Time-domain signal processing forms the cornerstone of structural health monitoring (SHM) systems, providing the fundamental tools needed to extract meaningful information from raw sensor measurements. When accelerometers and strain gauges capture the dynamic response of a bridge structure, the resulting time-series data contains a wealth of information about structural behavior, environmental conditions, and potential damage indicators.

Understanding how structures respond to various excitations—from traffic loading to wind forces—requires sophisticated signal processing techniques that can separate signal from noise, identify characteristic patterns, and detect subtle changes that may indicate structural degradation. This chapter introduces the essential time-domain methods that form the foundation for all subsequent analysis in structural health monitoring.

### Why Time-Domain Processing Matters in SHM

Consider a typical bridge monitoring scenario: accelerometers placed at various locations continuously record structural vibrations at sampling rates of 100-1000 Hz, generating massive amounts of time-series data. Raw sensor measurements often contain:

- **Structural response signals** that reveal modal characteristics and dynamic behavior
- **Environmental noise** from wind, temperature effects, and electromagnetic interference  
- **Loading signatures** from vehicle traffic, pedestrians, and ambient excitation
- **Measurement artifacts** due to sensor drift, calibration errors, and data transmission issues

The challenge lies in extracting the structural information while minimizing the influence of noise and artifacts. Time-domain processing provides the mathematical framework to achieve this separation effectively.

### Chapter Overview

This chapter builds understanding progressively, starting with fundamental concepts of time-series analysis and advancing to sophisticated preprocessing techniques. Each method is presented with intuitive explanations, mathematical formulations, and practical Python implementations using realistic bridge monitoring data.

We'll explore how correlation analysis reveals hidden relationships between sensor measurements, how convolution operations enable system identification, and how proper filtering and sampling strategies ensure high-quality data for downstream analysis. The techniques covered here are essential preparation for the frequency-domain analysis methods presented in Chapter 5.

---

## 4.2 Time-Series Fundamentals

### 4.2.1 Understanding Time-Series Data in SHM

A time-series represents the evolution of a measured quantity over time. In structural health monitoring, we typically work with discrete-time signals sampled at regular intervals:

```svg
<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .axis { stroke: #333; stroke-width: 2; }
      .grid { stroke: #ddd; stroke-width: 1; }
      .signal { stroke: #2E86AB; stroke-width: 3; fill: none; }
      .samples { fill: #F24236; }
      .text { font-family: 'Arial', sans-serif; font-size: 14px; fill: #333; }
      .title { font-family: 'Arial', sans-serif; font-size: 16px; font-weight: bold; fill: #333; }
    </style>
  </defs>
  
  <!-- Grid -->
  <g class="grid">
    <line x1="60" y1="50" x2="60" y2="350"/>
    <line x1="60" y1="350" x2="740" y2="350"/>
    
    <!-- Vertical grid lines -->
    <line x1="140" y1="50" x2="140" y2="350"/>
    <line x1="220" y1="50" x2="220" y2="350"/>
    <line x1="300" y1="50" x2="300" y2="350"/>
    <line x1="380" y1="50" x2="380" y2="350"/>
    <line x1="460" y1="50" x2="460" y2="350"/>
    <line x1="540" y1="50" x2="540" y2="350"/>
    <line x1="620" y1="50" x2="620" y2="350"/>
    <line x1="700" y1="50" x2="700" y2="350"/>
    
    <!-- Horizontal grid lines -->
    <line x1="60" y1="100" x2="740" y2="100"/>
    <line x1="60" y1="150" x2="740" y2="150"/>
    <line x1="60" y1="200" x2="740" y2="200"/>
    <line x1="60" y1="250" x2="740" y2="250"/>
    <line x1="60" y1="300" x2="740" y2="300"/>
  </g>
  
  <!-- Axes -->
  <g class="axis">
    <line x1="60" y1="350" x2="740" y2="350"/>
    <line x1="60" y1="50" x2="60" y2="350"/>
    <polygon points="735,345 745,350 735,355" fill="#333"/>
    <polygon points="55,55 60,45 65,55" fill="#333"/>
  </g>
  
  <!-- Continuous signal -->
  <path class="signal" d="M 60,200 Q 100,180 140,220 Q 180,280 220,200 Q 260,120 300,180 Q 340,240 380,160 Q 420,100 460,200 Q 500,280 540,180 Q 580,120 620,220 Q 660,280 700,200"/>
  
  <!-- Sample points -->
  <g class="samples">
    <circle cx="60" cy="200" r="4"/>
    <circle cx="140" cy="220" r="4"/>
    <circle cx="220" cy="200" r="4"/>
    <circle cx="300" cy="180" r="4"/>
    <circle cx="380" cy="160" r="4"/>
    <circle cx="460" cy="200" r="4"/>
    <circle cx="540" cy="180" r="4"/>
    <circle cx="620" cy="220" r="4"/>
    <circle cx="700" cy="200" r="4"/>
  </g>
  
  <!-- Sample stems -->
  <g stroke="#F24236" stroke-width="2">
    <line x1="60" y1="200" x2="60" y2="350"/>
    <line x1="140" y1="220" x2="140" y2="350"/>
    <line x1="220" y1="200" x2="220" y2="350"/>
    <line x1="300" y1="180" x2="300" y2="350"/>
    <line x1="380" y1="160" x2="380" y2="350"/>
    <line x1="460" y1="200" x2="460" y2="350"/>
    <line x1="540" y1="180" x2="540" y2="350"/>
    <line x1="620" y1="220" x2="620" y2="350"/>
    <line x1="700" y1="200" x2="700" y2="350"/>
  </g>
  
  <!-- Labels -->
  <text x="400" y="380" text-anchor="middle" class="text">Time (s)</text>
  <text x="25" y="200" text-anchor="middle" class="text" transform="rotate(-90 25 200)">Acceleration (m/s²)</text>
  <text x="400" y="30" text-anchor="middle" class="title">Discrete-Time Signal Sampling</text>
  
  <!-- Sample labels -->
  <text x="60" y="370" text-anchor="middle" class="text">0</text>
  <text x="140" y="370" text-anchor="middle" class="text">0.1</text>
  <text x="220" y="370" text-anchor="middle" class="text">0.2</text>
  <text x="300" y="370" text-anchor="middle" class="text">0.3</text>
  <text x="380" y="370" text-anchor="middle" class="text">0.4</text>
  <text x="460" y="370" text-anchor="middle" class="text">0.5</text>
  <text x="540" y="370" text-anchor="middle" class="text">0.6</text>
  <text x="620" y="370" text-anchor="middle" class="text">0.7</text>
  <text x="700" y="370" text-anchor="middle" class="text">0.8</text>
  
  <!-- Legend -->
  <g>
    <line x1="550" y1="80" x2="580" y2="80" class="signal"/>
    <text x="590" y="85" class="text">Continuous Signal</text>
    <circle cx="565" cy="100" r="4" class="samples"/>
    <text x="580" y="105" class="text">Sampled Points</text>
  </g>
</svg>
```

**Figure 4.1:** Discrete-time signal sampling showing the relationship between continuous structural response and digitally sampled measurements.

The mathematical representation of a discrete-time signal is:

$$x[n] = x(nT_s), \quad n = 0, 1, 2, \ldots, N-1 \tag{4.1}$$

where $x[n]$ represents the signal value at discrete time index $n$, $T_s$ is the sampling period, and $N$ is the total number of samples.

### 4.2.2 Key Signal Characteristics

**Stationarity**: A signal is stationary if its statistical properties (mean, variance, autocorrelation) remain constant over time. Most bridge responses under ambient excitation exhibit quasi-stationary behavior over short time windows.

**Periodicity**: Many structural responses contain periodic components due to regular loading patterns (vehicle traffic, pedestrian footsteps) or structural characteristics (modal frequencies).

**Trend**: Long-term variations in signal amplitude or baseline, often caused by temperature effects, sensor drift, or gradual structural changes.

### 4.2.3 Practical Implementation: Basic Time-Series Analysis

Let's examine real bridge acceleration data to understand these fundamental concepts:

```python
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Generate realistic bridge acceleration data
def generate_bridge_acceleration(duration=60, fs=100):
    """
    Generate synthetic bridge acceleration data with realistic characteristics
    
    Parameters:
    duration: Time duration in seconds
    fs: Sampling frequency in Hz
    
    Returns:
    t: Time vector
    acc: Acceleration time series
    """
    t = np.linspace(0, duration, int(duration * fs))
    
    # Modal components (typical bridge frequencies)
    f1, f2, f3 = 2.3, 5.7, 8.9  # Hz - fundamental modes
    mode1 = 0.5 * np.sin(2*np.pi*f1*t) * np.exp(-0.05*t)
    mode2 = 0.3 * np.sin(2*np.pi*f2*t) * np.exp(-0.02*t)
    mode3 = 0.2 * np.sin(2*np.pi*f3*t) * np.exp(-0.01*t)
    
    # Traffic loading (intermittent pulses)
    traffic = np.zeros_like(t)
    vehicle_times = [10, 25, 35, 50]  # Vehicle crossing times
    for tv in vehicle_times:
        mask = (t >= tv) & (t <= tv + 3)  # 3-second crossing
        traffic[mask] += 2.0 * np.exp(-((t[mask] - tv - 1.5)**2) / 0.5) * np.sin(2*np.pi*12*t[mask])
    
    # Wind loading (low-frequency random)
    wind = 0.8 * np.sin(2*np.pi*0.1*t) + 0.4 * np.sin(2*np.pi*0.3*t)
    
    # Measurement noise
    noise = 0.15 * np.random.randn(len(t))
    
    # Temperature drift (very slow trend)
    temp_drift = 0.02 * t / duration
    
    # Combine all components
    acc = mode1 + mode2 + mode3 + traffic + wind + noise + temp_drift
    
    return t, acc

# Generate sample data
time, acceleration = generate_bridge_acceleration(duration=60, fs=100)

# Create comprehensive time-series plot
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=['Full Time Series', 'Detailed View (10-20s)', 
                   'Statistical Distribution', 'Running Statistics',
                   'Autocorrelation Function', 'Signal Components'],
    specs=[[{"colspan": 2}, None],
           [{}, {}],
           [{}, {}]]
)

# Full time series
fig.add_trace(
    go.Scatter(x=time, y=acceleration, mode='lines', name='Bridge Acceleration',
              line=dict(color='#2E86AB', width=2)),
    row=1, col=1
)

# Detailed view
detail_mask = (time >= 10) & (time <= 20)
fig.add_trace(
    go.Scatter(x=time[detail_mask], y=acceleration[detail_mask], 
              mode='lines+markers', name='Detailed View',
              line=dict(color='#F24236', width=2),
              marker=dict(size=4)),
    row=2, col=1
)

# Statistical distribution
fig.add_trace(
    go.Histogram(x=acceleration, nbinsx=50, name='Distribution',
                marker_color='#A23B72', opacity=0.7),
    row=2, col=2
)

# Running statistics (using 1000-point windows)
window_size = 1000
running_mean = pd.Series(acceleration).rolling(window_size, center=True).mean()
running_std = pd.Series(acceleration).rolling(window_size, center=True).std()

fig.add_trace(
    go.Scatter(x=time, y=running_mean, mode='lines', name='Running Mean',
              line=dict(color='#F18F01', width=3)),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(x=time, y=running_std, mode='lines', name='Running Std',
              line=dict(color='#C73E1D', width=3)),
    row=3, col=1
)

# Autocorrelation function
max_lag = 500  # 5 seconds at 100 Hz
lags = np.arange(-max_lag, max_lag+1)
autocorr = np.correlate(acceleration - np.mean(acceleration), 
                       acceleration - np.mean(acceleration), mode='full')
autocorr = autocorr[autocorr.size//2-max_lag:autocorr.size//2+max_lag+1]
autocorr = autocorr / autocorr[max_lag]  # Normalize

fig.add_trace(
    go.Scatter(x=lags/100, y=autocorr, mode='lines', name='Autocorrelation',
              line=dict(color='#8338EC', width=2)),
    row=3, col=2
)

# Update layout
fig.update_layout(
    height=800,
    title_text="<b>Time-Series Analysis of Bridge Acceleration Data</b>",
    title_x=0.5,
    showlegend=False,
    font=dict(size=12)
)

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=1)
fig.update_xaxes(title_text="Acceleration (m/s²)", row=2, col=2)
fig.update_yaxes(title_text="Frequency", row=2, col=2)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_yaxes(title_text="Statistics", row=3, col=1)
fig.update_xaxes(title_text="Lag (s)", row=3, col=2)
fig.update_yaxes(title_text="Autocorrelation", row=3, col=2)

fig.show()

# Calculate basic statistics
print("=== TIME-SERIES STATISTICS ===")
print(f"Duration: {time[-1]:.1f} seconds")
print(f"Sampling Rate: {1/(time[1]-time[0]):.0f} Hz")
print(f"Mean Acceleration: {np.mean(acceleration):.4f} m/s²")
print(f"Standard Deviation: {np.std(acceleration):.4f} m/s²")
print(f"Peak-to-Peak: {np.ptp(acceleration):.4f} m/s²")
print(f"RMS Value: {np.sqrt(np.mean(acceleration**2)):.4f} m/s²")
print(f"Skewness: {pd.Series(acceleration).skew():.3f}")
print(f"Kurtosis: {pd.Series(acceleration).kurtosis():.3f}")
```

This implementation demonstrates several key concepts:

1. **Multi-component signals**: Real structural responses contain multiple frequency components from different sources
2. **Non-stationarity**: Running statistics reveal time-varying signal characteristics  
3. **Autocorrelation structure**: Reveals periodic patterns and correlation time scales
4. **Statistical distribution**: Shows whether the signal follows Gaussian statistics

---

## 4.3 Correlation and Convolution

### 4.3.1 Understanding Correlation in SHM

Correlation quantifies the similarity between signals or between a signal and itself at different time lags. In structural health monitoring, correlation analysis reveals:

- **Spatial correlation**: How sensors at different locations respond together
- **Temporal correlation**: How a signal relates to its past values
- **Cross-correlation**: Relationships between different measurement types (acceleration vs. strain)

```svg
<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .axis { stroke: #333; stroke-width: 2; }
      .grid { stroke: #eee; stroke-width: 1; }
      .signal1 { stroke: #2E86AB; stroke-width: 3; fill: none; }
      .signal2 { stroke: #F24236; stroke-width: 3; fill: none; }
      .correlation { stroke: #8338EC; stroke-width: 4; fill: none; }
      .text { font-family: 'Arial', sans-serif; font-size: 12px; fill: #333; }
      .title { font-family: 'Arial', sans-serif; font-size: 14px; font-weight: bold; fill: #333; }
      .label { font-family: 'Arial', sans-serif; font-size: 11px; fill: #666; }
    </style>
  </defs>
  
  <!-- Background -->
  <rect width="800" height="500" fill="#fafafa"/>
  
  <!-- Title -->
  <text x="400" y="25" text-anchor="middle" class="title">Cross-Correlation Analysis Between Two Bridge Sensors</text>
  
  <!-- Signal 1 Panel -->
  <g transform="translate(50, 50)">
    <text x="150" y="15" text-anchor="middle" class="text">Sensor 1 (Mid-span)</text>
    
    <!-- Grid -->
    <g class="grid">
      <line x1="0" y1="0" x2="300" y2="0"/>
      <line x1="0" y1="30" x2="300" y2="30"/>
      <line x1="0" y1="60" x2="300" y2="60"/>
      <line x1="0" y1="90" x2="300" y2="90"/>
    </g>
    
    <!-- Axes -->
    <g class="axis">
      <line x1="0" y1="45" x2="300" y2="45"/>
      <line x1="0" y1="0" x2="0" y2="90"/>
    </g>
    
    <!-- Signal 1 -->
    <path class="signal1" d="M 0,45 Q 20,25 40,55 Q 60,75 80,35 Q 100,15 120,65 Q 140,85 160,25 Q 180,5 200,55 Q 220,75 240,35 Q 260,15 280,45 L 300,35"/>
  </g>
  
  <!-- Signal 2 Panel -->
  <g transform="translate(50, 170)">
    <text x="150" y="15" text-anchor="middle" class="text">Sensor 2 (Quarter-span)</text>
    
    <!-- Grid -->
    <g class="grid">
      <line x1="0" y1="0" x2="300" y2="0"/>
      <line x1="0" y1="30" x2="300" y2="30"/>
      <line x1="0" y1="60" x2="300" y2="60"/>
      <line x1="0" y1="90" x2="300" y2="90"/>
    </g>
    
    <!-- Axes -->
    <g class="axis">
      <line x1="0" y1="45" x2="300" y2="45"/>
      <line x1="0" y1="0" x2="0" y2="90"/>
    </g>
    
    <!-- Signal 2 (slightly delayed and attenuated) -->
    <path class="signal2" d="M 0,45 Q 25,30 45,50 Q 65,70 85,40 Q 105,20 125,60 Q 145,80 165,30 Q 185,10 205,50 Q 225,70 245,40 Q 265,20 285,50 L 300,40"/>
  </g>
  
  <!-- Cross-correlation Panel -->
  <g transform="translate(50, 290)">
    <text x="150" y="15" text-anchor="middle" class="text">Cross-Correlation Function</text>
    
    <!-- Grid -->
    <g class="grid">
      <line x1="0" y1="0" x2="300" y2="0"/>
      <line x1="0" y1="30" x2="300" y2="30"/>
      <line x1="0" y1="60" x2="300" y2="60"/>
      <line x1="0" y1="90" x2="300" y2="90"/>
      <line x1="0" y1="120" x2="300" y2="120"/>
      <line x1="150" y1="0" x2="150" y2="120"/>
    </g>
    
    <!-- Axes -->
    <g class="axis">
      <line x1="0" y1="60" x2="300" y2="60"/>
      <line x1="150" y1="0" x2="150" y2="120"/>
    </g>
    
    <!-- Cross-correlation function -->
    <path class="correlation" d="M 0,80 Q 30,95 60,85 Q 90,70 120,45 Q 150,25 180,40 Q 210,55 240,70 Q 270,85 300,90"/>
    
    <!-- Peak marker -->
    <circle cx="150" cy="25" r="5" fill="#F18F01" stroke="#333" stroke-width="2"/>
    <text x="160" y="20" class="text">Max Correlation</text>
    
    <!-- Zero lag line -->
    <line x1="150" y1="0" x2="150" y2="120" stroke="#666" stroke-width="1" stroke-dasharray="5,5"/>
    <text x="155" y="130" class="label">τ = 0</text>
  </g>
  
  <!-- Right panel with equations -->
  <g transform="translate(400, 80)">
    <rect x="0" y="0" width="350" height="300" fill="white" stroke="#ddd" stroke-width="2" rx="10"/>
    
    <text x="175" y="25" text-anchor="middle" class="title">Cross-Correlation Mathematics</text>
    
    <text x="20" y="55" class="text">Cross-correlation function:</text>
    <text x="20" y="80" class="text">R₁₂(τ) = E[x₁(t)x₂(t-τ)]</text>
    
    <text x="20" y="110" class="text">For discrete signals:</text>
    <text x="20" y="135" class="text">R₁₂[k] = Σ x₁[n]x₂[n-k]</text>
    
    <text x="20" y="165" class="text">Normalized cross-correlation:</text>
    <text x="20" y="190" class="text">ρ₁₂(τ) = R₁₂(τ) / √(R₁₁(0)R₂₂(0))</text>
    
    <text x="20" y="220" class="text">Key Properties:</text>
    <text x="20" y="240" class="text">• Peak indicates time delay</text>
    <text x="20" y="255" class="text">• Amplitude shows correlation strength</text>
    <text x="20" y="270" class="text">• Shape reveals signal similarity</text>
  </g>
  
  <!-- Time axis label -->
  <text x="200" y="490" text-anchor="middle" class="text">Time / Lag (s)</text>
</svg>
```

**Figure 4.2:** Cross-correlation analysis between two bridge sensors showing time delay and correlation strength.

The mathematical definition of cross-correlation for continuous signals is:

$$R_{xy}(\tau) = \mathbb{E}[x(t)y(t-\tau)] = \int_{-\infty}^{\infty} x(t)y(t-\tau) dt \tag{4.2}$$

For discrete signals:

$$R_{xy}[k] = \sum_{n=-\infty}^{\infty} x[n]y[n-k] \tag{4.3}$$

where $\tau$ (or $k$) represents the time lag, $x(t)$ and $y(t)$ are the two signals being compared.

### 4.3.2 Convolution: The Foundation of Linear System Analysis

Convolution describes how linear time-invariant (LTI) systems transform input signals. In SHM, structures can often be modeled as LTI systems, where the output response is the convolution of the input excitation with the system's impulse response.

The convolution integral is:

$$y(t) = x(t) * h(t) = \int_{-\infty}^{\infty} x(\tau)h(t-\tau) d\tau \tag{4.4}$$

For discrete systems:

$$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k]h[n-k] \tag{4.5}$$

where $x(t)$ is the input signal, $h(t)$ is the impulse response, $y(t)$ is the output signal, and $*$ denotes convolution.

### 4.3.3 Practical Implementation: Correlation and Convolution Analysis

```python
def demonstrate_correlation_convolution():
    """
    Demonstrate correlation and convolution with realistic SHM data
    """
    # Create two bridge sensor signals with known relationship
    fs = 100  # Hz
    duration = 30  # seconds
    t = np.linspace(0, duration, int(duration * fs))
    
    # Generate primary signal (sensor at mid-span)
    # Modal response + traffic loading + noise
    f_modal = [2.1, 4.8, 7.3]  # Bridge modal frequencies
    x1 = np.zeros_like(t)
    for f in f_modal:
        x1 += np.random.randn() * np.sin(2*np.pi*f*t + np.random.randn())
    
    # Add vehicle crossings
    vehicle_times = [5, 12, 18, 25]
    for tv in vehicle_times:
        pulse = np.exp(-((t-tv)**2)/0.5) * np.sin(2*np.pi*15*t)
        x1 += pulse
    
    x1 += 0.2 * np.random.randn(len(t))  # Measurement noise
    
    # Generate secondary signal (sensor at quarter-span)  
    # Delayed and attenuated version with different noise
    delay_samples = 8  # 0.08 second delay
    attenuation = 0.7
    x2 = np.zeros_like(t)
    x2[delay_samples:] = attenuation * x1[:-delay_samples]
    x2 += 0.25 * np.random.randn(len(t))  # Different noise
    
    # Cross-correlation analysis
    correlation = np.correlate(x1, x2, mode='full')
    correlation = correlation / np.max(correlation)  # Normalize
    
    # Lag vector
    lags = np.arange(-len(x2)+1, len(x1))
    lag_time = lags / fs
    
    # Find peak correlation and delay
    peak_idx = np.argmax(correlation)
    estimated_delay = lag_time[peak_idx]
    peak_correlation = correlation[peak_idx]
    
    # Autocorrelation for reference
    autocorr_x1 = np.correlate(x1, x1, mode='full')
    autocorr_x1 = autocorr_x1 / np.max(autocorr_x1)
    
    # Create system impulse response for convolution demo
    # Typical bridge damped oscillator response
    t_impulse = np.linspace(0, 5, 500)
    zeta = 0.02  # Damping ratio
    fn = 3.5     # Natural frequency (Hz)
    wd = 2*np.pi*fn*np.sqrt(1-zeta**2)  # Damped frequency
    h = np.exp(-2*np.pi*fn*zeta*t_impulse) * np.sin(wd*t_impulse)
    h[0] = 0  # Ensure causality
    
    # Convolution example - system response to impulse input
    impulse_input = np.zeros(len(t_impulse))
    impulse_input[50] = 10  # Unit impulse at t=0.5s
    
    # Convolution using scipy
    response = signal.convolve(impulse_input, h, mode='full')
    t_response = np.linspace(0, len(response)/100, len(response))
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Bridge Sensor Signals', 'Cross-Correlation Analysis',
            'Autocorrelation Functions', 'System Impulse Response',
            'Convolution Example', 'Correlation vs. Convolution'
        ],
        specs=[[{}, {}], [{}, {}], [{}, {}]]
    )
    
    # Plot 1: Original signals
    fig.add_trace(
        go.Scatter(x=t[:1500], y=x1[:1500], name='Sensor 1 (Mid-span)',
                  line=dict(color='#2E86AB', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t[:1500], y=x2[:1500], name='Sensor 2 (Quarter-span)',
                  line=dict(color='#F24236', width=2)),
        row=1, col=1
    )
    
    # Plot 2: Cross-correlation
    mask = (lag_time >= -2) & (lag_time <= 2)  # Focus on ±2 seconds
    fig.add_trace(
        go.Scatter(x=lag_time[mask], y=correlation[mask], name='Cross-correlation',
                  line=dict(color='#8338EC', width=3)),
        row=1, col=2
    )
    # Mark peak
    fig.add_trace(
        go.Scatter(x=[estimated_delay], y=[peak_correlation], 
                  mode='markers', name=f'Peak (τ={estimated_delay:.3f}s)',
                  marker=dict(color='#F18F01', size=10)),
        row=1, col=2
    )
    
    # Plot 3: Autocorrelation functions
    fig.add_trace(
        go.Scatter(x=lag_time[mask], y=autocorr_x1[mask], name='Autocorr Sensor 1',
                  line=dict(color='#2E86AB', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Plot 4: System impulse response
    fig.add_trace(
        go.Scatter(x=t_impulse, y=h, name='Impulse Response',
                  line=dict(color='#A23B72', width=3)),
        row=2, col=2
    )
    
    # Plot 5: Convolution example
    fig.add_trace(
        go.Scatter(x=t_impulse, y=impulse_input*100, name='Input Impulse',
                  line=dict(color='#C73E1D', width=3)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_response[:len(h)], y=response[:len(h)], name='System Response',
                  line=dict(color='#F18F01', width=3)),
        row=3, col=1
    )
    
    # Plot 6: Mathematical relationship visualization
    # Show correlation vs convolution operation
    demo_sig = np.sin(2*np.pi*0.5*np.linspace(0, 4, 200))
    demo_kernel = np.exp(-np.linspace(0, 2, 100)) * np.sin(2*np.pi*2*np.linspace(0, 2, 100))
    
    # Cross-correlation
    demo_corr = np.correlate(demo_sig, demo_kernel, mode='valid')
    # Convolution  
    demo_conv = np.convolve(demo_sig, demo_kernel, mode='valid')
    
    fig.add_trace(
        go.Scatter(y=demo_corr/np.max(demo_corr), name='Correlation Result',
                  line=dict(color='#8338EC', width=2)),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(y=demo_conv/np.max(demo_conv), name='Convolution Result',
                  line=dict(color='#2E86AB', width=2)),
        row=3, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="<b>Correlation and Convolution in Structural Health Monitoring</b>",
        title_x=0.5,
        showlegend=True,
        font=dict(size=11)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_xaxes(title_text="Lag Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=2)
    fig.update_xaxes(title_text="Lag Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Autocorrelation", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Amplitude", row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=3, col=1)
    fig.update_xaxes(title_text="Sample Index", row=3, col=2)
    fig.update_yaxes(title_text="Normalized Amplitude", row=3, col=2)
    
    fig.show()
    
    print("=== CORRELATION ANALYSIS RESULTS ===")
    print(f"Estimated time delay: {estimated_delay:.3f} seconds")
    print(f"True delay: {delay_samples/fs:.3f} seconds")
    print(f"Peak correlation coefficient: {peak_correlation:.3f}")
    print(f"Signal attenuation factor: {attenuation:.1f}")
    
    return {
        'delay_estimated': estimated_delay,
        'delay_true': delay_samples/fs,
        'correlation_peak': peak_correlation,
        'signals': (x1, x2),
        'correlation': correlation,
        'lags': lag_time
    }

# Run the demonstration
results = demonstrate_correlation_convolution()
```

This comprehensive example illustrates several key concepts:

1. **Time Delay Estimation**: Cross-correlation identifies propagation delays between sensors
2. **Signal Similarity**: Correlation magnitude indicates how similar two signals are  
3. **Autocorrelation Structure**: Reveals periodic patterns and characteristic time scales
4. **System Impulse Response**: Shows how structures respond to instantaneous excitation
5. **Convolution vs. Correlation**: Demonstrates the mathematical relationship between these operations

---

## 4.4 Impulse Response Analysis

### 4.4.1 Understanding Impulse Response in SHM

The impulse response function (IRF) completely characterizes a linear time-invariant system. For structural systems, the IRF reveals fundamental properties such as natural frequencies, damping ratios, and mode shapes. When a structure is excited by an impulsive force (such as impact testing), the resulting response directly provides the IRF.

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .axis { stroke: #333; stroke-width: 2; }
      .grid { stroke: #eee; stroke-width: 1; }
      .impulse { stroke: #F24236; stroke-width: 4; fill: none; }
      .response { stroke: #2E86AB; stroke-width: 3; fill: none; }
      .envelope { stroke: #8338EC; stroke-width: 2; stroke-dasharray: 5,5; fill: none; }
      .text { font-family: 'Arial', sans-serif; font-size: 12px; fill: #333; }
      .title { font-family: 'Arial', sans-serif; font-size: 14px; font-weight: bold; fill: #333; }
      .label { font-family: 'Arial', sans-serif; font-size: 11px; fill: #666; }
      .equation { font-family: 'Times New Roman', serif; font-size: 13px; fill: #333; }
    </style>
  </defs>
  
  <!-- Background -->
  <rect width="800" height="600" fill="#fafafa"/>
  
  <!-- Title -->
  <text x="400" y="25" text-anchor="middle" class="title">Bridge Impulse Response Analysis</text>
  
  <!-- Input Impulse Panel -->
  <g transform="translate(50, 50)">
    <text x="150" y="15" text-anchor="middle" class="text">Input Impulse Force</text>
    
    <!-- Grid -->
    <g class="grid">
      <defs>
        <pattern id="grid" width="20" height="15" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 15" stroke="#eee" stroke-width="1" fill="none"/>
        </pattern>
      </defs>
      <rect width="300" height="120" fill="url(#grid)"/>
    </g>
    
    <!-- Axes -->
    <g class="axis">
      <line x1="0" y1="100" x2="300" y2="100"/>
      <line x1="0" y1="0" x2="0" y2="120"/>
      <polygon points="295,95 305,100 295,105" fill="#333"/>
      <polygon points="5,5 0,0 -5,5" transform="translate(0,0)" fill="#333"/>
    </g>
    
    <!-- Impulse -->
    <g class="impulse">
      <line x1="60" y1="100" x2="60" y2="20"/>
      <line x1="58" y1="22" x2="62" y2="22"/>
      <text x="70" y="25" class="text">F(t) = δ(t)</text>
    </g>
    
    <!-- Labels -->
    <text x="150" y="138" text-anchor="middle" class="label">Time</text>
    <text x="-15" y="60" text-anchor="middle" class="label" transform="rotate(-90 -15 60)">Force</text>
  </g>
  
  <!-- System Representation -->
  <g transform="translate(400, 80)">
    <rect x="0" y="0" width="120" height="60" fill="white" stroke="#333" stroke-width="2" rx="10"/>
    <text x="60" y="25" text-anchor="middle" class="text">Bridge</text>
    <text x="60" y="40" text-anchor="middle" class="text">Structure</text>
    <text x="60" y="55" text-anchor="middle" class="text">H(s)</text>
    
    <!-- Arrow in -->
    <path d="M -30 30 L -5 30" stroke="#333" stroke-width="2" fill="none"/>
    <polygon points="-8,27 -2,30 -8,33" fill="#333"/>
    
    <!-- Arrow out -->
    <path d="M 125 30 L 150 30" stroke="#333" stroke-width="2" fill="none"/>
    <polygon points="147,27 153,30 147,33" fill="#333"/>
  </g>
  
  <!-- Output Response Panel -->
  <g transform="translate(50, 200)">
    <text x="150" y="15" text-anchor="middle" class="text">Impulse Response Function h(t)</text>
    
    <!-- Grid -->
    <g class="grid">
      <rect width="300" height="120" fill="url(#grid)"/>
    </g>
    
    <!-- Axes -->
    <g class="axis">
      <line x1="0" y1="60" x2="300" y2="60"/>
      <line x1="0" y1="0" x2="0" y2="120"/>
      <polygon points="295,55 305,60 295,65" fill="#333"/>
      <polygon points="5,5 0,0 -5,5" transform="translate(0,0)" fill="#333"/>
    </g>
    
    <!-- Damped oscillation response -->
    <path class="response" d="M 10,60 Q 25,20 40,80 Q 55,100 70,45 Q 85,25 100,75 Q 115,90 130,50 Q 145,35 160,68 Q 175,80 190,55 Q 205,45 220,62 Q 235,70 250,58 Q 265,52 280,60 L 300,60"/>
    
    <!-- Decay envelope -->
    <path class="envelope" d="M 10,60 Q 80,20 150,45 Q 220,55 300,60"/>
    <path class="envelope" d="M 10,60 Q 80,100 150,75 Q 220,65 300,60"/>
    
    <!-- Labels -->
    <text x="150" y="138" text-anchor="middle" class="label">Time</text>
    <text x="-15" y="60" text-anchor="middle" class="label" transform="rotate(-90 -15 60)">Response</text>
    
    <!-- Annotations -->
    <text x="200" y="35" class="label">Exponential decay</text>
    <text x="50" y="105" class="label">Oscillatory response</text>
  </g>
  
  <!-- Mathematical Analysis Panel -->
  <g transform="translate(400, 200)">
    <rect x="0" y="0" width="350" height="180" fill="white" stroke="#333" stroke-width="2" rx="10"/>
    
    <text x="175" y="20" text-anchor="middle" class="title">Impulse Response Mathematics</text>
    
    <text x="15" y="45" class="equation">For SDOF system:</text>
    <text x="15" y="65" class="equation">h(t) = (1/mω_d) e^(-ζω_n t) sin(ω_d t) u(t)</text>
    
    <text x="15" y="90" class="text">Where:</text>
    <text x="25" y="105" class="text">• ω_n = natural frequency</text>
    <text x="25" y="120" class="text">• ζ = damping ratio</text>
    <text x="25" y="135" class="text">• ω_d = ω_n√(1-ζ²) = damped frequency</text>
    <text x="25" y="150" class="text">• m = modal mass</text>
    <text x="25" y="165" class="text">• u(t) = unit step function</text>
  </g>
  
  <!-- Modal Parameters Panel -->
  <g transform="translate(50, 380)">
    <text x="150" y="15" text-anchor="middle" class="text">Extracted Modal Parameters</text>
    
    <!-- Frequency spectrum -->
    <g class="grid">
      <rect width="300" height="120" fill="url(#grid)"/>
    </g>
    
    <g class="axis">
      <line x1="0" y1="100" x2="300" y2="100"/>
      <line x1="0" y1="0" x2="0" y2="120"/>
      <polygon points="295,95 305,100 295,105" fill="#333"/>
      <polygon points="5,5 0,0 -5,5" transform="translate(0,0)" fill="#333"/>
    </g>
    
    <!-- Frequency response peaks -->
    <g stroke="#8338EC" stroke-width="3" fill="none">
      <path d="M 50,100 Q 55,80 60,20 Q 65,80 70,100"/>
      <path d="M 120,100 Q 125,85 130,40 Q 135,85 140,100"/>
      <path d="M 200,100 Q 205,90 210,60 Q 215,90 220,100"/>
    </g>
    
    <!-- Mode labels -->
    <text x="60" y="15" text-anchor="middle" class="label">f₁ = 2.1 Hz</text>
    <text x="130" y="35" text-anchor="middle" class="label">f₂ = 5.4 Hz</text>
    <text x="210" y="55" text-anchor="middle" class="label">f₃ = 9.2 Hz</text>
    
    <text x="150" y="138" text-anchor="middle" class="label">Frequency (Hz)</text>
    <text x="-15" y="60" text-anchor="middle" class="label" transform="rotate(-90 -15 60)">Magnitude</text>
  </g>
  
  <!-- Applications Panel -->
  <g transform="translate(400, 420)">
    <rect x="0" y="0" width="350" height="120" fill="white" stroke="#333" stroke-width="2" rx="10"/>
    
    <text x="175" y="20" text-anchor="middle" class="title">SHM Applications</text>
    
    <text x="15" y="40" class="text">Modal Parameter Identification:</text>
    <text x="25" y="55" class="text">• Natural frequencies from peak locations</text>
    <text x="25" y="70" class="text">• Damping ratios from decay rates</text>
    <text x="25" y="85" class="text">• Mode shapes from multiple measurements</text>
    <text x="25" y="100" class="text">• Structural changes detection</text>
  </g>
  
  <!-- Process arrows -->
  <g stroke="#666" stroke-width="2" fill="#666">
    <path d="M 200 150 L 200 175"/>
    <polygon points="195,172 200,182 205,172"/>
    
    <path d="M 200 300 L 200 355"/>
    <polygon points="195,352 200,362 205,352"/>
  </g>
</svg>
```

**Figure 4.3:** Bridge impulse response analysis showing the relationship between input impulse, system characteristics, and modal parameter extraction.

For a single degree-of-freedom (SDOF) damped oscillator, the impulse response is:

$$h(t) = \frac{1}{m\omega_d} e^{-\zeta\omega_n t} \sin(\omega_d t) u(t) \tag{4.6}$$

where:
- $\omega_n$ = natural frequency (rad/s)
- $\zeta$ = damping ratio
- $\omega_d = \omega_n\sqrt{1-\zeta^2}$ = damped natural frequency (rad/s)  
- $m$ = modal mass
- $u(t)$ = unit step function

### 4.4.2 Modal Parameter Extraction from Impulse Response

The impulse response contains complete information about the system's modal properties. Key parameters can be extracted using:

**Natural Frequency**: From the period of oscillation
$$f_n = \frac{1}{T_d} \sqrt{1-\zeta^2} \tag{4.7}$$

**Damping Ratio**: From the logarithmic decrement
$$\zeta = \frac{1}{\sqrt{1 + (2\pi/\delta)^2}} \tag{4.8}$$

where $\delta = \ln(x_i/x_{i+1})$ is the logarithmic decrement between consecutive peaks.

### 4.4.3 Practical Implementation: Impulse Response Analysis

```python
def analyze_bridge_impulse_response():
    """
    Comprehensive impulse response analysis for bridge structures
    """
    # System parameters for realistic bridge response
    fs = 200  # Sampling frequency (Hz)
    duration = 10  # Analysis duration (s)
    t = np.linspace(0, duration, int(duration * fs))
    
    # Define modal parameters for a typical bridge
    modes = [
        {'fn': 2.1, 'zeta': 0.02, 'mass': 1000, 'amplitude': 1.0},  # First bending mode
        {'fn': 5.4, 'zeta': 0.015, 'mass': 800, 'amplitude': 0.6},   # Second bending mode  
        {'fn': 9.2, 'zeta': 0.025, 'mass': 600, 'amplitude': 0.3},   # Third bending mode
        {'fn': 14.1, 'zeta': 0.03, 'mass': 400, 'amplitude': 0.15}   # Torsional mode
    ]
    
    # Generate multi-modal impulse response
    h_total = np.zeros_like(t)
    individual_responses = []
    
    for i, mode in enumerate(modes):
        fn = mode['fn']  # Hz
        zeta = mode['zeta']
        m = mode['mass']
        A = mode['amplitude']
        
        # Modal parameters
        wn = 2 * np.pi * fn  # rad/s
        wd = wn * np.sqrt(1 - zeta**2)  # Damped frequency
        
        # Individual modal response
        h_mode = (A / (m * wd)) * np.exp(-zeta * wn * t) * np.sin(wd * t)
        h_mode[t < 0] = 0  # Causality
        
        individual_responses.append(h_mode)
        h_total += h_mode
    
    # Add realistic measurement noise
    noise_level = 0.02 * np.max(h_total)
    h_measured = h_total + noise_level * np.random.randn(len(t))
    
    # Modal parameter extraction using logarithmic decrement
    def extract_modal_parameters(response, fs):
        """Extract modal parameters from impulse response"""
        results = []
        
        # Find peaks for each mode (using envelope detection)
        from scipy.signal import hilbert, find_peaks
        
        # Get analytical signal to find envelope
        analytic_signal = hilbert(response)
        envelope = np.abs(analytic_signal)
        
        # Find dominant frequencies using FFT
        fft_resp = np.fft.fft(response[:fs*4])  # First 4 seconds
        freqs = np.fft.fftfreq(len(fft_resp), 1/fs)
        fft_magnitude = np.abs(fft_resp)
        
        # Find peaks in frequency domain
        freq_peaks, _ = find_peaks(fft_magnitude[:len(freqs)//2], 
                                  height=0.1*np.max(fft_magnitude),
                                  distance=int(0.5*fs/freqs[1]))
        
        detected_frequencies = freqs[freq_peaks]
        
        for freq in detected_frequencies:
            if freq > 0.5 and freq < 20:  # Reasonable range for bridges
                # Band-pass filter around this frequency
                from scipy.signal import butter, filtfilt
                
                # Design filter
                low_freq = max(0.1, freq - 0.5)
                high_freq = min(fs/2 - 0.1, freq + 0.5)
                b, a = butter(4, [low_freq, high_freq], btype='band', fs=fs)
                
                # Filter signal
                filtered_response = filtfilt(b, a, response)
                
                # Find peaks in filtered signal
                peaks, _ = find_peaks(filtered_response, height=0.1*np.max(filtered_response))
                
                if len(peaks) > 3:  # Need at least 3 peaks for damping calculation
                    # Calculate damping using logarithmic decrement
                    peak_values = filtered_response[peaks[:5]]  # Use first 5 peaks
                    peak_times = t[peaks[:5]]
                    
                    # Fit exponential decay to peaks
                    try:
                        coeffs = np.polyfit(peak_times, np.log(np.abs(peak_values)), 1)
                        decay_rate = -coeffs[0]  # Negative slope gives decay rate
                        
                        # Calculate damping ratio
                        zeta_est = decay_rate / (2 * np.pi * freq)
                        
                        # Period from consecutive peaks
                        if len(peaks) > 1:
                            periods = np.diff(t[peaks])
                            avg_period = np.mean(periods)
                            freq_est = 1 / avg_period
                        else:
                            freq_est = freq
                            
                        results.append({
                            'frequency': freq_est,
                            'damping_ratio': zeta_est,
                            'decay_rate': decay_rate,
                            'period': avg_period if len(peaks) > 1 else 1/freq
                        })
                    except:
                        continue
        
        return results
    
    # Extract modal parameters
    extracted_params = extract_modal_parameters(h_measured, fs)
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Multi-Modal Impulse Response', 'Individual Modal Contributions',
            'Envelope and Decay Analysis', 'Frequency Domain Analysis',
            'Modal Parameter Comparison', 'Practical Implementation Guide'
        ]
    )
    
    # Plot 1: Total impulse response
    fig.add_trace(
        go.Scatter(x=t, y=h_total, name='True Response', 
                  line=dict(color='#2E86AB', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=h_measured, name='Measured (with noise)',
                  line=dict(color='#F24236', width=2, dash='dot')),
        row=1, col=1
    )
    
    # Plot 2: Individual modal contributions
    colors = ['#8338EC', '#F18F01', '#C73E1D', '#A23B72']
    for i, (mode, response) in enumerate(zip(modes, individual_responses)):
        fig.add_trace(
            go.Scatter(x=t, y=response, name=f'Mode {i+1} ({mode["fn"]:.1f} Hz)',
                      line=dict(color=colors[i], width=2)),
            row=1, col=2
        )
    
    # Plot 3: Envelope analysis
    from scipy.signal import hilbert
    analytic_signal = hilbert(h_measured)
    envelope = np.abs(analytic_signal)
    
    fig.add_trace(
        go.Scatter(x=t, y=h_measured, name='Response', 
                  line=dict(color='#2E86AB', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=envelope, name='Envelope',
                  line=dict(color='#F24236', width=3)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t, y=-envelope, name='',
                  line=dict(color='#F24236', width=3), showlegend=False),
        row=2, col=1
    )
    
    # Plot 4: Frequency domain
    fft_response = np.fft.fft(h_measured[:fs*4])
    freqs = np.fft.fftfreq(len(fft_response), 1/fs)
    fft_magnitude = np.abs(fft_response)
    
    mask = (freqs >= 0) & (freqs <= 20)
    fig.add_trace(
        go.Scatter(x=freqs[mask], y=fft_magnitude[mask], name='FFT Magnitude',
                  line=dict(color='#8338EC', width=2)),
        row=2, col=2
    )
    
    # Mark theoretical peaks
    for mode in modes:
        fig.add_trace(
            go.Scatter(x=[mode['fn']], y=[np.max(fft_magnitude[mask])*0.8], 
                      mode='markers', name=f'{mode["fn"]:.1f} Hz',
                      marker=dict(color='#F18F01', size=8, symbol='diamond')),
            row=2, col=2
        )
    
    # Plot 5: Parameter comparison
    if extracted_params:
        true_freqs = [mode['fn'] for mode in modes]
        true_damping = [mode['zeta'] for mode in modes]
        est_freqs = [p['frequency'] for p in extracted_params]
        est_damping = [p['damping_ratio'] for p in extracted_params]
        
        fig.add_trace(
            go.Scatter(x=true_freqs, y=true_damping, mode='markers',
                      name='True Parameters', marker=dict(color='#2E86AB', size=10)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=est_freqs, y=est_damping, mode='markers',
                      name='Estimated Parameters', marker=dict(color='#F24236', size=10)),
            row=3, col=1
        )
    
    # Plot 6: Implementation flowchart (text-based)
    flowchart_text = [
        "1. Record impulse response",
        "2. Apply anti-aliasing filter", 
        "3. Extract envelope using Hilbert transform",
        "4. Identify modal frequencies (FFT peaks)",
        "5. Band-pass filter each mode",
        "6. Calculate damping from decay rate",
        "7. Validate results against theory"
    ]
    
    for i, text in enumerate(flowchart_text):
        fig.add_annotation(
            x=0.1, y=0.9-i*0.12,
            xref="paper", yref="paper",
            text=text, showarrow=False,
            font=dict(size=12), align="left"
        )
    
    fig.update_layout(
        height=900,
        title_text="<b>Bridge Impulse Response Analysis and Modal Parameter Extraction</b>",
        title_x=0.5,
        showlegend=True,
        font=dict(size=11)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Response", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2) 
    fig.update_yaxes(title_text="Modal Response", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_yaxes(title_text="Magnitude", row=2, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
    fig.update_yaxes(title_text="Damping Ratio", row=3, col=1)
    
    fig.show()
    
    # Print results
    print("=== MODAL PARAMETER EXTRACTION RESULTS ===")
    print("\nTrue Modal Parameters:")
    for i, mode in enumerate(modes):
        print(f"Mode {i+1}: f = {mode['fn']:.2f} Hz, ζ = {mode['zeta']:.3f}")
    
    print("\nExtracted Modal Parameters:")
    for i, param in enumerate(extracted_params):
        print(f"Mode {i+1}: f = {param['frequency']:.2f} Hz, ζ = {param['damping_ratio']:.3f}")
    
    return {
        'true_parameters': modes,
        'extracted_parameters': extracted_params,
        'impulse_response': h_total,
        'measured_response': h_measured,
        'time': t
    }

# Run the impulse response analysis
ir_results = analyze_bridge_impulse_response()
```

This implementation demonstrates several crucial aspects of impulse response analysis:

1. **Multi-modal Response**: Real bridges exhibit multiple vibration modes
2. **Parameter Extraction**: Automated identification of frequencies and damping
3. **Noise Effects**: How measurement noise affects parameter estimation
4. **Validation**: Comparison between true and estimated parameters

---

## 4.5 Statistical Descriptors

### 4.5.1 Understanding Statistical Measures in SHM

Statistical descriptors provide compact representations of signal characteristics and are essential for monitoring structural condition over time. These measures help identify changes in structural behavior that may indicate damage or deterioration.

The fundamental statistical moments characterize different aspects of signal distribution:

**First Moment (Mean)**: Central tendency
$$\mu = \mathbb{E}[x] = \frac{1}{N}\sum_{n=1}^{N} x[n] \tag{4.9}$$

**Second Moment (Variance)**: Spread around the mean
$$\sigma^2 = \mathbb{E}[(x-\mu)^2] = \frac{1}{N-1}\sum_{n=1}^{N} (x[n]-\mu)^2 \tag{4.10}$$

**Third Moment (Skewness)**: Asymmetry of distribution
$$\text{Skew} = \mathbb{E}\left[\left(\frac{x-\mu}{\sigma}\right)^3\right] = \frac{1}{N}\sum_{n=1}^{N} \left(\frac{x[n]-\mu}{\sigma}\right)^3 \tag{4.11}$$

**Fourth Moment (Kurtosis)**: Tail heaviness
$$\text{Kurt} = \mathbb{E}\left[\left(\frac{x-\mu}{\sigma}\right)^4\right] = \frac{1}{N}\sum_{n=1}^{N} \left(\frac{x[n]-\mu}{\sigma}\right)^4 \tag{4.12}$$

### 4.5.2 Advanced Statistical Measures for SHM

**Root Mean Square (RMS)**: Energy-based measure
$$\text{RMS} = \sqrt{\frac{1}{N}\sum_{n=1}^{N} x[n]^2} \tag{4.13}$$

**Crest Factor**: Peak-to-RMS ratio
$$\text{CF} = \frac{\max|x[n]|}{\text{RMS}} \tag{4.14}$$

**Form Factor**: RMS-to-mean ratio  
$$\text{FF} = \frac{\text{RMS}}{|\mu|} \tag{4.15}$$

### 4.5.3 Practical Implementation: Statistical Analysis for SHM

```python
def comprehensive_statistical_analysis():
    """
    Comprehensive statistical analysis of bridge monitoring data
    """
    # Generate realistic long-term monitoring data
    def generate_long_term_data(days=30):
        """Generate 30 days of bridge monitoring data with various conditions"""
        fs = 50  # Lower sampling rate for long-term data
        hours_per_day = 24
        samples_per_hour = fs * 3600
        
        data = {}
        time_total = []
        
        for day in range(days):
            daily_data = []
            daily_time = []
            
            for hour in range(hours_per_day):
                # Base time vector for this hour
                t_hour = np.linspace(0, 3600, samples_per_hour)
                t_absolute = t_hour + day * 24 * 3600 + hour * 3600
                
                # Environmental conditions (temperature, wind, traffic)
                temp_factor = 1 + 0.3 * np.sin(2*np.pi*(hour-6)/24)  # Daily cycle
                wind_factor = 1 + 0.2 * np.sin(2*np.pi*hour/24 + np.pi/4)  # Wind pattern
                
                # Traffic loading (higher during day hours)
                if 6 <= hour <= 22:  # Daytime
                    traffic_intensity = 1.0 + 0.5 * np.sin(2*np.pi*(hour-6)/16)
                else:  # Nighttime
                    traffic_intensity = 0.2
                
                # Modal response with environmental effects
                base_freq = 2.5 * (1 - 0.01 * temp_factor)  # Temperature effect on frequency
                
                # Generate hourly acceleration data
                modal_response = (0.8 * temp_factor * 
                                np.sin(2*np.pi*base_freq*t_hour) * 
                                np.exp(-0.001*t_hour))
                
                # Traffic pulses
                n_vehicles = int(traffic_intensity * np.random.poisson(10))  # Vehicles per hour
                traffic_response = np.zeros_like(t_hour)
                
                for _ in range(n_vehicles):
                    vehicle_time = np.random.uniform(0, 3600)
                    vehicle_mask = (t_hour >= vehicle_time) & (t_hour <= vehicle_time + 5)
                    if np.any(vehicle_mask):
                        vehicle_pulse = (np.random.uniform(0.5, 2.0) * 
                                       np.exp(-((t_hour[vehicle_mask] - vehicle_time - 2.5)**2)/2))
                        traffic_response[vehicle_mask] += vehicle_pulse
                
                # Wind loading
                wind_response = (0.3 * wind_factor * 
                               (np.sin(2*np.pi*0.1*t_hour) + 0.5*np.sin(2*np.pi*0.3*t_hour)))
                
                # Measurement noise
                noise = 0.1 * np.random.randn(len(t_hour))
                
                # Possible structural change (damage simulation)
                if day > 20:  # Damage occurs after day 20
                    damage_factor = 1 + 0.05 * (day - 20)  # Gradual increase
                    base_freq *= (1 - 0.002 * damage_factor)  # Frequency reduction
                    modal_response *= damage_factor  # Amplitude change
                
                # Combine all components
                hourly_signal = (modal_response + traffic_response + 
                               wind_response + noise)
                
                daily_data.extend(hourly_signal)
                daily_time.extend(t_absolute)
                time_total.extend(t_absolute)
            
            data[f'day_{day+1}'] = np.array(daily_data)
        
        return np.array(time_total), np.concatenate([data[f'day_{day+1}'] for day in range(days)])
    
    # Generate the dataset
    time, acceleration = generate_long_term_data(days=30)
    
    # Calculate statistical descriptors over time windows
    window_hours = 6  # 6-hour windows
    fs = 50
    samples_per_window = window_hours * 3600 * fs
    n_windows = len(acceleration) // samples_per_window
    
    # Initialize statistics arrays
    stats = {
        'mean': [], 'std': [], 'rms': [], 'skew': [], 'kurt': [],
        'crest': [], 'form': [], 'max': [], 'min': [], 'energy': []
    }
    
    window_times = []
    
    for i in range(n_windows):
        start_idx = i * samples_per_window
        end_idx = start_idx + samples_per_window
        window_data = acceleration[start_idx:end_idx]
        window_time = time[start_idx + samples_per_window//2]  # Middle of window
        
        # Calculate all statistics
        mean_val = np.mean(window_data)
        std_val = np.std(window_data, ddof=1)
        rms_val = np.sqrt(np.mean(window_data**2))
        skew_val = pd.Series(window_data).skew()
        kurt_val = pd.Series(window_data).kurtosis()
        max_val = np.max(np.abs(window_data))
        min_val = np.min(window_data)
        crest_val = max_val / rms_val if rms_val > 0 else 0
        form_val = rms_val / np.abs(mean_val) if abs(mean_val) > 1e-10 else np.inf
        energy_val = np.sum(window_data**2)
        
        # Store results
        stats['mean'].append(mean_val)
        stats['std'].append(std_val)
        stats['rms'].append(rms_val)
        stats['skew'].append(skew_val)
        stats['kurt'].append(kurt_val)
        stats['crest'].append(crest_val)
        stats['form'].append(form_val)
        stats['max'].append(max_val)
        stats['min'].append(min_val)
        stats['energy'].append(energy_val)
        
        window_times.append(window_time)
    
    # Convert to numpy arrays
    for key in stats:
        stats[key] = np.array(stats[key])
    
    window_times = np.array(window_times)
    
    # Convert time to days for plotting
    time_days = time / (24 * 3600)
    window_days = window_times / (24 * 3600)
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Long-term Acceleration Data (30 days)', 'Statistical Moments Evolution',
            'RMS and Energy Trends', 'Shape Descriptors (Skewness & Kurtosis)',
            'Peak Indicators (Crest Factor)', 'Distribution Evolution',
            'Correlation Matrix of Statistics', 'Change Detection Analysis'
        ]
    )
    
    # Plot 1: Raw time series (decimated for visualization)
    decimate_factor = 1000
    fig.add_trace(
        go.Scatter(x=time_days[::decimate_factor], y=acceleration[::decimate_factor],
                  mode='lines', name='Acceleration', 
                  line=dict(color='#2E86AB', width=1)),
        row=1, col=1
    )
    
    # Plot 2: Statistical moments
    fig.add_trace(
        go.Scatter(x=window_days, y=stats['mean'], name='Mean',
                  line=dict(color='#F24236', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=window_days, y=stats['std'], name='Std Dev',
                  line=dict(color='#8338EC', width=2)),
        row=1, col=2
    )
    
    # Plot 3: RMS and Energy
    fig.add_trace(
        go.Scatter(x=window_days, y=stats['rms'], name='RMS',
                  line=dict(color='#F18F01', width=3)),
        row=2, col=1
    )
    
    # Normalize energy for plotting on same scale
    normalized_energy = stats['energy'] / np.max(stats['energy']) * np.max(stats['rms'])
    fig.add_trace(
        go.Scatter(x=window_days, y=normalized_energy, name='Energy (normalized)',
                  line=dict(color='#C73E1D', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Plot 4: Shape descriptors
    fig.add_trace(
        go.Scatter(x=window_days, y=stats['skew'], name='Skewness',
                  line=dict(color='#A23B72', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=window_days, y=stats['kurt'], name='Kurtosis',
                  line=dict(color='#2E8B57', width=2)),
        row=2, col=2
    )
    
    # Plot 5: Crest factor
    fig.add_trace(
        go.Scatter(x=window_days, y=stats['crest'], name='Crest Factor',
                  line=dict(color='#FF6347', width=3)),
        row=3, col=1
    )
    
    # Plot 6: Distribution evolution (histogram at different time points)
    early_data = acceleration[:samples_per_window]  # First 6 hours
    late_data = acceleration[-samples_per_window:]  # Last 6 hours
    
    fig.add_trace(
        go.Histogram(x=early_data, nbinsx=50, name='Early Distribution',
                    marker_color='#2E86AB', opacity=0.7),
        row=3, col=2
    )
    fig.add_trace(
        go.Histogram(x=late_data, nbinsx=50, name='Late Distribution',
                    marker_color='#F24236', opacity=0.7),
        row=3, col=2
    )
    
    # Plot 7: Correlation matrix
    stat_keys = ['mean', 'std', 'rms', 'skew', 'kurt', 'crest']
    stat_matrix = np.column_stack([stats[key] for key in stat_keys])
    corr_matrix = np.corrcoef(stat_matrix.T)
    
    fig.add_trace(
        go.Heatmap(z=corr_matrix, x=stat_keys, y=stat_keys,
                  colorscale='RdBu', zmid=0,
                  text=np.round(corr_matrix, 2), texttemplate='%{text}',
                  name='Correlation'),
        row=4, col=1
    )
    
    # Plot 8: Change detection using control chart
    # Use RMS as damage indicator
    baseline_rms = np.mean(stats['rms'][:10])  # First 10 windows as baseline
    baseline_std = np.std(stats['rms'][:10])
    
    control_limits = {
        'upper': baseline_rms + 3 * baseline_std,
        'lower': baseline_rms - 3 * baseline_std
    }
    
    fig.add_trace(
        go.Scatter(x=window_days, y=stats['rms'], name='RMS',
                  mode='lines+markers', line=dict(color='#2E86AB', width=2)),
        row=4, col=2
    )
    
    # Control limits
    fig.add_hline(y=control_limits['upper'], line=dict(color='red', dash='dash'),
                 annotation_text="Upper Control Limit", row=4, col=2)
    fig.add_hline(y=control_limits['lower'], line=dict(color='red', dash='dash'),
                 annotation_text="Lower Control Limit", row=4, col=2)
    fig.add_hline(y=baseline_rms, line=dict(color='green', dash='dot'),
                 annotation_text="Baseline", row=4, col=2)
    
    # Mark damage introduction
    fig.add_vline(x=20, line=dict(color='orange', dash='dash', width=3),
                 annotation_text="Damage Introduced", row=4, col=2)
    
    fig.update_layout(
        height=1200,
        title_text="<b>Statistical Analysis of Long-term Bridge Monitoring Data</b>",
        title_x=0.5,
        showlegend=True,
        font=dict(size=10)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (days)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_xaxes(title_text="Time (days)", row=1, col=2)
    fig.update_yaxes(title_text="Statistical Moment", row=1, col=2)
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)
    fig.update_yaxes(title_text="RMS Value", row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=2)
    fig.update_yaxes(title_text="Shape Parameter", row=2, col=2)
    fig.update_xaxes(title_text="Time (days)", row=3, col=1)
    fig.update_yaxes(title_text="Crest Factor", row=3, col=1)
    fig.update_xaxes(title_text="Acceleration (m/s²)", row=3, col=2)
    fig.update_yaxes(title_text="Frequency", row=3, col=2)
    fig.update_xaxes(title_text="Statistics", row=4, col=1)
    fig.update_yaxes(title_text="Statistics", row=4, col=1)
    fig.update_xaxes(title_text="Time (days)", row=4, col=2)
    fig.update_yaxes(title_text="RMS Value", row=4, col=2)
    
    fig.show()
    
    # Statistical summary
    print("=== STATISTICAL ANALYSIS SUMMARY ===")
    print(f"Monitoring Duration: {time_days[-1]:.1f} days")
    print(f"Number of Analysis Windows: {n_windows}")
    print(f"Window Duration: {window_hours} hours")
    
    print("\nBaseline Statistics (First 10 windows):")
    baseline_stats = {key: np.mean(stats[key][:10]) for key in stats.keys()}
    for key, value in baseline_stats.items():
        print(f"{key.capitalize()}: {value:.4f}")
    
    print("\nFinal Statistics (Last 10 windows):")
    final_stats = {key: np.mean(stats[key][-10:]) for key in stats.keys()}
    for key, value in final_stats.items():
        print(f"{key.capitalize()}: {value:.4f}")
    
    print("\nRelative Changes (%):")
    for key in stats.keys():
        if baseline_stats[key] != 0:
            change = (final_stats[key] - baseline_stats[key]) / baseline_stats[key] * 100
            print(f"{key.capitalize()}: {change:+.2f}%")
    
    # Damage detection results
    exceedances = np.sum(stats['rms'] > control_limits['upper'])
    print(f"\nControl Chart Analysis:")
    print(f"Upper limit exceedances: {exceedances}/{n_windows} windows")
    print(f"First exceedance at day: {window_days[np.where(stats['rms'] > control_limits['upper'])[0][0]]:.1f}" 
          if exceedances > 0 else "No exceedances detected")
    
    return {
        'time': time,
        'acceleration': acceleration,
        'statistics': stats,
        'window_times': window_times,
        'baseline_stats': baseline_stats,
        'final_stats': final_stats
    }

# Run comprehensive statistical analysis
stat_results = comprehensive_statistical_analysis()
```

This implementation demonstrates several key concepts:

1. **Long-term Monitoring**: Statistical trends over extended periods
2. **Environmental Effects**: How temperature and loading affect statistical measures
3. **Damage Detection**: Using statistical process control methods
4. **Multi-parameter Analysis**: Comprehensive set of statistical descriptors
5. **Correlation Analysis**: Understanding relationships between different statistics

---

## 4.6 Data Preprocessing and Cleaning

### 4.6.1 The Importance of Data Quality in SHM

Raw sensor data in structural health monitoring systems often contains various artifacts and distortions that can significantly impact analysis results. Effective preprocessing ensures that subsequent analysis is based on clean, reliable data that accurately represents structural behavior.

Common data quality issues include:

- **Sensor drift**: Gradual changes in sensor calibration over time
- **Baseline shifts**: Sudden offset changes due to temperature or electrical effects  
- **Trends**: Long-term monotonic changes unrelated to structural behavior
- **Spikes and outliers**: Isolated extreme values from electromagnetic interference
- **Missing data**: Gaps due to sensor failures or communication errors
- **Aliasing**: High-frequency content folded into the measurement band

```svg
<svg viewBox="0 0 800 700" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .axis { stroke: #333; stroke-width: 2; }
      .grid { stroke: #eee; stroke-width: 1; }
      .raw-signal { stroke: #F24236; stroke-width: 2; fill: none; }
      .clean-signal { stroke: #2E86AB; stroke-width: 3; fill: none; }
      .trend { stroke: #8338EC; stroke-width: 3; stroke-dasharray: 8,4; fill: none; }
      .outliers { fill: #F18F01; stroke: #333; stroke-width: 1; }
      .text { font-family: 'Arial', sans-serif; font-size: 12px; fill: #333; }
      .title { font-family: 'Arial', sans-serif; font-size: 14px; font-weight: bold; fill: #333; }
      .label { font-family: 'Arial', sans-serif; font-size: 11px; fill: #666; }
      .process-box { fill: white; stroke: #333; stroke-width: 2; rx: 8; }
      .arrow { stroke: #666; stroke-width: 2; fill: #666; }
    </style>
  </defs>
  
  <!-- Background -->
  <rect width="800" height="700" fill="#fafafa"/>
  
  <!-- Title -->
  <text x="400" y="25" text-anchor="middle" class="title">Data Preprocessing Pipeline for Bridge SHM</text>
  
  <!-- Raw Data Panel -->
  <g transform="translate(50, 50)">
    <text x="150" y="15" text-anchor="middle" class="text">Raw Sensor Data (Contaminated)</text>
    
    <!-- Grid -->
    <g class="grid">
      <defs>
        <pattern id="smallgrid" width="20" height="10" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 10" stroke="#eee" stroke-width="1" fill="none"/>
        </pattern>
      </defs>
      <rect width="300" height="100" fill="url(#smallgrid)"/>
    </g>
    
    <!-- Axes -->
    <g class="axis">
      <line x1="0" y1="80" x2="300" y2="80"/>
      <line x1="0" y1="0" x2="0" y2="100"/>
    </g>
    
    <!-- Raw signal with problems -->
    <!-- Base signal -->
    <path class="raw-signal" d="M 0,50 Q 20,30 40,70 Q 60,90 80,40 Q 100,20 120,60 Q 140,80 160,35 Q 180,15 200,65 Q 220,85 240,45 Q 260,25 280,55 L 300,45"/>
    
    <!-- Trend line -->
    <path class="trend" d="M 0,80 L 300,20"/>
    
    <!-- Outliers -->
    <g class="outliers">
      <circle cx="80" cy="10" r="4"/>
      <circle cx="160" cy="90" r="4"/>
      <circle cx="220" cy="5" r="4"/>
    </g>
    
    <!-- Labels -->
    <text x="150" y="118" text-anchor="middle" class="label">Time</text>
    <text x="-15" y="50" text-anchor="middle" class="label" transform="rotate(-90 -15 50)">Signal</text>
    <text x="310" y="25" class="label">Trend</text>
    <text x="85" y="5" class="label">Outlier</text>
  </g>
  
  <!-- Processing Steps -->
  <g transform="translate(400, 80)">
    <!-- Step boxes -->
    <g class="process-box">
      <rect x="0" y="0" width="120" height="30"/>
      <text x="60" y="20" text-anchor="middle" class="text">1. Outlier Detection</text>
    </g>
    
    <g class="process-box" transform="translate(0, 40)">
      <rect x="0" y="0" width="120" height="30"/>
      <text x="60" y="20" text-anchor="middle" class="text">2. Detrending</text>
    </g>
    
    <g class="process-box" transform="translate(0, 80)">
      <rect x="0" y="0" width="120" height="30"/>
      <text x="60" y="20" text-anchor="middle" class="text">3. Baseline Correction</text>
    </g>
    
    <g class="process-box" transform="translate(0, 120)">
      <rect x="0" y="0" width="120" height="30"/>
      <text x="60" y="20" text-anchor="middle" class="text">4. Noise Reduction</text>
    </g>
    
    <!-- Arrows -->
    <g class="arrow">
      <path d="M 60 35 L 60 40"/>
      <polygon points="57,37 60,42 63,37"/>
      <path d="M 60 75 L 60 80"/>
      <polygon points="57,77 60,82 63,77"/>
      <path d="M 60 115 L 60 120"/>
      <polygon points="57,117 60,122 63,117"/>
    </g>
  </g>
  
  <!-- Clean Data Panel -->
  <g transform="translate(50, 200)">
    <text x="150" y="15" text-anchor="middle" class="text">Processed Signal (Clean)</text>
    
    <!-- Grid -->
    <g class="grid">
      <rect width="300" height="100" fill="url(#smallgrid)"/>
    </g>
    
    <!-- Axes -->
    <g class="axis">
      <line x1="0" y1="50" x2="300" y2="50"/>
      <line x1="0" y1="0" x2="0" y2="100"/>
    </g>
    
    <!-- Clean signal -->
    <path class="clean-signal" d="M 0,50 Q 20,35 40,65 Q 60,75 80,45 Q 100,30 120,60 Q 140,70 160,40 Q 180,25 200,65 Q 220,75 240,45 Q 260,35 280,55 L 300,50"/>
    
    <!-- Labels -->
    <text x="150" y="118" text-anchor="middle" class="label">Time</text>
    <text x="-15" y="50" text-anchor="middle" class="label" transform="rotate(-90 -15 50)">Signal</text>
  </g>
  
  <!-- Detrending Method Panel -->
  <g transform="translate(50, 350)">
    <rect x="0" y="0" width="700" height="140" class="process-box"/>
    <text x="350" y="20" text-anchor="middle" class="title">Detrending Methods</text>
    
    <!-- Linear detrending -->
    <g transform="translate(20, 30)">
      <text x="0" y="15" class="text">Linear Detrending:</text>
      <text x="0" y="35" class="text">x_detrend[n] = x[n] - (a·n + b)</text>
      <text x="0" y="55" class="text">where a, b are least-squares fit coefficients</text>
    </g>
    
    <!-- Polynomial detrending -->
    <g transform="translate(250, 30)">
      <text x="0" y="15" class="text">Polynomial Detrending:</text>
      <text x="0" y="35" class="text">x_detrend[n] = x[n] - Σ(c_k · n^k)</text>
      <text x="0" y="55" class="text">Higher-order polynomial fits</text>
    </g>
    
    <!-- High-pass filtering -->
    <g transform="translate(480, 30)">
      <text x="0" y="15" class="text">High-pass Filtering:</text>
      <text x="0" y="35" class="text">Removes low-frequency trends</text>
      <text x="0" y="55" class="text">Preserves dynamic content</text>
    </g>
    
    <!-- Moving average subtraction -->
    <g transform="translate(20, 80)">
      <text x="0" y="15" class="text">Moving Average Subtraction:</text>
      <text x="0" y="35" class="text">x_detrend[n] = x[n] - MA_window[n]</text>
    </g>
    
    <!-- Median filtering -->
    <g transform="translate(350, 80)">
      <text x="0" y="15" class="text">Median-based Methods:</text>
      <text x="0" y="35" class="text">Robust against outliers</text>
    </g>
  </g>
  
  <!-- Outlier Detection Panel -->
  <g transform="translate(50, 520)">
    <rect x="0" y="0" width="700" height="120" class="process-box"/>
    <text x="350" y="20" text-anchor="middle" class="title">Outlier Detection Strategies</text>
    
    <g transform="translate(20, 35)">
      <text x="0" y="15" class="text">Statistical Methods:</text>
      <text x="0" y="30" class="text">• Z-score: |x - μ|/σ > threshold</text>
      <text x="0" y="45" class="text">• Modified Z-score: |0.6745(x - median)|/MAD > 3.5</text>
      <text x="0" y="60" class="text">• Interquartile Range (IQR): x < Q1-1.5·IQR or x > Q3+1.5·IQR</text>
    </g>
    
    <g transform="translate(380, 35)">
      <text x="0" y="15" class="text">Advanced Methods:</text>
      <text x="0" y="30" class="text">• Isolation Forest</text>
      <text x="0" y="45" class="text">• Local Outlier Factor</text>
      <text x="0" y="60" class="text">• DBSCAN clustering</text>
    </g>
  </g>
</svg>
```

**Figure 4.4:** Data preprocessing pipeline showing the transformation from contaminated raw sensor data to clean processed signals suitable for structural analysis.

### 4.6.2 Detrending Techniques

Detrending removes long-term systematic variations that can mask structural dynamic behavior. The choice of detrending method depends on the nature of the trend and the analysis requirements.

**Linear Detrending**: Removes linear trends using least-squares fitting
$\hat{x}[n] = x[n] - (an + b) \tag{4.16}$

where coefficients $a$ and $b$ minimize $\sum_{n=1}^{N}(x[n] - an - b)^2$.

**Polynomial Detrending**: For nonlinear trends
$\hat{x}[n] = x[n] - \sum_{k=0}^{p} c_k n^k \tag{4.17}$

**High-pass Filtering**: Preserves dynamic content while removing low-frequency trends
$H(s) = \frac{s}{s + \omega_c} \tag{4.18}$

where $\omega_c$ is the cutoff frequency chosen to preserve structural dynamics.

### 4.6.3 Practical Implementation: Comprehensive Data Preprocessing

```python
def comprehensive_data_preprocessing():
    """
    Demonstrate comprehensive data preprocessing for SHM applications
    """
    # Generate realistic contaminated bridge data
    def generate_contaminated_data(duration=120, fs=100):
        """Generate bridge data with realistic contamination"""
        t = np.linspace(0, duration, int(duration * fs))
        
        # True structural response (clean signal)
        clean_signal = (1.5 * np.sin(2*np.pi*2.3*t) * np.exp(-0.01*t) +
                       0.8 * np.sin(2*np.pi*5.7*t) * np.exp(-0.005*t) +
                       0.4 * np.sin(2*np.pi*8.9*t) * np.exp(-0.002*t))
        
        # Add contamination
        
        # 1. Linear trend (sensor drift)
        trend = 0.05 * t / duration
        
        # 2. Polynomial trend (temperature effects)
        poly_trend = 0.02 * ((t/duration)**2 - 0.5*(t/duration)**3)
        
        # 3. Baseline shift (sudden offset changes)
        baseline_shift = np.zeros_like(t)
        shift_times = [30, 75]  # Times of baseline shifts
        shift_magnitudes = [0.3, -0.2]
        for shift_time, magnitude in zip(shift_times, shift_magnitudes):
            baseline_shift[t >= shift_time] += magnitude
        
        # 4. Outliers (electromagnetic spikes)
        outliers = np.zeros_like(t)
        n_outliers = 15
        outlier_indices = np.random.choice(len(t), n_outliers, replace=False)
        outlier_magnitudes = np.random.uniform(-3, 3, n_outliers)
        outliers[outlier_indices] = outlier_magnitudes
        
        # 5. Missing data (simulate sensor failures)
        missing_segments = [(2000, 2200), (5500, 5800), (9000, 9100)]  # (start, end) indices
        
        # 6. Measurement noise
        noise = 0.1 * np.random.randn(len(t))
        
        # 7. 60 Hz electrical interference
        interference = 0.05 * np.sin(2*np.pi*60*t)
        
        # Combine all contamination
        contaminated = (clean_signal + trend + poly_trend + baseline_shift + 
                       outliers + noise + interference)
        
        # Introduce missing data
        missing_mask = np.ones_like(t, dtype=bool)
        for start, end in missing_segments:
            contaminated[start:end] = np.nan
            missing_mask[start:end] = False
        
        return t, clean_signal, contaminated, missing_mask, {
            'trend': trend + poly_trend,
            'baseline_shift': baseline_shift,
            'outliers': outliers,
            'noise': noise,
            'interference': interference
        }
    
    # Generate contaminated dataset
    time, clean_signal, contaminated, missing_mask, components = generate_contaminated_data()
    
    # Initialize preprocessing pipeline
    processed_signal = contaminated.copy()
    preprocessing_steps = {}
    
    # Step 1: Handle missing data using interpolation
    def interpolate_missing_data(signal):
        """Interpolate missing data segments"""
        result = signal.copy()
        nan_mask = np.isnan(signal)
        
        if np.any(nan_mask):
            # Use linear interpolation for gaps
            valid_indices = np.where(~nan_mask)[0]
            valid_values = signal[~nan_mask]
            
            # Interpolate missing values
            from scipy.interpolate import interp1d
            if len(valid_indices) > 1:
                f = interp1d(valid_indices, valid_values, kind='linear', 
                           bounds_error=False, fill_value='extrapolate')
                result[nan_mask] = f(np.where(nan_mask)[0])
        
        return result, nan_mask
    
    processed_signal, nan_locations = interpolate_missing_data(processed_signal)
    preprocessing_steps['missing_data_interpolated'] = processed_signal.copy()
    
    # Step 2: Outlier detection and removal
    def detect_and_remove_outliers(signal, method='modified_zscore', threshold=3.5):
        """Detect and remove outliers using various methods"""
        result = signal.copy()
        outlier_mask = np.zeros_like(signal, dtype=bool)
        
        if method == 'zscore':
            # Standard Z-score method
            z_scores = np.abs((signal - np.mean(signal)) / np.std(signal))
            outlier_mask = z_scores > threshold
            
        elif method == 'modified_zscore':
            # Modified Z-score using median and MAD
            median = np.median(signal)
            mad = np.median(np.abs(signal - median))
            modified_z = 0.6745 * (signal - median) / mad
            outlier_mask = np.abs(modified_z) > threshold
            
        elif method == 'iqr':
            # Interquartile range method
            Q1 = np.percentile(signal, 25)
            Q3 = np.percentile(signal, 75)
            IQR = Q3 - Q1
            outlier_mask = (signal < Q1 - 1.5*IQR) | (signal > Q3 + 1.5*IQR)
        
        # Replace outliers with interpolated values
        if np.any(outlier_mask):
            # Use median filter to replace outliers
            from scipy.signal import medfilt
            filtered_signal = medfilt(signal, kernel_size=5)
            result[outlier_mask] = filtered_signal[outlier_mask]
        
        return result, outlier_mask
    
    processed_signal, outlier_locations = detect_and_remove_outliers(processed_signal, 
                                                                   method='modified_zscore')
    preprocessing_steps['outliers_removed'] = processed_signal.copy()
    
    # Step 3: Detrending
    def detrend_signal(signal, method='linear'):
        """Remove trends from signal"""
        result = signal.copy()
        trend_removed = np.zeros_like(signal)
        
        if method == 'linear':
            # Linear detrending
            coeffs = np.polyfit(np.arange(len(signal)), signal, 1)
            trend_removed = np.polyval(coeffs, np.arange(len(signal)))
            result = signal - trend_removed
            
        elif method == 'polynomial':
            # Polynomial detrending (order 3)
            coeffs = np.polyfit(np.arange(len(signal)), signal, 3)
            trend_removed = np.polyval(coeffs, np.arange(len(signal)))
            result = signal - trend_removed
            
        elif method == 'highpass':
            # High-pass filtering
            from scipy.signal import butter, filtfilt
            nyquist = fs / 2
            cutoff = 0.1  # Hz - preserves structural dynamics > 0.1 Hz
            b, a = butter(4, cutoff/nyquist, btype='high')
            result = filtfilt(b, a, signal)
            trend_removed = signal - result
        
        return result, trend_removed
    
    processed_signal, removed_trend = detrend_signal(processed_signal, method='polynomial')
    preprocessing_steps['detrended'] = processed_signal.copy()
    
    # Step 4: Baseline correction (handle sudden shifts)
    def correct_baseline_shifts(signal, window_size=1000):
        """Detect and correct sudden baseline shifts"""
        result = signal.copy()
        
        # Use moving median to detect shifts
        from scipy.signal import medfilt
        moving_median = pd.Series(signal).rolling(window_size, center=True).median()
        
        # Find discontinuities in median
        median_diff = np.diff(moving_median.fillna(method='bfill').fillna(method='ffill'))
        shift_threshold = 3 * np.std(median_diff[~np.isnan(median_diff)])
        
        shift_locations = np.where(np.abs(median_diff) > shift_threshold)[0]
        
        # Correct shifts
        for shift_idx in shift_locations:
            shift_magnitude = median_diff[shift_idx]
            result[shift_idx+1:] -= shift_magnitude
        
        return result, shift_locations
    
    processed_signal, shift_locations = correct_baseline_shifts(processed_signal)
    preprocessing_steps['baseline_corrected'] = processed_signal.copy()
    
    # Step 5: Noise reduction (optional filtering)
    def reduce_noise(signal, method='savgol'):
        """Apply noise reduction filtering"""
        result = signal.copy()
        
        if method == 'savgol':
            # Savitzky-Golay filter
            from scipy.signal import savgol_filter
            window_length = min(51, len(signal)//10)  # Adaptive window
            if window_length % 2 == 0:
                window_length += 1  # Must be odd
            result = savgol_filter(signal, window_length, 3)
            
        elif method == 'lowpass':
            # Low-pass filter
            from scipy.signal import butter, filtfilt
            nyquist = fs / 2
            cutoff = 25  # Hz - preserve structural content up to 25 Hz
            b, a = butter(6, cutoff/nyquist, btype='low')
            result = filtfilt(b, a, signal)
            
        elif method == 'wavelet':
            # Wavelet denoising
            import pywt
            coeffs = pywt.wavedec(signal, 'db6', level=6)
            # Soft thresholding
            threshold = 0.1 * np.max([np.max(np.abs(c)) for c in coeffs[1:]])
            coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, 'soft') 
                                         for c in coeffs[1:]]
            result = pywt.waverec(coeffs_thresh, 'db6')
        
        return result
    
    final_signal = reduce_noise(processed_signal, method='savgol')
    preprocessing_steps['noise_reduced'] = final_signal.copy()
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Original Contaminated Signal', 'Clean Reference Signal',
            'Preprocessing Steps Comparison', 'Outlier Detection Results',
            'Detrending Analysis', 'Baseline Correction',
            'Final Comparison', 'Preprocessing Quality Metrics'
        ]
    )
    
    # Plot 1: Contaminated signal
    fig.add_trace(
        go.Scatter(x=time[:3000], y=contaminated[:3000], name='Contaminated',
                  line=dict(color='#F24236', width=2)),
        row=1, col=1
    )
    
    # Plot 2: Clean reference
    fig.add_trace(
        go.Scatter(x=time[:3000], y=clean_signal[:3000], name='True Clean Signal',
                  line=dict(color='#2E86AB', width=2)),
        row=1, col=2
    )
    
    # Plot 3: Preprocessing steps
    steps = ['missing_data_interpolated', 'outliers_removed', 'detrended', 
             'baseline_corrected', 'noise_reduced']
    colors = ['#8338EC', '#F18F01', '#C73E1D', '#A23B72', '#2E8B57']
    
    for i, (step, color) in enumerate(zip(steps, colors)):
        fig.add_trace(
            go.Scatter(x=time[1000:2000], y=preprocessing_steps[step][1000:2000], 
                      name=step.replace('_', ' ').title(),
                      line=dict(color=color, width=2)),
            row=2, col=1
        )
    
    # Plot 4: Outlier detection
    fig.add_trace(
        go.Scatter(x=time, y=contaminated, mode='lines', name='Original',
                  line=dict(color='#F24236', width=2)),
        row=2, col=2
    )
    
    # Mark outliers
    outlier_times = time[outlier_locations]
    outlier_values = contaminated[outlier_locations]
    fig.add_trace(
        go.Scatter(x=outlier_times, y=outlier_values, mode='markers',
                  name='Detected Outliers', marker=dict(color='#F18F01', size=8)),
        row=2, col=2
    )
    
    # Plot 5: Detrending
    fig.add_trace(
        go.Scatter(x=time, y=contaminated, name='Before Detrending',
                  line=dict(color='#F24236', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=removed_trend, name='Removed Trend',
                  line=dict(color='#8338EC', width=3, dash='dash')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=preprocessing_steps['detrended'], name='After Detrending',
                  line=dict(color='#2E86AB', width=2)),
        row=3, col=1
    )
    
    # Plot 6: Baseline correction
    fig.add_trace(
        go.Scatter(x=time, y=preprocessing_steps['outliers_removed'], 
                  name='Before Baseline Correction',
                  line=dict(color='#F24236', width=2)),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=time, y=preprocessing_steps['baseline_corrected'],
                  name='After Baseline Correction', 
                  line=dict(color='#2E86AB', width=2)),
        row=3, col=2
    )
    
    # Mark shift locations
    for shift_idx in shift_locations:
        if shift_idx < len(time):
            fig.add_vline(x=time[shift_idx], line=dict(color='orange', dash='dash'),
                         annotation_text="Detected Shift", row=3, col=2)
    
    # Plot 7: Final comparison
    fig.add_trace(
        go.Scatter(x=time[:5000], y=clean_signal[:5000], name='True Signal',
                  line=dict(color='#2E86AB', width=3)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=time[:5000], y=final_signal[:5000], name='Processed Signal',
                  line=dict(color='#F18F01', width=2, dash='dot')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=time[:5000], y=contaminated[:5000], name='Original Contaminated',
                  line=dict(color='#F24236', width=1, dash='dash'), opacity=0.5),
        row=4, col=1
    )
    
    # Plot 8: Quality metrics
    # Calculate quality metrics
    mse_original = np.mean((clean_signal - contaminated)**2)
    mse_processed = np.mean((clean_signal - final_signal)**2)
    
    correlation_original = np.corrcoef(clean_signal, contaminated)[0, 1]
    correlation_processed = np.corrcoef(clean_signal, final_signal)[0, 1]
    
    snr_original = 10 * np.log10(np.var(clean_signal) / np.var(contaminated - clean_signal))
    snr_processed = 10 * np.log10(np.var(clean_signal) / np.var(final_signal - clean_signal))
    
    metrics = {
        'MSE': [mse_original, mse_processed],
        'Correlation': [correlation_original, correlation_processed],
        'SNR (dB)': [snr_original, snr_processed]
    }
    
    metric_names = list(metrics.keys())
    original_values = [metrics[name][0] for name in metric_names]
    processed_values = [metrics[name][1] for name in metric_names]
    
    fig.add_trace(
        go.Bar(x=metric_names, y=original_values, name='Original Data',
              marker_color='#F24236'),
        row=4, col=2
    )
    fig.add_trace(
        go.Bar(x=metric_names, y=processed_values, name='Processed Data',
              marker_color='#2E86AB'),
        row=4, col=2
    )
    
    fig.update_layout(
        height=1000,
        title_text="<b>Comprehensive Data Preprocessing for Bridge SHM</b>",
        title_x=0.5,
        showlegend=True,
        font=dict(size=10)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=3, col=2)
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=4, col=1)
    fig.update_xaxes(title_text="Metric", row=4, col=2)
    fig.update_yaxes(title_text="Value", row=4, col=2)
    
    fig.show()
    
    print("=== DATA PREPROCESSING RESULTS ===")
    print(f"Total data points: {len(time)}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Duration: {time[-1]:.1f} seconds")
    
    print(f"\nMissing data segments: {len([1 for start, end in [(2000, 2200), (5500, 5800), (9000, 9100)]])}")
    print(f"Outliers detected: {np.sum(outlier_locations)}")
    print(f"Baseline shifts detected: {len(shift_locations)}")
    
    print(f"\nQuality Improvement:")
    print(f"MSE: {mse_original:.6f} → {mse_processed:.6f} ({(mse_processed/mse_original-1)*100:+.1f}%)")
    print(f"Correlation: {correlation_original:.4f} → {correlation_processed:.4f}")
    print(f"SNR: {snr_original:.2f} → {snr_processed:.2f} dB ({snr_processed-snr_original:+.2f} dB improvement)")
    
    return {
        'original': contaminated,
        'processed': final_signal,
        'clean_reference': clean_signal,
        'preprocessing_steps': preprocessing_steps,
        'quality_metrics': metrics,
        'time': time
    }

# Run comprehensive preprocessing demonstration
preprocess_results = comprehensive_data_preprocessing()
```

---

## 4.7 Digital Filtering Fundamentals

### 4.7.1 Understanding Digital Filters in SHM

Digital filtering is essential for extracting relevant information from noisy structural response data. Filters enable separation of structural signals from environmental noise, extraction of specific frequency bands containing modal information, and removal of unwanted frequency components.

The fundamental types of digital filters used in SHM include:

**Low-pass filters**: Remove high-frequency noise while preserving structural dynamics
**High-pass filters**: Remove low-frequency trends and quasi-static components  
**Band-pass filters**: Extract specific frequency ranges containing modal information
**Band-stop (notch) filters**: Remove specific interference frequencies (e.g., 60 Hz power line)

### 4.7.2 Filter Design and Implementation

The design of digital filters involves selecting appropriate filter parameters to meet specific requirements:

**Filter Order**: Higher order provides sharper transition but may introduce phase distortion
**Cutoff Frequency**: Determines the frequency separation point
**Filter Type**: Butterworth, Chebyshev, Elliptic filters have different characteristics

For a Butterworth low-pass filter, the transfer function is:

$H(s) = \frac{1}{1 + (\frac{s}{\omega_c})^{2n}} \tag{4.19}$

where $\omega_c$ is the cutoff frequency and $n$ is the filter order.

### 4.7.3 Practical Implementation: Digital Filtering for SHM

```python
def digital_filtering_for_shm():
    """
    Comprehensive digital filtering examples for structural health monitoring
    """
    # Generate realistic bridge response with multiple frequency components
    fs = 200  # Sampling frequency (Hz)
    duration = 60  # Duration (s)
    t = np.linspace(0, duration, int(duration * fs))
    
    # Structural modal responses
    f_modes = [1.8, 4.2, 8.7, 12.3, 18.9]  # Bridge natural frequencies
    modal_response = np.zeros_like(t)
    
    for i, freq in enumerate(f_modes):
        amplitude = 1.0 / (i + 1)  # Decreasing amplitude with mode number
        damping = 0.02 * (1 + i * 0.5)  # Increasing damping
        modal_response += amplitude * np.exp(-damping * 2*np.pi*freq * t) * np.sin(2*np.pi*freq*t)
    
    # Environmental and interference components
    wind_response = 0.8 * np.sin(2*np.pi*0.15*t) + 0.4 * np.sin(2*np.pi*0.3*t)  # Wind loading
    power_interference = 0.3 * np.sin(2*np.pi*60*t)  # 60 Hz power line interference
    high_freq_noise = 0.2 * np.random.randn(len(t))  # High-frequency noise
    low_freq_drift = 0.1 * t / duration  # Low-frequency drift
    
    # Combine all components
    raw_signal = modal_response + wind_response + power_interference + high_freq_noise + low_freq_drift
    
    # Design different types of filters
    from scipy.signal import butter, cheby1, cheby2, ellip, filtfilt, freqz
    
    def design_and_apply_filters(signal, fs):
        """Design and apply various digital filters"""
        nyquist = fs / 2
        
        filters_applied = {}
        filter_designs = {}
        
        # 1. High-pass filter (remove low-frequency drift)
        hp_cutoff = 0.5  # Hz
        b_hp, a_hp = butter(4, hp_cutoff/nyquist, btype='high')
        signal_hp = filtfilt(b_hp, a_hp, signal)
        filters_applied['highpass'] = signal_hp
        filter_designs['highpass'] = (b_hp, a_hp, hp_cutoff)
        
        # 2. Low-pass filter (remove high-frequency noise)
        lp_cutoff = 25  # Hz
        b_lp, a_lp = butter(6, lp_cutoff/nyquist, btype='low')
        signal_lp = filtfilt(b_lp, a_lp, signal)
        filters_applied['lowpass'] = signal_lp
        filter_designs['lowpass'] = (b_lp, a_lp, lp_cutoff)
        
        # 3. Band-pass filter (extract structural modes 1-20 Hz)
        bp_low, bp_high = 1.0, 20.0  # Hz
        b_bp, a_bp = butter(4, [bp_low/nyquist, bp_high/nyquist], btype='band')
        signal_bp = filtfilt(b_bp, a_bp, signal)
        filters_applied['bandpass'] = signal_bp
        filter_designs['bandpass'] = (b_bp, a_bp, (bp_low, bp_high))
        
        # 4. Notch filter (remove 60 Hz power line interference)
        notch_freq = 60.0  # Hz
        Q = 30  # Quality factor
        b_notch, a_notch = butter(2, [(notch_freq-1)/nyquist, (notch_freq+1)/nyquist], btype='bandstop')
        signal_notch = filtfilt(b_notch, a_notch, signal)
        filters_applied['notch'] = signal_notch
        filter_designs['notch'] = (b_notch, a_notch, notch_freq)
        
        # 5. Combined filtering (typical SHM preprocessing chain)
        # High-pass → Band-pass → Notch
        signal_combined = filtfilt(b_hp, a_hp, signal)  # Remove drift
        signal_combined = filtfilt(b_bp, a_bp, signal_combined)  # Extract structural range
        signal_combined = filtfilt(b_notch, a_notch, signal_combined)  # Remove power line
        filters_applied['combined'] = signal_combined
        
        # 6. Compare different filter types for the same cutoff
        filter_types = {}
        cutoff = 10.0  # Hz for comparison
        
        # Butterworth
        b_butter, a_butter = butter(4, cutoff/nyquist, btype='low')
        filter_types['butterworth'] = filtfilt(b_butter, a_butter, signal)
        
        # Chebyshev Type I
        b_cheby1, a_cheby1 = cheby1(4, 0.5, cutoff/nyquist, btype='low')  # 0.5 dB ripple
        filter_types['chebyshev1'] = filtfilt(b_cheby1, a_cheby1, signal)
        
        # Chebyshev Type II
        b_cheby2, a_cheby2 = cheby2(4, 40, cutoff/nyquist, btype='low')  # 40 dB stopband
        filter_types['chebyshev2'] = filtfilt(b_cheby2, a_cheby2, signal)
        
        # Elliptic
        b_ellip, a_ellip = ellip(4, 0.5, 40, cutoff/nyquist, btype='low')
        filter_types['elliptic'] = filtfilt(b_ellip, a_ellip, signal)
        
        return filters_applied, filter_designs, filter_types
    
    # Apply filters
    filtered_signals, filter_configs, filter_comparison = design_and_apply_filters(raw_signal, fs)
    
    # Calculate frequency responses
    def calculate_frequency_responses(filter_configs, fs):
        """Calculate frequency responses for visualization"""
        responses = {}
        
        for filter_name, (b, a, cutoff) in filter_configs.items():
            w, h = freqz(b, a, worN=8000, fs=fs)
            responses[filter_name] = {
                'frequencies': w,
                'magnitude': 20 * np.log10(abs(h)),
                'phase': np.angle(h),
                'cutoff': cutoff
            }
        
        return responses
    
    freq_responses = calculate_frequency_responses(filter_configs, fs)
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Original Signal Components', 'Filter Frequency Responses',
            'Time-Domain Filter Comparison', 'Band-pass Filtering Results',
            'Notch Filter Performance', 'Filter Type Comparison',
            'Combined Filtering Pipeline', 'Spectral Analysis Results'
        ]
    )
    
    # Plot 1: Original signal components
    fig.add_trace(
        go.Scatter(x=t[:2000], y=raw_signal[:2000], name='Raw Signal',
                  line=dict(color='#F24236', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t[:2000], y=modal_response[:2000], name='Structural Modes',
                  line=dict(color='#2E86AB', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=t[:2000], y=wind_response[:2000], name='Wind Loading',
                  line=dict(color='#8338EC', width=1)),
        row=1, col=1
    )
    
    # Plot 2: Frequency responses
    colors = ['#F24236', '#2E86AB', '#8338EC', '#F18F01']
    filter_names = ['highpass', 'lowpass', 'bandpass', 'notch']
    
    for i, (name, color) in enumerate(zip(filter_names, colors)):
        if name in freq_responses:
            resp = freq_responses[name]
            fig.add_trace(
                go.Scatter(x=resp['frequencies'], y=resp['magnitude'],
                          name=f'{name.title()} Filter',
                          line=dict(color=color, width=2)),
                row=1, col=2
            )
    
    # Plot 3: Time-domain comparison
    fig.add_trace(
        go.Scatter(x=t[5000:6000], y=raw_signal[5000:6000], name='Original',
                  line=dict(color='#F24236', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t[5000:6000], y=filtered_signals['highpass'][5000:6000], name='High-pass',
                  line=dict(color='#2E86AB', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t[5000:6000], y=filtered_signals['lowpass'][5000:6000], name='Low-pass',
                  line=dict(color='#8338EC', width=2)),
        row=2, col=1
    )
    
    # Plot 4: Band-pass filtering
    fig.add_trace(
        go.Scatter(x=t[1000:2000], y=raw_signal[1000:2000], name='Original',
                  line=dict(color='#F24236', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=t[1000:2000], y=filtered_signals['bandpass'][1000:2000], name='Band-pass (1-20 Hz)',
                  line=dict(color='#2E86AB', width=3)),
        row=2, col=2
    )
    
    # Plot 5: Notch filter performance
    # Show FFT before and after notch filtering
    fft_original = np.fft.fft(raw_signal[:fs*10])  # 10 seconds
    fft_notched = np.fft.fft(filtered_signals['notch'][:fs*10])
    freqs_fft = np.fft.fftfreq(len(fft_original), 1/fs)
    
    mask = (freqs_fft >= 0) & (freqs_fft <= 100)
    fig.add_trace(
        go.Scatter(x=freqs_fft[mask], y=20*np.log10(np.abs(fft_original[mask])),
                  name='Before Notch Filter', line=dict(color='#F24236', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=freqs_fft[mask], y=20*np.log10(np.abs(fft_notched[mask])),
                  name='After Notch Filter', line=dict(color='#2E86AB', width=2)),
        row=3, col=1
    )
    
    # Mark 60 Hz
    fig.add_vline(x=60, line=dict(color='orange', dash='dash'),
                 annotation_text="60 Hz", row=3, col=1)
    
    # Plot 6: Filter type comparison
    comparison_colors = ['#2E86AB', '#F24236', '#8338EC', '#F18F01']
    comparison_names = ['butterworth', 'chebyshev1', 'chebyshev2', 'elliptic']
    
    for name, color in zip(comparison_names, comparison_colors):
        fig.add_trace(
            go.Scatter(x=t[3000:4000], y=filter_comparison[name][3000:4000],
                      name=name.title(), line=dict(color=color, width=2)),
            row=3, col=2
        )
    
    # Plot 7: Combined filtering pipeline
    pipeline_signals = [
        ('Original', raw_signal, '#F24236'),
        ('High-pass', filtered_signals['highpass'], '#8338EC'),
        ('+ Band-pass', filtered_signals['bandpass'], '#2E86AB'),
        ('+ Notch', filtered_signals['combined'], '#F18F01')
    ]
    
    for name, signal, color in pipeline_signals:
        fig.add_trace(
            go.Scatter(x=t[2000:3000], y=signal[2000:3000], name=name,
                      line=dict(color=color, width=2)),
            row=4, col=1
        )
    
    # Plot 8: Spectral analysis comparison
    # Before and after filtering
    fft_before = np.fft.fft(raw_signal)
    fft_after = np.fft.fft(filtered_signals['combined'])
    freqs = np.fft.fftfreq(len(fft_before), 1/fs)
    
    mask = (freqs >= 0) & (freqs <= 30)
    fig.add_trace(
        go.Scatter(x=freqs[mask], y=20*np.log10(np.abs(fft_before[mask])),
                  name='Before Filtering', line=dict(color='#F24236', width=2)),
        row=4, col=2
    )
    fig.add_trace(
        go.Scatter(x=freqs[mask], y=20*np.log10(np.abs(fft_after[mask])),
                  name='After Combined Filtering', line=dict(color='#2E86AB', width=3)),
        row=4, col=2
    )
    
    # Mark modal frequencies
    for freq in f_modes:
        if freq <= 30:
            fig.add_vline(x=freq, line=dict(color='green', dash='dot'),
                         annotation_text=f"Mode {freq:.1f}Hz", row=4, col=2)
    
    fig.update_layout(
        height=1200,
        title_text="<b>Digital Filtering Applications in Bridge SHM</b>",
        title_x=0.5,
        showlegend=True,
        font=dict(size=10)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
    fig.update_yaxes(title_text="Magnitude (dB)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=3, col=2)
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=4, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=2)
    fig.update_yaxes(title_text="Magnitude (dB)", row=4, col=2)
    
    fig.show()
    
    # Calculate and display filtering performance metrics
    def calculate_filter_performance(original, filtered, reference, fs):
        """Calculate performance metrics for filtering"""
        # Signal-to-noise ratio improvement
        noise_original = original - reference
        noise_filtered = filtered - reference
        
        snr_original = 10 * np.log10(np.var(reference) / np.var(noise_original))
        snr_filtered = 10 * np.log10(np.var(reference) / np.var(noise_filtered))
        snr_improvement = snr_filtered - snr_original
        
        # Correlation with reference
        corr_original = np.corrcoef(original, reference)[0, 1]
        corr_filtered = np.corrcoef(filtered, reference)[0, 1]
        
        # Root mean square error
        rmse_original = np.sqrt(np.mean((original - reference)**2))
        rmse_filtered = np.sqrt(np.mean((filtered - reference)**2))
        
        return {
            'snr_improvement': snr_improvement,
            'correlation_improvement': corr_filtered - corr_original,
            'rmse_improvement': (rmse_original - rmse_filtered) / rmse_original * 100
        }
    
    # Evaluate performance
    performance = calculate_filter_performance(
        raw_signal, filtered_signals['combined'], modal_response, fs
    )
    
    print("=== DIGITAL FILTERING RESULTS ===")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Signal duration: {duration} seconds")
    print(f"Modal frequencies: {f_modes} Hz")
    
    print("\nFilter Configurations:")
    for name, (b, a, cutoff) in filter_configs.items():
        print(f"{name.capitalize()}: Order {len(b)-1}, Cutoff: {cutoff}")
    
    print(f"\nFiltering Performance:")
    print(f"SNR improvement: {performance['snr_improvement']:+.2f} dB")
    print(f"Correlation improvement: {performance['correlation_improvement']:+.4f}")
    print(f"RMSE improvement: {performance['rmse_improvement']:+.1f}%")
    
    return {
        'original_signal': raw_signal,
        'modal_response': modal_response,
        'filtered_signals': filtered_signals,
        'frequency_responses': freq_responses,
        'filter_comparison': filter_comparison,
        'performance': performance,
        'time': t
    }

# Run digital filtering demonstration
filter_results = digital_filtering_for_shm()
```

---

## 4.8 Sampling Theory and Anti-Aliasing

### 4.8.1 The Nyquist-Shannon Sampling Theorem

The Nyquist-Shannon sampling theorem states that a continuous signal can be perfectly reconstructed from its samples if the sampling frequency is greater than twice the highest frequency component in the signal. This fundamental principle governs all digital signal processing in SHM systems.

The Nyquist frequency is defined as:

$f_{Nyquist} = \frac{f_s}{2} \tag{4.20}$

where $f_s$ is the sampling frequency. When signal components exist above the Nyquist frequency, aliasing occurs, where high-frequency components appear as false low-frequency components in the sampled data.

### 4.8.2 Anti-Aliasing Filter Design

Anti-aliasing filters must be applied before sampling to prevent frequency folding. These analog filters should have cutoff frequencies well below the Nyquist frequency to account for filter roll-off characteristics.

The guard band ratio is commonly set to:

$\text{Guard Band Ratio} = \frac{f_s}{f_{max}} \geq 2.56 \tag{4.21}$

where $f_{max}$ is the maximum frequency of interest. This provides adequate protection against aliasing while accounting for practical filter limitations.

### 4.8.3 Practical Implementation: Sampling and Anti-Aliasing

```python
def sampling_and_antialiasing_demo():
    """
    Comprehensive demonstration of sampling theory and anti-aliasing for SHM
    """
    # Create high-resolution reference signal
    fs_high = 1000  # High sampling rate for "continuous" reference
    duration = 4  # seconds
    t_continuous = np.linspace(0, duration, int(duration * fs_high))
    
    # Generate realistic bridge response with multiple frequency components
    def generate_bridge_signal(t):
        """Generate realistic bridge response"""
        # Structural modes (typical for a bridge)
        modes = [
            (2.1, 1.0, 0.02),   # (frequency, amplitude, damping)
            (5.7, 0.6, 0.015),
            (8.9, 0.4, 0.025),
            (12.3, 0.3, 0.03),
            (18.5, 0.2, 0.035),
            (25.7, 0.15, 0.04),  # Higher frequency mode
            (45.2, 0.1, 0.05),   # Very high frequency
            (67.8, 0.05, 0.06)   # Should be filtered out
        ]
        
        signal = np.zeros_like(t)
        for freq, amp, damp in modes:
            damped_oscillation = amp * np.exp(-damp * 2*np.pi*freq * t) * np.sin(2*np.pi*freq*t)
            signal += damped_oscillation
        
        # Add wind loading (very low frequency)
        signal += 0.3 * np.sin(2*np.pi*0.2*t) + 0.15 * np.sin(2*np.pi*0.5*t)
        
        # Add measurement noise
        signal += 0.05 * np.random.randn(len(t))
        
        return signal
    
    # Generate the continuous reference signal
    continuous_signal = generate_bridge_signal(t_continuous)
    
    # Demonstrate aliasing effects with different sampling rates
    sampling_rates = [20, 40, 100, 200]  # Hz
    sampled_signals = {}
    
    for fs in sampling_rates:
        # Calculate sampling interval
        dt = 1.0 / fs
        
        # Create time vector for this sampling rate
        n_samples = int(duration * fs)
        t_sampled = np.linspace(0, duration, n_samples)
        
        # Sample the continuous signal (without anti-aliasing filter)
        # Interpolate continuous signal at sampling points
        sampled_unfiltered = np.interp(t_sampled, t_continuous, continuous_signal)
        
        # Apply ideal anti-aliasing filter (low-pass at Nyquist frequency)
        from scipy.signal import butter, filtfilt
        
        # Design anti-aliasing filter
        nyquist = fs / 2
        cutoff = nyquist * 0.8  # 80% of Nyquist to account for filter roll-off
        
        # Filter the high-resolution signal first
        b_aa, a_aa = butter(6, cutoff/500, btype='low')  # Normalize to high fs
        filtered_continuous = filtfilt(b_aa, a_aa, continuous_signal)
        
        # Then sample the filtered signal
        sampled_filtered = np.interp(t_sampled, t_continuous, filtered_continuous)
        
        sampled_signals[fs] = {
            'time': t_sampled,
            'unfiltered': sampled_unfiltered,
            'filtered': sampled_filtered,
            'nyquist': nyquist,
            'cutoff': cutoff
        }
    
    # Demonstrate reconstruction using different methods
    def reconstruct_signal(t_samples, x_samples, t_reconstruct, method='linear'):
        """Reconstruct signal using different interpolation methods"""
        if method == 'linear':
            return np.interp(t_reconstruct, t_samples, x_samples)
        elif method == 'sinc':
            # Sinc interpolation (ideal reconstruction)
            fs = 1.0 / (t_samples[1] - t_samples[0])
            reconstructed = np.zeros_like(t_reconstruct)
            
            for i, t_recon in enumerate(t_reconstruct):
                sinc_sum = 0
                for j, (t_samp, x_samp) in enumerate(zip(t_samples, x_samples)):
                    if abs(t_recon - t_samp) < 1e-10:
                        sinc_sum += x_samp
                    else:
                        sinc_arg = np.pi * fs * (t_recon - t_samp)
                        sinc_sum += x_samp * np.sin(sinc_arg) / sinc_arg
                
                reconstructed[i] = sinc_sum
            
            return reconstructed
    
    # Analyze frequency content using FFT
    def analyze_frequency_content(signal, fs):
        """Analyze frequency content of signal"""
        n = len(signal)
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(n, 1/fs)
        
        # Take only positive frequencies
        positive_mask = frequencies >= 0
        return frequencies[positive_mask], np.abs(fft_result[positive_mask])
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Continuous vs Sampled Signals', 'Frequency Content Analysis',
            'Aliasing Effects at Different Sampling Rates', 'Anti-Aliasing Filter Performance',
            'Signal Reconstruction Comparison', 'Practical Sampling Guidelines'
        ],
        specs=[[{}, {}], [{}, {}], [{}, {}]]
    )
    
    # Plot 1: Continuous vs sampled signals
    # Show a short time window to clearly see sampling points
    t_window = (t_continuous >= 1) & (t_continuous <= 3)
    
    fig.add_trace(
        go.Scatter(x=t_continuous[t_window], y=continuous_signal[t_window],
                  name='Continuous Signal', line=dict(color='#2E86AB', width=3)),
        row=1, col=1
    )
    
    # Show sampling points for 100 Hz case
    fs_demo = 100
    t_samples_demo = sampled_signals[fs_demo]['time']
    x_samples_demo = sampled_signals[fs_demo]['filtered']
    sample_window = (t_samples_demo >= 1) & (t_samples_demo <= 3)
    
    fig.add_trace(
        go.Scatter(x=t_samples_demo[sample_window], y=x_samples_demo[sample_window],
                  mode='markers', name=f'Samples ({fs_demo} Hz)',
                  marker=dict(color='#F24236', size=8)),
        row=1, col=1
    )
    
    # Add reconstruction
    t_recon = t_continuous[t_window]
    x_recon = reconstruct_signal(t_samples_demo, x_samples_demo, t_recon, method='linear')
    fig.add_trace(
        go.Scatter(x=t_recon, y=x_recon, name='Linear Reconstruction',
                  line=dict(color='#F18F01', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Plot 2: Frequency content analysis
    freq_cont, mag_cont = analyze_frequency_content(continuous_signal, fs_high)
    mask_freq = freq_cont <= 80  # Show up to 80 Hz
    
    fig.add_trace(
        go.Scatter(x=freq_cont[mask_freq], y=20*np.log10(mag_cont[mask_freq]),
                  name='Original Spectrum', line=dict(color='#2E86AB', width=2)),
        row=1, col=2
    )
    
    # Show Nyquist frequencies for different sampling rates
    colors = ['#F24236', '#8338EC', '#F18F01', '#C73E1D']
    for i, (fs, color) in enumerate(zip(sampling_rates, colors)):
        fig.add_vline(x=fs/2, line=dict(color=color, dash='dash'),
                     annotation_text=f"{fs}Hz Nyquist", row=1, col=2)
    
    # Plot 3: Aliasing effects
    # Compare properly sampled vs undersampled signals
    fs_good = 200  # Adequate sampling
    fs_bad = 40    # Inadequate sampling (aliasing expected)
    
    # Time window for comparison
    t_comp = np.linspace(0, 2, 1000)
    
    # Good sampling
    x_good_recon = reconstruct_signal(sampled_signals[fs_good]['time'],
                                    sampled_signals[fs_good]['filtered'],
                                    t_comp, method='linear')
    
    # Bad sampling (with aliasing)
    x_bad_recon = reconstruct_signal(sampled_signals[fs_bad]['time'],
                                   sampled_signals[fs_bad]['unfiltered'],
                                   t_comp, method='linear')
    
    # Reference (interpolated continuous)
    x_ref = np.interp(t_comp, t_continuous, continuous_signal)
    
    fig.add_trace(
        go.Scatter(x=t_comp, y=x_ref, name='Reference',
                  line=dict(color='#2E86AB', width=3)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_comp, y=x_good_recon, name=f'Good Sampling ({fs_good} Hz)',
                  line=dict(color='#2E8B57', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_comp, y=x_bad_recon, name=f'Aliased ({fs_bad} Hz)',
                  line=dict(color='#F24236', width=2)),
        row=2, col=1
    )
    
    # Plot 4: Anti-aliasing filter performance
    # Show filtered vs unfiltered for undersampled case
    fig.add_trace(
        go.Scatter(x=sampled_signals[fs_bad]['time'], 
                  y=sampled_signals[fs_bad]['unfiltered'],
                  name='Without Anti-aliasing', mode='lines+markers',
                  line=dict(color='#F24236', width=2),
                  marker=dict(size=4)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=sampled_signals[fs_bad]['time'],
                  y=sampled_signals[fs_bad]['filtered'],
                  name='With Anti-aliasing', mode='lines+markers',
                  line=dict(color='#2E86AB', width=2),
                  marker=dict(size=4)),
        row=2, col=2
    )
    
    # Plot 5: Reconstruction methods comparison
    # Compare linear vs sinc interpolation
    fs_recon = 100
    t_sparse = sampled_signals[fs_recon]['time'][::2]  # Every other sample
    x_sparse = sampled_signals[fs_recon]['filtered'][::2]
    
    t_dense = np.linspace(0, 2, 1000)
    x_linear = reconstruct_signal(t_sparse, x_sparse, t_dense, method='linear')
    # For sinc, use a simpler approximation due to computational complexity
    x_sinc = reconstruct_signal(t_sparse, x_sparse, t_dense, method='linear')  # Simplified
    
    fig.add_trace(
        go.Scatter(x=t_dense, y=x_linear, name='Linear Interpolation',
                  line=dict(color='#F18F01', width=2)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=t_sparse, y=x_sparse, mode='markers', name='Sample Points',
                  marker=dict(color='#F24236', size=8)),
        row=3, col=1
    )
    
    # Plot 6: Practical guidelines
    # Show recommended sampling rates for different applications
    guidelines_text = [
        "SHM Sampling Guidelines:",
        "",
        "• Bridge monitoring: 100-200 Hz",
        "• High-rise buildings: 200-500 Hz", 
        "• Impact testing: 1-2 kHz",
        "• Ultrasonic testing: 10+ MHz",
        "",
        "Anti-aliasing requirements:",
        "• Guard band ratio ≥ 2.56",
        "• Filter order ≥ 6",
        "• Cutoff ≤ 0.8 × f_Nyquist"
    ]
    
    for i, text in enumerate(guidelines_text):
        fig.add_annotation(
            x=0.05, y=0.95-i*0.08,
            xref="paper", yref="paper",
            text=text, showarrow=False,
            font=dict(size=12), align="left"
        )
    
    fig.update_layout(
        height=1000,
        title_text="<b>Sampling Theory and Anti-Aliasing in Bridge SHM</b>",
        title_x=0.5,
        showlegend=True,
        font=dict(size=11)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=3, col=1)
    
    fig.show()
    
    # Calculate aliasing errors
    print("=== SAMPLING AND ANTI-ALIASING ANALYSIS ===")
    print(f"Reference signal duration: {duration} seconds")
    print(f"High-resolution sampling rate: {fs_high} Hz")
    
    print("\nSampling Rate Analysis:")
    for fs in sampling_rates:
        nyquist = fs / 2
        max_freq_captured = min(nyquist, 80)  # Assume max structural frequency is 80 Hz
        
        # Calculate reconstruction error
        t_ref = np.linspace(0, duration, 1000)
        x_ref = np.interp(t_ref, t_continuous, continuous_signal)
        x_recon = reconstruct_signal(sampled_signals[fs]['time'],
                                   sampled_signals[fs]['filtered'],
                                   t_ref, method='linear')
        
        rmse = np.sqrt(np.mean((x_ref - x_recon)**2))
        correlation = np.corrcoef(x_ref, x_recon)[0, 1]
        
        print(f"  {fs} Hz sampling:")
        print(f"    Nyquist frequency: {nyquist} Hz")
        print(f"    Max captured frequency: {max_freq_captured} Hz")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    Correlation: {correlation:.4f}")
        
        if fs >= 160:  # Adequate for bridge monitoring
            print("    ✓ Adequate for bridge SHM")
        elif fs >= 80:
            print("    ⚠ Marginal for bridge SHM")
        else:
            print("    ✗ Inadequate for bridge SHM")
    
    print(f"\nAnti-aliasing Filter Requirements:")
    for fs in sampling_rates:
        cutoff_ideal = fs / 2.56  # Guard band
        print(f"  {fs} Hz sampling: Cutoff ≤ {cutoff_ideal:.1f} Hz")
    
    return {
        'continuous_signal': continuous_signal,
        'sampled_signals': sampled_signals,
        'sampling_rates': sampling_rates,
        'time_continuous': t_continuous
    }

# Run sampling and anti-aliasing demonstration
sampling_results = sampling_and_antialiasing_demo()
```

---

## 4.9 Robust Handling of Outliers and Missing Data

### 4.9.1 Understanding Data Quality Issues in SHM

Real-world structural monitoring systems frequently encounter data quality problems that can significantly impact analysis results. Missing data occurs due to sensor failures, communication interruptions, or power outages, while outliers result from electromagnetic interference, sensor malfunctions, or extreme environmental conditions.

Effective handling of these issues is crucial for:
- Maintaining continuity of monitoring systems
- Ensuring reliable damage detection
- Preventing false alarms
- Supporting automated analysis pipelines

### 4.9.2 Outlier Detection Methods

**Statistical Approaches**:

Z-Score Method:
$Z = \frac{|x - \mu|}{\sigma} > \text{threshold} \tag{4.22}$

Modified Z-Score (robust to outliers):
$M = \frac{0.6745(x - \text{median})}{\text{MAD}} > 3.5 \tag{4.23}$

where MAD is the median absolute deviation.

**Advanced Methods**: Machine learning approaches like Isolation Forest and Local Outlier Factor can detect complex outlier patterns in high-dimensional SHM data.

### 4.9.3 Missing Data Imputation Strategies

**Simple Methods**:
- Linear interpolation for short gaps
- Mean/median substitution for random missing values
- Last observation carried forward (LOCF)

**Advanced Methods**:
- Multiple imputation using chained equations
- Matrix completion techniques
- Autoregressive models for time series gaps

### 4.9.4 Practical Implementation: Robust Data Quality Management

```python
def robust_data_quality_management():
    """
    Comprehensive data quality management for SHM systems
    """
    # Generate realistic bridge monitoring data with quality issues
    def generate_contaminated_dataset(days=7, fs=50):
        """Generate a week of bridge monitoring data with realistic quality issues"""
        total_samples = days * 24 * 3600 * fs
        t = np.linspace(0, days * 24 * 3600, total_samples)
        
        # Base structural response
        base_signal = (
            2.0 * np.sin(2*np.pi*2.1*t/3600) * np.exp(-0.001*t/3600) +  # Modal response
            1.0 * np.sin(2*np.pi*5.4*t/3600) * np.exp(-0.0005*t/3600) +
            0.5 * np.sin(2*np.pi*8.7*t/3600) * np.exp(-0.0002*t/3600)
        )
        
        # Environmental loading patterns
        daily_cycle = 0.8 * np.sin(2*np.pi*t/(24*3600))  # Daily temperature cycle
        weekly_cycle = 0.3 * np.sin(2*np.pi*t/(7*24*3600))  # Weekly traffic pattern
        
        # Traffic loading (higher during work hours)
        hour_of_day = (t % (24*3600)) / 3600
        traffic_intensity = np.where((hour_of_day >= 6) & (hour_of_day <= 20), 1.0, 0.3)
        
        # Generate random vehicle events
        np.random.seed(42)  # For reproducibility
        n_vehicles = int(len(t) * 0.001)  # 0.1% of time points have vehicle events
        vehicle_indices = np.random.choice(len(t), n_vehicles, replace=False)
        vehicle_response = np.zeros_like(t)
        
        for idx in vehicle_indices:
            if idx < len(t) - 100:  # Ensure we don't go out of bounds
                duration = np.random.randint(20, 100)  # 0.4-2 second events
                magnitude = np.random.uniform(0.5, 2.0) * traffic_intensity[idx]
                pulse = magnitude * np.exp(-np.linspace(0, 5, duration)**2)
                end_idx = min(idx + duration, len(t))
                vehicle_response[idx:end_idx] += pulse[:end_idx-idx]
        
        # Combine signal components
        clean_signal = base_signal + daily_cycle + weekly_cycle + vehicle_response
        
        # Add measurement noise
        noise = 0.1 * np.random.randn(len(t))
        signal_with_noise = clean_signal + noise
        
        # Introduce data quality issues
        contaminated_signal = signal_with_noise.copy()
        quality_mask = np.ones_like(t, dtype=bool)
        
        # 1. Missing data segments (sensor failures)
        missing_segments = [
            (int(0.2*len(t)), int(0.21*len(t))),   # 2.4 hour outage on day 2
            (int(0.45*len(t)), int(0.46*len(t))),  # 2.4 hour outage on day 4
            (int(0.7*len(t)), int(0.705*len(t))),  # 1.2 hour outage on day 6
        ]
        
        for start, end in missing_segments:
            contaminated_signal[start:end] = np.nan
            quality_mask[start:end] = False
        
        # 2. Random missing points (communication errors)
        n_random_missing = int(0.001 * len(t))  # 0.1% random missing
        random_missing_idx = np.random.choice(len(t), n_random_missing, replace=False)
        contaminated_signal[random_missing_idx] = np.nan
        quality_mask[random_missing_idx] = False
        
        # 3. Outliers (electromagnetic interference, sensor spikes)
        n_outliers = int(0.0005 * len(t))  # 0.05% outliers
        outlier_indices = np.random.choice(len(t), n_outliers, replace=False)
        outlier_magnitudes = np.random.uniform(-5, 5, n_outliers)
        
        # Different types of outliers
        spike_outliers = outlier_indices[:n_outliers//2]  # Isolated spikes
        contaminated_signal[spike_outliers] += outlier_magnitudes[:n_outliers//2]
        
        # Persistent outliers (sensor drift)
        drift_start = outlier_indices[n_outliers//2]
        drift_duration = min(500, len(t) - drift_start)  # Up to 10 seconds of drift
        drift_magnitude = np.random.uniform(-2, 2)
        contaminated_signal[drift_start:drift_start+drift_duration] += drift_magnitude
        
        # 4. Sensor calibration shifts
        shift_times = [int(0.3*len(t)), int(0.6*len(t))]  # Days 3 and 5
        shift_magnitudes = [0.5, -0.3]
        
        for shift_time, magnitude in zip(shift_times, shift_magnitudes):
            contaminated_signal[shift_time:] += magnitude
        
        return {
            'time': t,
            'clean_signal': clean_signal,
            'contaminated_signal': contaminated_signal,
            'quality_mask': quality_mask,
            'missing_segments': missing_segments,
            'outlier_indices': outlier_indices,
            'shift_times': shift_times
        }
    
    # Generate dataset
    data = generate_contaminated_dataset(days=7, fs=50)
    
    # Initialize quality control pipeline
    processed_signal = data['contaminated_signal'].copy()
    quality_steps = {}
    
    # Step 1: Outlier Detection and Flagging
    def detect_outliers_comprehensive(signal, methods=['zscore', 'modified_zscore', 'iqr']):
        """Comprehensive outlier detection using multiple methods"""
        outlier_flags = {}
        combined_outliers = np.zeros_like(signal, dtype=bool)
        
        # Remove NaN values for outlier detection
        valid_data = signal[~np.isnan(signal)]
        valid_indices = np.where(~np.isnan(signal))[0]
        
        if len(valid_data) == 0:
            return outlier_flags, combined_outliers
        
        for method in methods:
            method_outliers = np.zeros_like(signal, dtype=bool)
            
            if method == 'zscore':
                # Standard Z-score
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                z_scores = np.abs((valid_data - mean_val) / std_val)
                method_flags = z_scores > 3.0
                method_outliers[valid_indices[method_flags]] = True
                
            elif method == 'modified_zscore':
                # Modified Z-score using MAD
                median_val = np.median(valid_data)
                mad = np.median(np.abs(valid_data - median_val))
                if mad > 0:
                    modified_z = 0.6745 * (valid_data - median_val) / mad
                    method_flags = np.abs(modified_z) > 3.5
                    method_outliers[valid_indices[method_flags]] = True
                
            elif method == 'iqr':
                # Interquartile range method
                Q1 = np.percentile(valid_data, 25)
                Q3 = np.percentile(valid_data, 75)
                IQR = Q3 - Q1
                if IQR > 0:
                    method_flags = ((valid_data < Q1 - 1.5*IQR) | 
                                  (valid_data > Q3 + 1.5*IQR))
                    method_outliers[valid_indices[method_flags]] = True
            
            outlier_flags[method] = method_outliers
            combined_outliers |= method_outliers
        
        # Additional: Isolation Forest for complex patterns
        if len(valid_data) > 100:
            from sklearn.ensemble import IsolationForest
            
            # Prepare features (value, local slope, local variance)
            features = []
            for i in range(len(valid_data)):
                value = valid_data[i]
                
                # Local slope (using neighbors)
                if i > 0 and i < len(valid_data) - 1:
                    slope = (valid_data[i+1] - valid_data[i-1]) / 2
                else:
                    slope = 0
                
                # Local variance (using sliding window)
                window_start = max(0, i-10)
                window_end = min(len(valid_data), i+11)
                local_var = np.var(valid_data[window_start:window_end])
                
                features.append([value, slope, local_var])
            
            features = np.array(features)
            
            # Apply Isolation Forest
            iso_forest = IsolationForest(contamination=0.01, random_state=42)
            anomaly_labels = iso_forest.fit_predict(features)
            
            iso_outliers = np.zeros_like(signal, dtype=bool)
            iso_flags = anomaly_labels == -1
            iso_outliers[valid_indices[iso_flags]] = True
            
            outlier_flags['isolation_forest'] = iso_outliers
            combined_outliers |= iso_outliers
        
        return outlier_flags, combined_outliers
    
    outlier_results, outlier_mask = detect_outliers_comprehensive(processed_signal)
    quality_steps['outlier_detection'] = {
        'signal': processed_signal.copy(),
        'outliers': outlier_results,
        'combined_mask': outlier_mask
    }
    
    # Step 2: Missing Data Imputation
    def impute_missing_data(signal, method='adaptive'):
        """Advanced missing data imputation"""
        result = signal.copy()
        imputation_mask = np.isnan(signal)
        
        if not np.any(imputation_mask):
            return result, imputation_mask
        
        if method == 'adaptive':
            # Adaptive imputation based on gap length
            
            # Find continuous missing segments
            missing_segments = []
            in_segment = False
            segment_start = 0
            
            for i in range(len(imputation_mask)):
                if imputation_mask[i] and not in_segment:
                    # Start of missing segment
                    in_segment = True
                    segment_start = i
                elif not imputation_mask[i] and in_segment:
                    # End of missing segment
                    in_segment = False
                    missing_segments.append((segment_start, i))
            
            # Handle last segment if it ends at array end
            if in_segment:
                missing_segments.append((segment_start, len(imputation_mask)))
            
            # Impute each segment based on its length
            for start, end in missing_segments:
                gap_length = end - start
                
                if gap_length <= 5:  # Short gaps: linear interpolation
                    # Find nearest valid points
                    before_idx = start - 1 if start > 0 else None
                    after_idx = end if end < len(result) else None
                    
                    if before_idx is not None and after_idx is not None:
                        # Both boundaries available
                        before_val = result[before_idx]
                        after_val = result[after_idx]
                        interpolated = np.linspace(before_val, after_val, gap_length + 2)[1:-1]
                        result[start:end] = interpolated
                    elif before_idx is not None:
                        # Only before value available
                        result[start:end] = result[before_idx]
                    elif after_idx is not None:
                        # Only after value available
                        result[start:end] = result[after_idx]
                
                elif gap_length <= 100:  # Medium gaps: polynomial interpolation
                    # Use wider context for polynomial fitting
                    context_size = min(50, start, len(result) - end)
                    
                    if context_size > 10:
                        # Collect context data
                        before_context = result[start-context_size:start]
                        after_context = result[end:end+context_size]
                        
                        if not np.any(np.isnan(before_context)) and not np.any(np.isnan(after_context)):
                            # Fit polynomial to context
                            context_data = np.concatenate([before_context, after_context])
                            context_indices = np.concatenate([
                                np.arange(start-context_size, start),
                                np.arange(end, end+context_size)
                            ])
                            
                            # Fit polynomial (degree 3)
                            poly_coeffs = np.polyfit(context_indices, context_data, 3)
                            missing_indices = np.arange(start, end)
                            interpolated = np.polyval(poly_coeffs, missing_indices)
                            result[start:end] = interpolated
                        else:
                            # Fall back to linear interpolation
                            if start > 0 and end < len(result):
                                before_val = result[start-1]
                                after_val = result[end]
                                interpolated = np.linspace(before_val, after_val, gap_length + 2)[1:-1]
                                result[start:end] = interpolated
                
                else:  # Long gaps: statistical imputation
                    # Use seasonal decomposition or AR model
                    # For simplicity, use mean of surrounding periods
                    
                    # Find daily/weekly patterns
                    fs = 50  # From our generation
                    samples_per_hour = fs * 3600
                    
                    # Look for same time in previous days
                    replacement_candidates = []
                    
                    for day_offset in range(1, 8):  # Check previous 7 days
                        candidate_start = start - day_offset * 24 * samples_per_hour
                        candidate_end = end - day_offset * 24 * samples_per_hour
                        
                        if candidate_start >= 0 and candidate_end < len(result):
                            candidate_data = result[candidate_start:candidate_end]
                            if not np.any(np.isnan(candidate_data)):
                                replacement_candidates.append(candidate_data)
                    
                    if replacement_candidates:
                        # Use mean of candidates
                        mean_replacement = np.mean(replacement_candidates, axis=0)
                        result[start:end] = mean_replacement
                    else:
                        # Fall back to median of entire dataset
                        valid_data = result[~np.isnan(result)]
                        if len(valid_data) > 0:
                            result[start:end] = np.median(valid_data)
        
        return result, imputation_mask
    
    processed_signal, missing_mask = impute_missing_data(processed_signal, method='adaptive')
    quality_steps['missing_data_imputed'] = {
        'signal': processed_signal.copy(),
        'missing_mask': missing_mask
    }
    
    # Step 3: Outlier Replacement
    def replace_outliers(signal, outlier_mask, method='interpolation'):
        """Replace identified outliers"""
        result = signal.copy()
        
        if method == 'interpolation':
            # Replace outliers with interpolated values
            outlier_indices = np.where(outlier_mask)[0]
            
            for idx in outlier_indices:
                # Find nearest non-outlier neighbors
                search_radius = 10
                left_neighbor = None
                right_neighbor = None
                
                # Search left
                for i in range(idx-1, max(-1, idx-search_radius), -1):
                    if not outlier_mask[i]:
                        left_neighbor = (i, result[i])
                        break
                
                # Search right
                for i in range(idx+1, min(len(result), idx+search_radius+1)):
                    if not outlier_mask[i]:
                        right_neighbor = (i, result[i])
                        break
                
                # Interpolate
                if left_neighbor and right_neighbor:
                    # Linear interpolation
                    left_idx, left_val = left_neighbor
                    right_idx, right_val = right_neighbor
                    
                    weight = (idx - left_idx) / (right_idx - left_idx)
                    result[idx] = left_val + weight * (right_val - left_val)
                    
                elif left_neighbor:
                    result[idx] = left_neighbor[1]
                elif right_neighbor:
                    result[idx] = right_neighbor[1]
        
        elif method == 'median_filter':
            # Use median filtering to replace outliers
            from scipy.signal import medfilt
            filtered = medfilt(signal, kernel_size=5)
            result[outlier_mask] = filtered[outlier_mask]
        
        return result
    
    processed_signal = replace_outliers(processed_signal, outlier_mask, method='interpolation')
    quality_steps['outliers_replaced'] = processed_signal.copy()
    
    # Step 4: Trend and Drift Correction
    def correct_drifts_and_shifts(signal, detection_method='change_point'):
        """Detect and correct sensor drifts and calibration shifts"""
        result = signal.copy()
        
        if detection_method == 'change_point':
            # Simple change point detection using moving statistics
            window_size = 500
            
            # Calculate moving mean
            moving_mean = pd.Series(signal).rolling(window_size, center=True).mean()
            
            # Find significant changes in moving mean
            mean_diff = np.abs(np.diff(moving_mean.fillna(method='bfill').fillna(method='ffill')))
            threshold = 3 * np.std(mean_diff[~np.isnan(mean_diff)])
            
            change_points = np.where(mean_diff > threshold)[0]
            
            # Correct shifts
            for cp in change_points:
                if cp < len(signal) - window_size:
                    # Calculate shift magnitude
                    before_segment = signal[max(0, cp-window_size//2):cp]
                    after_segment = signal[cp:cp+window_size//2]
                    
                    before_mean = np.nanmean(before_segment)
                    after_mean = np.nanmean(after_segment)
                    shift = after_mean - before_mean
                    
                    # Apply correction
                    result[cp:] -= shift
        
        return result, change_points if 'change_points' in locals() else []
    
    processed_signal, detected_shifts = correct_drifts_and_shifts(processed_signal)
    quality_steps['drift_corrected'] = {
        'signal': processed_signal.copy(),
        'shift_points': detected_shifts
    }
    
    # Final step: Quality assessment
    def assess_data_quality(original, processed, reference):
        """Assess the quality improvement achieved"""
        
        # Remove NaN values for comparison
        valid_mask = ~(np.isnan(original) | np.isnan(processed) | np.isnan(reference))
        
        if np.sum(valid_mask) == 0:
            return {}
        
        orig_valid = original[valid_mask]
        proc_valid = processed[valid_mask]
        ref_valid = reference[valid_mask]
        
        metrics = {
            'data_recovery_rate': np.sum(~np.isnan(processed)) / len(processed),
            'rmse_original': np.sqrt(np.mean((orig_valid - ref_valid)**2)),
            'rmse_processed': np.sqrt(np.mean((proc_valid - ref_valid)**2)),
            'correlation_original': np.corrcoef(orig_valid, ref_valid)[0,1] if len(ref_valid) > 1 else 0,
            'correlation_processed': np.corrcoef(proc_valid, ref_valid)[0,1] if len(ref_valid) > 1 else 0,
            'outlier_removal_rate': np.sum(outlier_mask) / len(outlier_mask),
            'missing_data_rate': np.sum(missing_mask) / len(missing_mask)
        }
        
        return metrics
    
    quality_metrics = assess_data_quality(
        data['contaminated_signal'], 
        processed_signal, 
        data['clean_signal']
    )
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Original vs Contaminated Signal', 'Outlier Detection Results',
            'Missing Data Imputation', 'Drift and Shift Correction',
            'Processing Pipeline Comparison', 'Quality Metrics Assessment',
            'Long-term Quality Trends', 'Final Quality Control Results'
        ]
    )
    
    # Convert time to hours for better visualization
    time_hours = data['time'] / 3600
    
    # Plot 1: Original vs contaminated
    fig.add_trace(
        go.Scatter(x=time_hours[:5000], y=data['clean_signal'][:5000],
                  name='Clean Reference', line=dict(color='#2E86AB', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_hours[:5000], y=data['contaminated_signal'][:5000],
                  name='Contaminated', line=dict(color='#F24236', width=1)),
        row=1, col=1
    )
    
    # Plot 2: Outlier detection
    contaminated_no_nan = data['contaminated_signal'].copy()
    contaminated_no_nan[np.isnan(contaminated_no_nan)] = 0  # For visualization
    
    fig.add_trace(
        go.Scatter(x=time_hours, y=contaminated_no_nan,
                  name='Signal with Outliers', line=dict(color='#F24236', width=1)),
        row=1, col=2
    )
    
    # Mark detected outliers
    outlier_times = time_hours[outlier_mask]
    outlier_values = contaminated_no_nan[outlier_mask]
    fig.add_trace(
        go.Scatter(x=outlier_times, y=outlier_values, mode='markers',
                  name='Detected Outliers', 
                  marker=dict(color='#F18F01', size=6, symbol='x')),
        row=1, col=2
    )
    
    # Plot 3: Missing data imputation
    fig.add_trace(
        go.Scatter(x=time_hours, y=quality_steps['missing_data_imputed']['signal'],
                  name='After Imputation', line=dict(color='#2E86AB', width=2)),
        row=2, col=1
    )
    
    # Mark originally missing segments
    missing_times = time_hours[missing_mask]
    missing_values = quality_steps['missing_data_imputed']['signal'][missing_mask]
    fig.add_trace(
        go.Scatter(x=missing_times, y=missing_values, mode='markers',
                  name='Imputed Data', 
                  marker=dict(color='#8338EC', size=4, symbol='square')),
        row=2, col=1
    )
    
    # Plot 4: Drift correction
    fig.add_trace(
        go.Scatter(x=time_hours, y=quality_steps['outliers_replaced'],
                  name='Before Drift Correction', line=dict(color='#F24236', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=time_hours, y=processed_signal,
                  name='After Drift Correction', line=dict(color='#2E86AB', width=2)),
        row=2, col=2
    )
    
    # Plot 5: Processing pipeline
    pipeline_signals = [
        ('Original Contaminated', data['contaminated_signal']),
        ('Missing Data Imputed', quality_steps['missing_data_imputed']['signal']),
        ('Outliers Replaced', quality_steps['outliers_replaced']),
        ('Final Processed', processed_signal)
    ]
    
    colors = ['#F24236', '#8338EC', '#F18F01', '#2E86AB']
    
    # Show a focused time window
    focus_start, focus_end = 2000, 3000
    for i, (name, signal) in enumerate(pipeline_signals):
        fig.add_trace(
            go.Scatter(x=time_hours[focus_start:focus_end], 
                      y=signal[focus_start:focus_end],
                      name=name, line=dict(color=colors[i], width=2)),
            row=3, col=1
        )
    
    # Plot 6: Quality metrics
    metric_names = ['RMSE\nImprovement', 'Correlation\nImprovement', 
                   'Data Recovery\nRate', 'Outlier Removal\nRate']
    
    rmse_improvement = (quality_metrics['rmse_original'] - quality_metrics['rmse_processed']) / quality_metrics['rmse_original'] * 100
    corr_improvement = (quality_metrics['correlation_processed'] - quality_metrics['correlation_original']) * 100
    
    metric_values = [
        rmse_improvement,
        corr_improvement,
        quality_metrics['data_recovery_rate'] * 100,
        quality_metrics['outlier_removal_rate'] * 100
    ]
    
    fig.add_trace(
        go.Bar(x=metric_names, y=metric_