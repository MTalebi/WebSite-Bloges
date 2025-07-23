# Chapter 5: Frequency-Domain Analysis for Structural Health Monitoring

**Instructor:** Mohammad Talebi-Kalaleh – University of Alberta

---

## 5.1 Introduction and Motivation

Understanding how structures vibrate and respond to various excitations is fundamental to structural health monitoring (SHM). While time-domain analysis provides immediate insights into structural response, frequency-domain analysis reveals the underlying modal characteristics that define a structure's dynamic fingerprint. Bridge health monitoring systems have demonstrated that frequency-domain analysis exhibits better robustness, heightened sensitivity to parameter variations, and eliminates the need for modal parameter identification compared to time-domain approaches.

When analyzing the dynamic behavior of bridges, engineers face several challenges. Bridge monitoring data is typically influenced by operational vehicle loads, environmental conditions, and measurement noise. Frequency-domain techniques help overcome these limitations by revealing the structure's inherent modal properties—natural frequencies, mode shapes, and damping characteristics—that remain relatively stable despite changing operational conditions.

Modern structural health monitoring systems increasingly rely on frequency-domain processing, where vibration data collected during train crossings or traffic loading is processed using Fast Fourier Transform (FFT) and analyzed using machine learning to detect anomalies that indicate potential structural issues. This chapter explores the mathematical foundations and practical applications of these powerful analytical tools.

### Why Frequency-Domain Analysis Matters in SHM

The frequency domain provides unique advantages for structural monitoring:

1. **Modal Identification**: Reveals natural frequencies, mode shapes, and damping ratios
2. **Damage Detection**: Structural damage typically manifests as changes in modal parameters
3. **Noise Robustness**: Frequency averaging reduces the impact of measurement noise
4. **Operational Convenience**: Works with ambient excitation without requiring controlled input forces
5. **Real-time Monitoring**: Enables automated detection of structural changes

---

## 5.2 Mathematical Foundations

### 5.2.1 The Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT)

The foundation of frequency-domain analysis lies in the Fourier transform, which decomposes a time-domain signal into its constituent frequency components. For a discrete time signal $x[n]$ with $N$ samples, the Discrete Fourier Transform is defined as:

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N} \qquad \text{(5.1)}$$

where:
- $X[k]$ is the frequency-domain representation at frequency bin $k$
- $x[n]$ is the time-domain signal at sample $n$  
- $j$ is the imaginary unit
- $k = 0, 1, 2, ..., N-1$ represents frequency bins
- $N$ is the total number of samples

The frequency resolution $\Delta f$ of the DFT is determined by:

$$\Delta f = \frac{f_s}{N} \qquad \text{(5.2)}$$

where $f_s$ is the sampling frequency in Hz.

The Fast Fourier Transform (FFT) is an efficient algorithm for computing the DFT, reducing computational complexity from $O(N^2)$ to $O(N \log N)$. This efficiency makes real-time frequency analysis possible for SHM applications.

#### Understanding FFT Output for SHM

The FFT produces complex-valued coefficients that contain both magnitude and phase information:

$$X[k] = |X[k]| e^{j\phi[k]} \qquad \text{(5.3)}$$

where:
- $|X[k]| = \sqrt{\text{Re}(X[k])^2 + \text{Im}(X[k])^2}$ is the magnitude
- $\phi[k] = \arctan\left(\frac{\text{Im}(X[k])}{\text{Re}(X[k])}\right)$ is the phase

For structural analysis, we primarily focus on the magnitude spectrum, which reveals the energy content at each frequency.

### 5.2.2 Power Spectral Density (PSD)

Power spectral density (PSD) is considered the gold standard of vibration analysis because it normalizes the amplitude to frequency bin width, allowing accurate comparison of random vibration signals that have different signal lengths.

The one-sided power spectral density is calculated as:

$$S_{xx}(f) = \frac{2|X(f)|^2}{f_s \cdot N} \qquad \text{(5.4)}$$

where:
- $S_{xx}(f)$ has units of $\text{(signal units)}^2/\text{Hz}$
- The factor of 2 accounts for negative frequencies (except at DC and Nyquist)
- For acceleration signals, units are typically $\text{m}^2/\text{s}^4/\text{Hz}$ or $\text{g}^2/\text{Hz}$

#### Cross-Power Spectral Density

For multi-channel measurements common in bridge monitoring, the cross-power spectral density between signals $x(t)$ and $y(t)$ is:

$$S_{xy}(f) = \frac{2X^*(f)Y(f)}{f_s \cdot N} \qquad \text{(5.5)}$$

where $X^*(f)$ denotes the complex conjugate of $X(f)$.

The cross-PSD contains both magnitude and phase information, crucial for understanding spatial relationships in structural response.

---

## 5.3 Conceptual Framework for Frequency-Domain SHM

```svg
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .title-text { font: bold 16px Arial; fill: #2c3e50; }
      .header-text { font: bold 14px Arial; fill: #34495e; }
      .regular-text { font: 12px Arial; fill: #2c3e50; }
      .box { fill: #ecf0f1; stroke: #34495e; stroke-width: 2; }
      .process-box { fill: #3498db; stroke: #2980b9; stroke-width: 2; }
      .analysis-box { fill: #e74c3c; stroke: #c0392b; stroke-width: 2; }
      .output-box { fill: #27ae60; stroke: #229954; stroke-width: 2; }
      .arrow { stroke: #34495e; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#34495e"/>
    </marker>
  </defs>
  
  <!-- Title -->
  <text x="400" y="25" text-anchor="middle" class="title-text">Frequency-Domain Analysis Workflow for Bridge SHM</text>
  
  <!-- Input Data -->
  <rect x="50" y="60" width="120" height="60" class="box"/>
  <text x="110" y="85" text-anchor="middle" class="header-text">Time-Domain</text>
  <text x="110" y="100" text-anchor="middle" class="regular-text">Bridge Response</text>
  <text x="110" y="115" text-anchor="middle" class="regular-text">Acceleration/Strain</text>
  
  <!-- Preprocessing -->
  <rect x="220" y="60" width="120" height="60" class="process-box"/>
  <text x="280" y="85" text-anchor="middle" class="header-text" fill="white">Preprocessing</text>
  <text x="280" y="100" text-anchor="middle" class="regular-text" fill="white">• Detrending</text>
  <text x="280" y="115" text-anchor="middle" class="regular-text" fill="white">• Windowing</text>
  
  <!-- Transform Methods -->
  <rect x="50" y="180" width="100" height="80" class="analysis-box"/>
  <text x="100" y="205" text-anchor="middle" class="header-text" fill="white">FFT</text>
  <text x="100" y="220" text-anchor="middle" class="regular-text" fill="white">Basic spectral</text>
  <text x="100" y="235" text-anchor="middle" class="regular-text" fill="white">analysis</text>
  <text x="100" y="250" text-anchor="middle" class="regular-text" fill="white">Modal peaks</text>
  
  <rect x="170" y="180" width="100" height="80" class="analysis-box"/>
  <text x="220" y="205" text-anchor="middle" class="header-text" fill="white">PSD</text>
  <text x="220" y="220" text-anchor="middle" class="regular-text" fill="white">Random</text>
  <text x="220" y="235" text-anchor="middle" class="regular-text" fill="white">vibration</text>
  <text x="220" y="250" text-anchor="middle" class="regular-text" fill="white">analysis</text>
  
  <rect x="290" y="180" width="100" height="80" class="analysis-box"/>
  <text x="340" y="205" text-anchor="middle" class="header-text" fill="white">STFT</text>
  <text x="340" y="220" text-anchor="middle" class="regular-text" fill="white">Time-frequency</text>
  <text x="340" y="235" text-anchor="middle" class="regular-text" fill="white">Non-stationary</text>
  <text x="340" y="250" text-anchor="middle" class="regular-text" fill="white">signals</text>
  
  <rect x="410" y="180" width="100" height="80" class="analysis-box"/>
  <text x="460" y="205" text-anchor="middle" class="header-text" fill="white">Wavelets</text>
  <text x="460" y="220" text-anchor="middle" class="regular-text" fill="white">Multi-resolution</text>
  <text x="460" y="235" text-anchor="middle" class="regular-text" fill="white">Transient</text>
  <text x="460" y="250" text-anchor="middle" class="regular-text" fill="white">detection</text>
  
  <!-- Modal Identification -->
  <rect x="530" y="180" width="120" height="80" class="analysis-box"/>
  <text x="590" y="200" text-anchor="middle" class="header-text" fill="white">Modal ID</text>
  <text x="590" y="215" text-anchor="middle" class="regular-text" fill="white">• Peak Picking</text>
  <text x="590" y="230" text-anchor="middle" class="regular-text" fill="white">• FDD/EFDD</text>
  <text x="590" y="245" text-anchor="middle" class="regular-text" fill="white">• SVD Methods</text>
  
  <!-- Outputs -->
  <rect x="50" y="320" width="110" height="60" class="output-box"/>
  <text x="105" y="340" text-anchor="middle" class="header-text" fill="white">Natural</text>
  <text x="105" y="355" text-anchor="middle" class="header-text" fill="white">Frequencies</text>
  <text x="105" y="370" text-anchor="middle" class="regular-text" fill="white">Modal peaks</text>
  
  <rect x="180" y="320" width="110" height="60" class="output-box"/>
  <text x="235" y="340" text-anchor="middle" class="header-text" fill="white">Mode Shapes</text>
  <text x="235" y="355" text-anchor="middle" class="regular-text" fill="white">Spatial patterns</text>
  <text x="235" y="370" text-anchor="middle" class="regular-text" fill="white">Deformation</text>
  
  <rect x="310" y="320" width="110" height="60" class="output-box"/>
  <text x="365" y="340" text-anchor="middle" class="header-text" fill="white">Damping</text>
  <text x="365" y="355" text-anchor="middle" class="header-text" fill="white">Ratios</text>
  <text x="365" y="370" text-anchor="middle" class="regular-text" fill="white">Energy dissipation</text>
  
  <rect x="440" y="320" width="110" height="60" class="output-box"/>
  <text x="495" y="340" text-anchor="middle" class="header-text" fill="white">Damage</text>
  <text x="495" y="355" text-anchor="middle" class="header-text" fill="white">Indicators</text>
  <text x="495" y="370" text-anchor="middle" class="regular-text" fill="white">Frequency shifts</text>
  
  <rect x="570" y="320" width="110" height="60" class="output-box"/>
  <text x="625" y="340" text-anchor="middle" class="header-text" fill="white">Operational</text>
  <text x="625" y="355" text-anchor="middle" class="header-text" fill="white">Features</text>
  <text x="625" y="370" text-anchor="middle" class="regular-text" fill="white">PSDT, etc.</text>
  
  <!-- SHM Applications -->
  <rect x="250" y="450" width="200" height="80" class="box"/>
  <text x="350" y="470" text-anchor="middle" class="header-text">SHM Applications</text>
  <text x="350" y="490" text-anchor="middle" class="regular-text">• Real-time monitoring</text>
  <text x="350" y="505" text-anchor="middle" class="regular-text">• Damage detection</text>
  <text x="350" y="520" text-anchor="middle" class="regular-text">• Condition assessment</text>
  
  <!-- Arrows -->
  <line x1="170" y1="90" x2="220" y2="90" class="arrow"/>
  <line x1="280" y1="120" x2="280" y2="180" class="arrow"/>
  
  <!-- Transform to outputs arrows -->
  <line x1="100" y1="260" x2="105" y2="320" class="arrow"/>
  <line x1="220" y1="260" x2="235" y2="320" class="arrow"/>
  <line x1="340" y1="260" x2="365" y2="320" class="arrow"/>
  <line x1="460" y1="260" x2="495" y2="320" class="arrow"/>
  <line x1="590" y1="260" x2="625" y2="320" class="arrow"/>
  
  <!-- Outputs to applications -->
  <line x1="350" y1="380" x2="350" y2="450" class="arrow"/>
</svg>
```

**Figure 5.1:** Comprehensive workflow for frequency-domain analysis in structural health monitoring, showing the progression from raw time-domain data to actionable SHM information.

---

## 5.4 Fast Fourier Transform (FFT) for Bridge Analysis

### 5.4.1 Implementation and Practical Considerations

The FFT is the computational workhorse of frequency-domain analysis. For bridge monitoring applications, several practical considerations are crucial:

**Windowing:** To reduce spectral leakage, signals are typically multiplied by window functions. The Hanning window is commonly used for random vibration analysis:

$$w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right) \qquad \text{(5.6)}$$

**Overlap Processing:** To maintain good time resolution while reducing variance, overlapped processing is employed, typically with 50% overlap between consecutive windows.

**Averaging:** Multiple FFT estimates are averaged to reduce noise and improve statistical reliability:

$$\overline{S}_{xx}(f) = \frac{1}{K} \sum_{i=1}^{K} S_{xx}^{(i)}(f) \qquad \text{(5.7)}$$

where $K$ is the number of averaged spectra.

### 5.4.2 Python Implementation for Bridge Data Analysis

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

def generate_bridge_acceleration_data(duration=60, fs=100, bridge_modes=None):
    """
    Generate realistic bridge acceleration data with multiple modal responses
    
    Parameters:
    -----------
    duration : float
        Signal duration in seconds
    fs : float  
        Sampling frequency in Hz
    bridge_modes : list of tuples
        Each tuple contains (frequency_hz, damping_ratio, amplitude)
        
    Returns:
    --------
    t : ndarray
        Time vector
    acceleration : ndarray
        Acceleration signal in m/s²
    """
    
    if bridge_modes is None:
        # Typical bridge modal parameters for a medium-span bridge
        bridge_modes = [
            (1.85, 0.02, 0.15),   # First bending mode
            (3.42, 0.025, 0.08),  # Second bending mode  
            (5.78, 0.03, 0.05),   # First torsional mode
            (7.95, 0.028, 0.03),  # Third bending mode
            (12.3, 0.035, 0.02),  # Higher order mode
        ]
    
    t = np.linspace(0, duration, int(fs * duration))
    acceleration = np.zeros_like(t)
    
    # Add modal responses with realistic characteristics
    for freq, damping, amplitude in bridge_modes:
        # Damped sinusoidal response with random phase
        phase = np.random.uniform(0, 2*np.pi)
        modal_response = amplitude * np.exp(-damping * 2 * np.pi * freq * t) * \
                        np.sin(2 * np.pi * freq * t + phase)
        acceleration += modal_response
    
    # Add ambient noise (traffic, wind, measurement noise)
    noise_level = 0.02  # m/s²
    acceleration += np.random.normal(0, noise_level, len(t))
    
    # Add some low-frequency content (traffic loading)
    traffic_freq = 0.3  # Hz
    traffic_amplitude = 0.05  # m/s²
    acceleration += traffic_amplitude * np.sin(2 * np.pi * traffic_freq * t)
    
    return t, acceleration

def compute_fft_spectrum(signal_data, fs, window='hann', nperseg=None, noverlap=None):
    """
    Compute FFT spectrum with proper windowing and averaging
    
    Parameters:
    -----------
    signal_data : ndarray
        Input time series
    fs : float
        Sampling frequency
    window : str
        Window function type
    nperseg : int
        Length of each segment for averaging
    noverlap : int
        Number of points to overlap between segments
        
    Returns:
    --------
    frequencies : ndarray
        Frequency vector in Hz
    magnitude : ndarray  
        Magnitude spectrum
    phase : ndarray
        Phase spectrum in radians
    """
    
    if nperseg is None:
        nperseg = min(len(signal_data), 4096)
    if noverlap is None:
        noverlap = nperseg // 2
        
    # Use Welch's method for better spectral estimates
    frequencies, psd = signal.welch(signal_data, fs, window=window, 
                                   nperseg=nperseg, noverlap=noverlap)
    
    # Convert PSD to magnitude spectrum
    magnitude = np.sqrt(psd * fs)
    
    # Compute phase spectrum using FFT of entire signal
    fft_result = fft(signal_data * signal.get_window(window, len(signal_data)))
    phase = np.angle(fft_result[:len(frequencies)])
    
    return frequencies, magnitude, phase

# Generate realistic bridge acceleration data
print("Generating realistic bridge acceleration data...")
t, accel = generate_bridge_acceleration_data(duration=120, fs=100)

# Create DataFrame for better data handling
data = pd.DataFrame({
    'Time_s': t,
    'Acceleration_ms2': accel
})

print("Data characteristics:")
print(f"Duration: {t[-1]:.1f} seconds")
print(f"Sampling frequency: {1/(t[1]-t[0]):.1f} Hz") 
print(f"RMS acceleration: {np.sqrt(np.mean(accel**2)):.4f} m/s²")
print(f"Peak acceleration: {np.max(np.abs(accel)):.4f} m/s²")

# Compute FFT spectrum
freq, magnitude, phase = compute_fft_spectrum(accel, fs=100)

# Create comprehensive plots
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=['Time-Domain Signal', 'Frequency Spectrum (Magnitude)',
                   'Time Series Detail (0-10s)', 'Phase Spectrum',  
                   'Frequency Spectrum (Log Scale)', 'Modal Peak Analysis'],
    specs=[[{"colspan": 1}, {"colspan": 1}],
           [{"colspan": 1}, {"colspan": 1}],
           [{"colspan": 1}, {"colspan": 1}]]
)

# Time domain plot
fig.add_trace(go.Scatter(x=t, y=accel, name='Bridge Acceleration',
                        line=dict(color='#2E86AB', width=1)),
              row=1, col=1)

# Frequency domain magnitude 
fig.add_trace(go.Scatter(x=freq, y=magnitude, name='Magnitude Spectrum',
                        line=dict(color='#A23B72', width=2)),
              row=1, col=2)

# Time series detail
detail_mask = t <= 10
fig.add_trace(go.Scatter(x=t[detail_mask], y=accel[detail_mask], 
                        name='Signal Detail',
                        line=dict(color='#F18F01', width=1.5)),
              row=2, col=1)

# Phase spectrum
fig.add_trace(go.Scatter(x=freq, y=np.degrees(phase), name='Phase (degrees)',
                        line=dict(color='#C73E1D', width=1.5)),
              row=2, col=2)

# Log scale magnitude
fig.add_trace(go.Scatter(x=freq, y=magnitude, name='Log Magnitude',
                        line=dict(color='#2E86AB', width=2)),
              row=3, col=1)

# Find and annotate modal peaks
from scipy.signal import find_peaks
peaks, properties = find_peaks(magnitude[:len(freq)//4], height=0.001, distance=10)
peak_freqs = freq[peaks]
peak_mags = magnitude[peaks]

fig.add_trace(go.Scatter(x=peak_freqs, y=peak_mags, 
                        mode='markers', name='Modal Peaks',
                        marker=dict(size=10, color='red', symbol='diamond')),
              row=3, col=2)

fig.add_trace(go.Scatter(x=freq[:len(freq)//4], y=magnitude[:len(freq)//4], 
                        name='Spectrum (0-12.5 Hz)',
                        line=dict(color='#2E86AB', width=2)),
              row=3, col=2)

# Update layout
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=3, col=1)
fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=2)

fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
fig.update_yaxes(title_text="Magnitude (m/s²)", row=1, col=2)
fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=1)
fig.update_yaxes(title_text="Phase (degrees)", row=2, col=2)
fig.update_yaxes(title_text="Magnitude (m/s²)", type="log", row=3, col=1)
fig.update_yaxes(title_text="Magnitude (m/s²)", row=3, col=2)

fig.update_layout(height=900, title_text="FFT Analysis of Bridge Acceleration Data",
                 showlegend=False, template="plotly_white")

fig.show()

# Print identified modal frequencies
print("\nIdentified Modal Frequencies:")
print("=" * 40)
for i, (f, mag) in enumerate(zip(peak_freqs, peak_mags)):
    if f > 0.5 and f < 15:  # Focus on structural modes
        print(f"Mode {i+1:2d}: {f:6.2f} Hz (Magnitude: {mag:.4f} m/s²)")
```

This implementation demonstrates several key concepts:

1. **Realistic Data Generation**: The synthetic bridge data includes multiple modal responses with appropriate damping characteristics
2. **Proper Windowing**: Uses Hanning windowing to reduce spectral leakage  
3. **Averaging**: Employs Welch's method for improved spectral estimates
4. **Peak Detection**: Automatically identifies modal frequencies
5. **Comprehensive Visualization**: Shows both time and frequency domain representations

---

## 5.5 Power Spectral Density (PSD) Analysis

### 5.5.1 Theory and Applications in SHM

Power spectral density transmissibility (PSDT) has emerged as a robust structural feature that is less sensitive to operational vehicle loads while maintaining high sensitivity to structural parameter changes. The PSDT is defined as:

$$T_{xy}(f) = \frac{S_{xy}(f)}{S_{xx}(f)} \qquad \text{(5.8)}$$

where $S_{xy}(f)$ is the cross-PSD between responses at locations $x$ and $y$, and $S_{xx}(f)$ is the auto-PSD at the reference location.

For bridge monitoring, PSDT offers several advantages:
- **Operational invariance**: Less affected by changing traffic patterns
- **Damage sensitivity**: Responds to stiffness changes more clearly than raw PSD
- **Noise robustness**: Cross-correlation reduces uncorrelated noise effects

### 5.5.2 Advanced PSD Implementation

```python
def compute_advanced_psd(signal1, signal2=None, fs=100, method='welch', 
                        nperseg=2048, noverlap=None, window='hann'):
    """
    Compute auto-PSD and cross-PSD with advanced processing options
    
    Parameters:
    -----------
    signal1, signal2 : ndarray
        Input signals (if signal2 is None, computes auto-PSD of signal1)
    fs : float
        Sampling frequency
    method : str
        PSD estimation method ('welch', 'periodogram', 'multitaper')
    nperseg : int
        Length of each segment
    noverlap : int
        Number of overlapping points
    window : str
        Window function
        
    Returns:
    --------
    frequencies : ndarray
        Frequency vector
    psd : ndarray
        Power spectral density
    coherence : ndarray (if cross-PSD)
        Magnitude squared coherence
    """
    
    if noverlap is None:
        noverlap = nperseg // 2
        
    if signal2 is None:
        # Auto-PSD computation
        if method == 'welch':
            frequencies, psd = signal.welch(signal1, fs, window=window,
                                          nperseg=nperseg, noverlap=noverlap)
        elif method == 'periodogram':
            frequencies, psd = signal.periodogram(signal1, fs, window=window)
        
        return frequencies, psd, None
    
    else:
        # Cross-PSD and coherence computation
        frequencies, cross_psd = signal.csd(signal1, signal2, fs, window=window,
                                          nperseg=nperseg, noverlap=noverlap)
        
        frequencies, coherence = signal.coherence(signal1, signal2, fs, window=window,
                                                nperseg=nperseg, noverlap=noverlap)
        
        return frequencies, cross_psd, coherence

def compute_psdt(reference_signal, response_signals, fs=100):
    """
    Compute Power Spectral Density Transmissibility matrix
    
    Parameters:
    -----------
    reference_signal : ndarray
        Reference acceleration signal
    response_signals : ndarray (2D)
        Matrix of response signals (each column is a measurement point)
    fs : float
        Sampling frequency
        
    Returns:
    --------
    frequencies : ndarray
        Frequency vector
    psdt_matrix : ndarray (complex)
        PSDT matrix (frequency x locations)
    """
    
    n_locations = response_signals.shape[1]
    
    # Compute reference auto-PSD
    freq, ref_psd, _ = compute_advanced_psd(reference_signal, fs=fs)
    
    # Initialize PSDT matrix
    psdt_matrix = np.zeros((len(freq), n_locations), dtype=complex)
    
    for i in range(n_locations):
        # Cross-PSD between reference and each response point
        _, cross_psd, _ = compute_advanced_psd(reference_signal, 
                                             response_signals[:, i], fs=fs)
        
        # PSDT calculation (avoid division by zero)
        psdt_matrix[:, i] = cross_psd / (ref_psd + 1e-12)
    
    return freq, psdt_matrix

# Generate multi-point bridge data for PSDT analysis
def generate_multipoint_bridge_data(n_points=5, duration=60, fs=100):
    """Generate acceleration data at multiple points along bridge span"""
    
    t = np.linspace(0, duration, int(fs * duration))
    
    # Bridge modes with spatial distribution
    modes = [
        {'freq': 1.85, 'damping': 0.02, 'shape': np.sin(np.pi * np.linspace(0, 1, n_points))},
        {'freq': 3.42, 'damping': 0.025, 'shape': np.sin(2 * np.pi * np.linspace(0, 1, n_points))},
        {'freq': 5.78, 'damping': 0.03, 'shape': np.sin(3 * np.pi * np.linspace(0, 1, n_points))},
    ]
    
    # Initialize response matrix
    responses = np.zeros((len(t), n_points))
    
    for mode in modes:
        # Modal amplitude varies with time (ambient excitation)
        modal_amplitude = 0.1 * np.random.normal(0, 1, len(t))
        modal_amplitude = signal.filtfilt(*signal.butter(4, 0.5, fs=fs), modal_amplitude)
        
        # Apply modal shape to each point
        for i in range(n_points):
            damped_response = modal_amplitude * mode['shape'][i] * \
                             np.exp(-mode['damping'] * 2 * np.pi * mode['freq'] * t)
            responses[:, i] += damped_response
    
    # Add uncorrelated noise at each point
    noise_level = 0.01
    for i in range(n_points):
        responses[:, i] += np.random.normal(0, noise_level, len(t))
    
    return t, responses

# Generate multi-point data
print("Generating multi-point bridge acceleration data...")
t_multi, responses = generate_multipoint_bridge_data(n_points=4, duration=120, fs=100)

# Select reference point (typically at maximum response location)
reference_idx = 1  # Second point as reference
reference_signal = responses[:, reference_idx]
response_signals = responses

# Compute PSDT
freq_psdt, psdt_matrix = compute_psdt(reference_signal, response_signals, fs=100)

# Compute regular PSDs for comparison  
freq_psd, ref_psd, _ = compute_advanced_psd(reference_signal, fs=100)

# Create comprehensive PSD analysis plots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Multi-Point Time Signals', 'Auto-PSD at Reference Point',
                   'PSDT Magnitude', 'PSDT Phase'],
    specs=[[{"colspan": 1}, {"colspan": 1}],
           [{"colspan": 1}, {"colspan": 1}]]
)

# Plot time signals from all points
colors = px.colors.qualitative.Set1
for i in range(responses.shape[1]):
    fig.add_trace(go.Scatter(x=t_multi[:1000], y=responses[:1000, i], 
                            name=f'Point {i+1}',
                            line=dict(color=colors[i], width=1.5)),
                  row=1, col=1)

# Reference point auto-PSD
fig.add_trace(go.Scatter(x=freq_psd, y=ref_psd, 
                        name='Reference Auto-PSD',
                        line=dict(color='#2E86AB', width=2)),
              row=1, col=2)

# PSDT magnitude for all points
for i in range(psdt_matrix.shape[1]):
    fig.add_trace(go.Scatter(x=freq_psdt, y=np.abs(psdt_matrix[:, i]), 
                            name=f'PSDT Point {i+1}',
                            line=dict(color=colors[i], width=2)),
                  row=2, col=1)

# PSDT phase for all points  
for i in range(psdt_matrix.shape[1]):
    fig.add_trace(go.Scatter(x=freq_psdt, y=np.degrees(np.angle(psdt_matrix[:, i])), 
                            name=f'Phase Point {i+1}',
                            line=dict(color=colors[i], width=2)),
                  row=2, col=2)

# Update layout
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)

fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
fig.update_yaxes(title_text="PSD (m²/s⁴/Hz)", type="log", row=1, col=2)
fig.update_yaxes(title_text="PSDT Magnitude", row=2, col=1)
fig.update_yaxes(title_text="PSDT Phase (degrees)", row=2, col=2)

fig.update_layout(height=800, title_text="Power Spectral Density Transmissibility Analysis",
                 showlegend=True, template="plotly_white")

fig.show()

# Calculate spectral moments for damage detection
def calculate_spectral_moments(freq, psd, freq_range=(1, 10)):
    """Calculate spectral moments within specified frequency range"""
    
    # Find indices within frequency range
    mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
    freq_subset = freq[mask]
    psd_subset = psd[mask]
    
    # Calculate moments
    m0 = np.trapz(psd_subset, freq_subset)  # 0th moment (area under curve)
    m1 = np.trapz(freq_subset * psd_subset, freq_subset)  # 1st moment
    m2 = np.trapz(freq_subset**2 * psd_subset, freq_subset)  # 2nd moment
    
    # Derived parameters
    mean_freq = m1 / m0 if m0 > 0 else 0
    rms_freq = np.sqrt(m2 / m0) if m0 > 0 else 0
    
    return {
        'm0': m0, 'm1': m1, 'm2': m2,
        'mean_frequency': mean_freq,
        'rms_frequency': rms_freq
    }

# Calculate moments for monitoring
moments = calculate_spectral_moments(freq_psd, ref_psd)
print("\nSpectral Moment Analysis:")
print("=" * 30)
print(f"0th moment (m0): {moments['m0']:.6f}")
print(f"Mean frequency: {moments['mean_frequency']:.3f} Hz")
print(f"RMS frequency: {moments['rms_frequency']:.3f} Hz")
```

This advanced PSD implementation showcases:

1. **Multiple estimation methods**: Welch, periodogram, and multitaper options
2. **PSDT calculation**: For operational invariant features
3. **Multi-point analysis**: Spatial correlation in bridge response
4. **Spectral moments**: Quantitative features for damage detection
5. **Coherence analysis**: Understanding signal relationships

---

## 5.6 Short-Time Fourier Transform (STFT) for Non-Stationary Analysis

### 5.6.1 Mathematical Formulation

STFT excels in capturing time-varying power quality disturbances, enabling the extraction of both fundamental and harmonic components with precise temporal resolution, making it particularly valuable for detecting transient changes in structural response.

The STFT of a signal $x(t)$ is defined as:

$$X(m,k) = \sum_{n=0}^{N-1} x[n] w[n-m] e^{-j2\pi kn/N} \qquad \text{(5.9)}$$

where:
- $m$ is the time index (window position)
- $k$ is the frequency index
- $w[n]$ is the window function
- $N$ is the window length

The STFT provides a time-frequency representation that reveals how the spectral content of a signal changes over time—crucial for analyzing bridge response during vehicle crossings or seismic events.

#### Time-Frequency Resolution Trade-off

The STFT involves a fundamental trade-off between time and frequency resolution:

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi} \qquad \text{(5.10)}$$

This uncertainty principle means that improving time resolution (shorter windows) reduces frequency resolution and vice versa.

### 5.6.2 STFT Implementation for Bridge Event Analysis

```python
def compute_stft_analysis(signal_data, fs, nperseg=512, noverlap=None, window='hann'):
    """
    Compute Short-Time Fourier Transform for time-frequency analysis
    
    Parameters:
    -----------
    signal_data : ndarray
        Input time series
    fs : float
        Sampling frequency
    nperseg : int
        Length of each segment (controls frequency resolution)
    noverlap : int
        Number of points to overlap between segments
    window : str
        Window function
        
    Returns:
    --------
    frequencies : ndarray
        Frequency vector
    times : ndarray
        Time vector for STFT
    stft_magnitude : ndarray
        STFT magnitude spectrogram  
    stft_phase : ndarray
        STFT phase spectrogram
    """
    
    if noverlap is None:
        noverlap = nperseg // 2
        
    # Compute STFT
    frequencies, times, stft_complex = signal.stft(signal_data, fs, 
                                                  window=window,
                                                  nperseg=nperseg, 
                                                  noverlap=noverlap)
    
    stft_magnitude = np.abs(stft_complex)
    stft_phase = np.angle(stft_complex)
    
    return frequencies, times, stft_magnitude, stft_phase

def generate_bridge_vehicle_crossing_event(duration=30, fs=100):
    """
    Generate bridge acceleration data during a vehicle crossing event
    Shows time-varying modal content as vehicle approaches, crosses, and leaves
    """
    
    t = np.linspace(0, duration, int(fs * duration))
    acceleration = np.zeros_like(t)
    
    # Vehicle crossing parameters
    vehicle_speed = 25  # m/s
    bridge_length = 50  # m
    crossing_time = bridge_length / vehicle_speed  # seconds
    crossing_start = 10  # seconds into recording
    crossing_end = crossing_start + crossing_time
    
    # Time-varying modal excitation during crossing
    for i, time_val in enumerate(t):
        if crossing_start <= time_val <= crossing_end:
            # Vehicle is on bridge - higher amplitude, frequency modulation
            position = (time_val - crossing_start) / crossing_time
            
            # First mode with position-dependent amplitude
            mode1_amp = 0.2 * np.sin(np.pi * position)  # Max at mid-span
            acceleration[i] += mode1_amp * np.sin(2 * np.pi * 1.85 * time_val)
            
            # Second mode excited more at quarter points
            mode2_amp = 0.1 * np.sin(2 * np.pi * position)
            acceleration[i] += mode2_amp * np.sin(2 * np.pi * 3.42 * time_val)
            
            # Transient frequency due to vehicle dynamics
            instantaneous_freq = 1.85 + 0.3 * np.sin(2 * np.pi * 0.5 * position)
            acceleration[i] += 0.05 * np.sin(2 * np.pi * instantaneous_freq * time_val)
            
        else:
            # Ambient vibration only
            acceleration[i] += 0.02 * np.sin(2 * np.pi * 1.85 * time_val)
            acceleration[i] += 0.01 * np.sin(2 * np.pi * 3.42 * time_val)
    
    # Add noise and ambient excitation
    acceleration += np.random.normal(0, 0.01, len(t))
    
    # Add some transient impacts (vehicles hitting bridge joints)
    impact_times = [8, 22, 25]
    for impact_time in impact_times:
        impact_idx = int(impact_time * fs)
        if impact_idx < len(acceleration):
            # Short duration, high frequency transient
            impact_duration = 0.1  # seconds
            impact_samples = int(impact_duration * fs)
            for j in range(impact_samples):
                idx = impact_idx + j
                if idx < len(acceleration):
                    decay = np.exp(-j / (0.02 * fs))  # Fast decay
                    acceleration[idx] += 0.3 * decay * np.sin(2 * np.pi * 25 * t[idx])
    
    return t, acceleration, crossing_start, crossing_end

# Generate vehicle crossing event data
print("Generating bridge vehicle crossing event data...")
t_event, accel_event, crossing_start, crossing_end = generate_bridge_vehicle_crossing_event()

# Compute STFT with different resolutions
freq_stft_high, time_stft_high, stft_mag_high, stft_phase_high = compute_stft_analysis(
    accel_event, fs=100, nperseg=256, window='hann')

freq_stft_low, time_stft_low, stft_mag_low, stft_phase_low = compute_stft_analysis(
    accel_event, fs=100, nperseg=1024, window='hann')

# Create comprehensive STFT visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=['Vehicle Crossing Event Signal', 'High Time Resolution STFT',
                   'Time-Frequency Detail (0-10 Hz)', 'Low Time Resolution STFT',
                   'Modal Frequency Tracking', 'Phase Evolution'],
    specs=[[{"colspan": 2}, None],
           [{"colspan": 1}, {"colspan": 1}],
           [{"colspan": 1}, {"colspan": 1}]],
    vertical_spacing=0.08
)

# Time domain signal with crossing indicators
fig.add_trace(go.Scatter(x=t_event, y=accel_event, 
                        name='Bridge Acceleration',
                        line=dict(color='#2E86AB', width=1.5)),
              row=1, col=1)

# Add vertical lines for vehicle crossing period
fig.add_vline(x=crossing_start, line_dash="dash", line_color="red", 
              annotation_text="Vehicle enters", row=1, col=1)
fig.add_vline(x=crossing_end, line_dash="dash", line_color="red",
              annotation_text="Vehicle exits", row=1, col=1)

# High time resolution STFT spectrogram
fig.add_trace(go.Heatmap(x=time_stft_high, y=freq_stft_high, z=20*np.log10(stft_mag_high + 1e-12),
                        colorscale='Viridis', name='STFT (High Time Res)',
                        colorbar=dict(title="Magnitude (dB)", x=0.48)),
              row=2, col=1)

# Low time resolution STFT spectrogram  
fig.add_trace(go.Heatmap(x=time_stft_low, y=freq_stft_low, z=20*np.log10(stft_mag_low + 1e-12),
                        colorscale='Plasma', name='STFT (Low Time Res)',
                        colorbar=dict(title="Magnitude (dB)", x=0.98)),
              row=2, col=2)

# Focus on structural frequency range (0-10 Hz) for detailed analysis
freq_mask = freq_stft_high <= 10
fig.add_trace(go.Heatmap(x=time_stft_high, y=freq_stft_high[freq_mask], 
                        z=20*np.log10(stft_mag_high[freq_mask, :] + 1e-12),
                        colorscale='Cividis', name='Structural Modes',
                        colorbar=dict(title="Magnitude (dB)", x=0.48)),
              row=3, col=1)

# Extract modal frequency evolution for first mode
def track_modal_frequency(freq_vector, stft_magnitude, target_freq=1.85, bandwidth=0.5):
    """Track evolution of modal frequency over time"""
    
    freq_mask = (freq_vector >= target_freq - bandwidth) & (freq_vector <= target_freq + bandwidth)
    freq_subset = freq_vector[freq_mask]
    stft_subset = stft_magnitude[freq_mask, :]
    
    modal_freq_evolution = []
    for time_idx in range(stft_subset.shape[1]):
        # Find peak frequency at each time step
        peak_idx = np.argmax(stft_subset[:, time_idx])
        modal_freq_evolution.append(freq_subset[peak_idx])
    
    return np.array(modal_freq_evolution)

modal_freq_track = track_modal_frequency(freq_stft_high, stft_mag_high)

fig.add_trace(go.Scatter(x=time_stft_high, y=modal_freq_track,
                        name='First Mode Tracking',
                        line=dict(color='red', width=3),
                        mode='lines+markers'),
              row=3, col=2)

# Add baseline frequency reference
fig.add_hline(y=1.85, line_dash="dash", line_color="blue",
              annotation_text="Baseline frequency", row=3, col=2)

# Update layout
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=2)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=2)

fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=2)
fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
fig.update_yaxes(title_text="Modal Frequency (Hz)", row=3, col=2)

fig.update_layout(height=1000, title_text="STFT Analysis of Bridge Vehicle Crossing Event",
                 showlegend=False, template="plotly_white")

fig.show()

# Quantitative analysis of frequency variations
crossing_mask = (time_stft_high >= crossing_start) & (time_stft_high <= crossing_end)
baseline_freq = np.mean(modal_freq_track[~crossing_mask])
crossing_freq_variation = modal_freq_track[crossing_mask]

print("\nSTFT Modal Frequency Analysis:")
print("=" * 35)
print(f"Baseline frequency: {baseline_freq:.3f} Hz")
print(f"Max frequency during crossing: {np.max(crossing_freq_variation):.3f} Hz")
print(f"Min frequency during crossing: {np.min(crossing_freq_variation):.3f} Hz")
print(f"Frequency variation range: {np.max(crossing_freq_variation) - np.min(crossing_freq_variation):.3f} Hz")
print(f"Relative variation: {100*(np.max(crossing_freq_variation) - baseline_freq)/baseline_freq:.2f}%")
```

This STFT implementation demonstrates:

1. **Time-frequency resolution trade-offs**: Comparing different window sizes
2. **Event detection**: Identifying vehicle crossing periods
3. **Modal tracking**: Following frequency evolution during loading events
4. **Transient analysis**: Detecting impact events and their frequency content
5. **Quantitative assessment**: Measuring frequency variations during structural loading

---

## 5.7 Wavelet Transform for Multi-Resolution Analysis

### 5.7.1 Continuous Wavelet Transform (CWT)

Wavelet transform has gained popularity as an efficient method of signal processing in SHM, overcoming many limitations of the Fourier transform by providing both time and frequency localization capabilities.

The continuous wavelet transform is defined as:

$$W(a,b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt \qquad \text{(5.11)}$$

where:
- $\psi(t)$ is the mother wavelet
- $a$ is the scale parameter (inversely related to frequency)
- $b$ is the translation parameter (time shift)
- $*$ denotes complex conjugate

The relationship between wavelet scale and frequency is approximately:

$$f = \frac{f_c}{a \cdot \Delta t} \qquad \text{(5.12)}$$

where $f_c$ is the center frequency of the mother wavelet and $\Delta t$ is the sampling period.

### 5.7.2 Wavelet-Based Damage Detection

```python
import pywt
from scipy.signal import hilbert

def wavelet_analysis_bridge(signal_data, fs, wavelet='cmor1.5-1.0', scales=None):
    """
    Comprehensive wavelet analysis for bridge structural health monitoring
    
    Parameters:
    -----------
    signal_data : ndarray
        Input acceleration signal
    fs : float
        Sampling frequency
    wavelet : str
        Mother wavelet type
    scales : ndarray
        Wavelet scales (if None, automatically generated)
        
    Returns:
    --------
    frequencies : ndarray
        Frequency vector corresponding to scales
    times : ndarray
        Time vector
    cwt_coefficients : ndarray (complex)
        Continuous wavelet transform coefficients
    """
    
    # Generate time vector
    dt = 1.0 / fs
    times = np.arange(len(signal_data)) * dt
    
    # Generate scales if not provided
    if scales is None:
        # Focus on structural frequency range (0.1 to 50 Hz)
        min_freq, max_freq = 0.1, 50
        min_scale = pywt.frequency2scale(wavelet, max_freq) / dt
        max_scale = pywt.frequency2scale(wavelet, min_freq) / dt
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 100)
    
    # Compute CWT
    cwt_coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet, dt)
    
    return frequencies, times, cwt_coefficients

def detect_damage_events(cwt_coefficients, frequencies, times, threshold_percentile=95):
    """
    Detect damage-related events using wavelet analysis
    
    Parameters:
    -----------
    cwt_coefficients : ndarray
        CWT coefficients
    frequencies : ndarray
        Frequency vector
    times : ndarray
        Time vector
    threshold_percentile : float
        Percentile for anomaly detection threshold
        
    Returns:
    --------
    damage_events : list
        List of detected events with time and frequency info
    """
    
    # Calculate energy at each time-frequency point
    energy = np.abs(cwt_coefficients)**2
    
    # Focus on structural frequency range (1-20 Hz)
    freq_mask = (frequencies >= 1) & (frequencies <= 20)
    structural_energy = energy[freq_mask, :]
    structural_freqs = frequencies[freq_mask]
    
    # Calculate threshold for anomaly detection
    threshold = np.percentile(structural_energy, threshold_percentile)
    
    # Find events exceeding threshold
    anomaly_indices = np.where(structural_energy > threshold)
    
    damage_events = []
    for i in range(len(anomaly_indices[0])):
        freq_idx = anomaly_indices[0][i]
        time_idx = anomaly_indices[1][i]
        
        damage_events.append({
            'time': times[time_idx],
            'frequency': structural_freqs[freq_idx],
            'energy': structural_energy[freq_idx, time_idx],
            'severity': structural_energy[freq_idx, time_idx] / threshold
        })
    
    return damage_events

def generate_bridge_with_damage_progression(duration=180, fs=100):
    """
    Generate bridge data with gradual damage progression
    Simulates stiffness loss over time affecting modal frequencies
    """
    
    t = np.linspace(0, duration, int(fs * duration))
    acceleration = np.zeros_like(t)
    
    # Progressive damage scenario
    initial_freq1 = 1.85  # Hz
    initial_freq2 = 3.42  # Hz
    
    # Damage progression (frequency reduction over time)
    damage_progression = np.exp(-t / 300)  # Exponential stiffness loss
    
    for i, time_val in enumerate(t):
        # Time-varying modal frequencies due to damage
        current_freq1 = initial_freq1 * (0.7 + 0.3 * damage_progression[i])
        current_freq2 = initial_freq2 * (0.8 + 0.2 * damage_progression[i])
        
        # Modal responses with changing frequencies
        acceleration[i] += 0.1 * np.sin(2 * np.pi * current_freq1 * time_val)
        acceleration[i] += 0.05 * np.sin(2 * np.pi * current_freq2 * time_val)
        
        # Add sudden damage events (e.g., crack propagation)
        if time_val in [60, 120]:  # Discrete damage events
            damage_idx = i
            event_duration = 2.0  # seconds
            event_samples = int(event_duration * fs)
            
            for j in range(event_samples):
                if damage_idx + j < len(acceleration):
                    # High-frequency transient from crack propagation
                    decay = np.exp(-j / (0.5 * fs))
                    acceleration[damage_idx + j] += 0.5 * decay * \
                        np.sin(2 * np.pi * 15 * t[damage_idx + j])
    
    # Add ambient noise
    acceleration += np.random.normal(0, 0.02, len(t))
    
    return t, acceleration

# Generate damaged bridge data
print("Generating bridge data with damage progression...")
t_damage, accel_damage = generate_bridge_with_damage_progression(duration=180, fs=100)

# Perform wavelet analysis
freq_wav, time_wav, cwt_coeff = wavelet_analysis_bridge(accel_damage, fs=100, 
                                                       wavelet='cmor1.5-1.0')

# Detect damage events
damage_events = detect_damage_events(cwt_coeff, freq_wav, time_wav)

# Create comprehensive wavelet analysis visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=['Bridge Signal with Damage Progression', 'Wavelet Scalogram (CWT)',
                   'Structural Frequency Range (1-10 Hz)', 'Modal Frequency Evolution',
                   'Damage Event Detection', 'Wavelet Energy Distribution'],
    specs=[[{"colspan": 2}, None],
           [{"colspan": 1}, {"colspan": 1}],
           [{"colspan": 1}, {"colspan": 1}]],
    vertical_spacing=0.1
)

# Time domain signal
fig.add_trace(go.Scatter(x=t_damage, y=accel_damage,
                        name='Acceleration with Damage',
                        line=dict(color='#2E86AB', width=1)),
              row=1, col=1)

# Mark damage events
damage_times = [60, 120]
for dt in damage_times:
    fig.add_vline(x=dt, line_dash="dash", line_color="red",
                  annotation_text=f"Damage Event", row=1, col=1)

# Full wavelet scalogram
wavelet_magnitude = np.abs(cwt_coeff)
fig.add_trace(go.Heatmap(x=time_wav, y=freq_wav, 
                        z=20*np.log10(wavelet_magnitude + 1e-12),
                        colorscale='Viridis', name='CWT Magnitude',
                        colorbar=dict(title="Magnitude (dB)", x=0.98)),
              row=2, col=1)

# Focus on structural frequency range
freq_mask_struct = (freq_wav >= 1) & (freq_wav <= 10)
fig.add_trace(go.Heatmap(x=time_wav, y=freq_wav[freq_mask_struct],
                        z=20*np.log10(wavelet_magnitude[freq_mask_struct, :] + 1e-12),
                        colorscale='Plasma', name='Structural Range',
                        colorbar=dict(title="Magnitude (dB)", x=0.48)),
              row=2, col=2)

# Modal frequency tracking using wavelet ridge extraction
def extract_wavelet_ridges(cwt_coeff, frequencies, n_ridges=2):
    """Extract ridges from wavelet transform for modal tracking"""
    
    ridges = []
    magnitude = np.abs(cwt_coeff)
    
    for time_idx in range(magnitude.shape[1]):
        # Find peaks in frequency domain at each time
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(magnitude[:, time_idx], height=np.max(magnitude[:, time_idx])*0.1)
        
        # Sort by magnitude and take strongest peaks
        peak_magnitudes = magnitude[peaks, time_idx]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        
        time_ridges = []
        for i in range(min(n_ridges, len(sorted_indices))):
            peak_idx = peaks[sorted_indices[i]]
            time_ridges.append(frequencies[peak_idx])
        
        ridges.append(time_ridges)
    
    return np.array(ridges)

modal_ridges = extract_wavelet_ridges(cwt_coeff, freq_wav, n_ridges=2)

# Plot modal frequency evolution
if modal_ridges.shape[1] >= 2:
    fig.add_trace(go.Scatter(x=time_wav, y=modal_ridges[:, 0],
                            name='First Mode',
                            line=dict(color='red', width=3),
                            mode='lines'),
                  row=3, col=1)
    
    fig.add_trace(go.Scatter(x=time_wav, y=modal_ridges[:, 1],
                            name='Second Mode',
                            line=dict(color='blue', width=3),
                            mode='lines'),
                  row=3, col=1)

# Damage event markers
if damage_events:
    event_times = [event['time'] for event in damage_events[:20]]  # Show first 20 events
    event_freqs = [event['frequency'] for event in damage_events[:20]]
    event_severities = [event['severity'] for event in damage_events[:20]]
    
    fig.add_trace(go.Scatter(x=event_times, y=event_freqs,
                            mode='markers', name='Damage Events',
                            marker=dict(size=[min(20, s*5) for s in event_severities],
                                      color=event_severities,
                                      colorscale='Reds',
                                      showscale=True,
                                      colorbar=dict(title="Severity", x=0.48))),
                  row=3, col=2)

# Energy distribution analysis
total_energy = np.sum(np.abs(cwt_coeff)**2, axis=1)
fig.add_trace(go.Scatter(x=total_energy, y=freq_wav,
                        name='Energy Distribution',
                        line=dict(color='green', width=2)),
              row=3, col=2)

# Update layout
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=2)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_xaxes(title_text="Energy / Time (s)", row=3, col=2)

fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=2)
fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=2)

fig.update_layout(height=1000, title_text="Wavelet Analysis for Damage Detection in Bridges",
                 showlegend=True, template="plotly_white")

fig.show()

# Quantitative damage assessment
print("\nWavelet-Based Damage Analysis:")
print("=" * 35)
print(f"Total detected anomalous events: {len(damage_events)}")

if len(damage_events) > 0:
    # Group events by time windows
    time_windows = [(0, 60), (60, 120), (120, 180)]
    for start, end in time_windows:
        window_events = [e for e in damage_events if start <= e['time'] <= end]
        if window_events:
            avg_severity = np.mean([e['severity'] for e in window_events])
            max_severity = np.max([e['severity'] for e in window_events])
            print(f"Time {start}-{end}s: {len(window_events)} events, "
                  f"avg severity: {avg_severity:.2f}, max: {max_severity:.2f}")

# Frequency degradation analysis
if modal_ridges.shape[1] >= 1:
    initial_freq = np.mean(modal_ridges[:10, 0])  # First 10 samples
    final_freq = np.mean(modal_ridges[-10:, 0])   # Last 10 samples
    frequency_loss = 100 * (initial_freq - final_freq) / initial_freq
    
    print(f"\nModal Frequency Analysis:")
    print(f"Initial frequency: {initial_freq:.3f} Hz")
    print(f"Final frequency: {final_freq:.3f} Hz")
    print(f"Frequency loss: {frequency_loss:.2f}%")
```

This wavelet implementation showcases:

1. **Multi-resolution analysis**: Time-frequency decomposition with adaptive resolution
2. **Damage event detection**: Automated identification of anomalous events
3. **Modal tracking**: Following frequency evolution through wavelet ridges
4. **Progressive damage assessment**: Quantifying long-term frequency degradation
5. **Event severity classification**: Ranking damage events by energy content

---

## 5.8 Non-Parametric Modal Identification

### 5.8.1 Peak Picking Method

The simplest non-parametric modal identification approach is peak picking (PP), where natural frequencies are identified directly from peaks in the frequency spectrum. While straightforward, peak picking encounters limitations when closely spaced modes occur, which are invariably present in complex structures like bridges.

For a single-degree-of-freedom (SDOF) system, the frequency response function near a resonance can be approximated as:

$$H(\omega) \approx \frac{1}{2j\omega_n\zeta} \cdot \frac{1}{s - s_n} \qquad \text{(5.13)}$$

where $s_n = -\zeta\omega_n + j\omega_n\sqrt{1-\zeta^2}$ is the complex pole.

### 5.8.2 Frequency Domain Decomposition (FDD)

Frequency Domain Decomposition (FDD) was introduced to address limitations of peak picking by using Singular Value Decomposition (SVD) of the power spectral density matrix to separate closely spaced modes.

For a multi-input, multi-output system, the output PSD matrix can be written as:

$$\mathbf{G}_{yy}(\omega) = \mathbf{H}(\omega) \mathbf{G}_{ff}(\omega) \mathbf{H}^H(\omega) \qquad \text{(5.14)}$$

where:
- $\mathbf{G}_{yy}(\omega)$ is the output PSD matrix
- $\mathbf{H}(\omega)$ is the frequency response matrix
- $\mathbf{G}_{ff}(\omega)$ is the input PSD matrix
- $H$ denotes Hermitian transpose

Under white noise excitation, the SVD of the output PSD matrix at each frequency becomes:

$$\mathbf{G}_{yy}(\omega_i) = \mathbf{U}_i \mathbf{S}_i \mathbf{U}_i^H \qquad \text{(5.15)}$$

where $\mathbf{U}_i$ contains the singular vectors (mode shapes) and $\mathbf{S}_i$ contains singular values.

### 5.8.3 Enhanced Frequency Domain Decomposition (EFDD)

Enhanced FDD (EFDD) improves upon basic FDD by extracting modal damping ratios through correlation function analysis of the modal coordinates in the frequency domain.

The EFDD process involves:

1. **SVD decomposition** at each frequency line
2. **Mode shape identification** from singular vectors at peaks
3. **SDOF function extraction** around each mode
4. **Inverse FFT** to obtain correlation functions
5. **Damping estimation** from correlation function decay

The correlation function for a SDOF system is:

$$R(\tau) = A e^{-\zeta\omega_n|\tau|} \cos(\omega_d\tau + \phi) \qquad \text{(5.16)}$$

where $\omega_d = \omega_n\sqrt{1-\zeta^2}$ is the damped frequency.

### 5.8.4 Comprehensive Modal Identification Implementation

```python
def frequency_domain_decomposition(response_matrix, fs, frequency_range=(0.1, 50)):
    """
    Perform Frequency Domain Decomposition (FDD) for modal identification
    
    Parameters:
    -----------
    response_matrix : ndarray (n_samples x n_channels)
        Multi-channel response data
    fs : float
        Sampling frequency
    frequency_range : tuple
        Frequency range for analysis
        
    Returns:
    --------
    frequencies : ndarray
        Frequency vector
    singular_values : ndarray
        First singular values at each frequency
    mode_shapes : ndarray
        Mode shapes (singular vectors) at modal frequencies
    modal_frequencies : list
        Identified modal frequencies
    """
    
    n_channels = response_matrix.shape[1]
    
    # Compute cross-PSD matrix
    nperseg = min(4096, response_matrix.shape[0] // 4)
    
    # Initialize PSD matrix
    freq_temp, _ = signal.welch(response_matrix[:, 0], fs, nperseg=nperseg)
    n_freq = len(freq_temp)
    psd_matrix = np.zeros((n_freq, n_channels, n_channels), dtype=complex)
    
    # Compute all auto and cross PSDs
    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                # Auto-PSD
                freq, psd_matrix[:, i, j] = signal.welch(response_matrix[:, i], fs, 
                                                        nperseg=nperseg)
            else:
                # Cross-PSD
                freq, psd_matrix[:, i, j] = signal.csd(response_matrix[:, i], 
                                                      response_matrix[:, j], fs,
                                                      nperseg=nperseg)
    
    # Perform SVD at each frequency
    singular_values = np.zeros((n_freq, n_channels))
    singular_vectors = np.zeros((n_freq, n_channels, n_channels), dtype=complex)
    
    for i in range(n_freq):
        U, S, Vh = np.linalg.svd(psd_matrix[i, :, :])
        singular_values[i, :] = S
        singular_vectors[i, :, :] = U
    
    # Find modal frequencies using peak detection on first singular value
    freq_mask = (freq >= frequency_range[0]) & (freq <= frequency_range[1])
    freq_subset = freq[freq_mask]
    sv1_subset = singular_values[freq_mask, 0]
    
    # Smooth singular values for better peak detection
    sv1_smooth = signal.savgol_filter(sv1_subset, 5, 2)
    
    peaks, properties = find_peaks(sv1_smooth, 
                                  height=np.max(sv1_smooth)*0.1,
                                  distance=len(freq_subset)//50)
    
    modal_frequencies = freq_subset[peaks]
    mode_shapes = []
    
    for peak_idx in peaks:
        freq_idx = np.where(freq_mask)[0][peak_idx]
        mode_shape = singular_vectors[freq_idx, :, 0].real  # First singular vector
        mode_shapes.append(mode_shape)
    
    mode_shapes = np.array(mode_shapes).T
    
    return freq, singular_values[:, 0], mode_shapes, modal_frequencies

def enhanced_fdd_damping_estimation(response_matrix, fs, modal_freq, mode_shape,
                                   freq_bandwidth=0.1):
    """
    Enhanced FDD damping estimation using correlation function analysis
    
    Parameters:
    -----------
    response_matrix : ndarray
        Multi-channel response data
    fs : float
        Sampling frequency  
    modal_freq : float
        Modal frequency to analyze
    mode_shape : ndarray
        Mode shape vector
    freq_bandwidth : float
        Frequency bandwidth around modal frequency
        
    Returns:
    --------
    damping_ratio : float
        Estimated modal damping ratio
    correlation_func : ndarray
        Modal correlation function
    """
    
    # Normalize mode shape
    mode_shape = mode_shape / np.linalg.norm(mode_shape)
    
    # Extract modal coordinate using mode shape
    modal_coordinate = np.dot(response_matrix, mode_shape)
    
    # Compute PSD of modal coordinate
    freq, modal_psd = signal.welch(modal_coordinate, fs, nperseg=4096)
    
    # Extract frequency range around modal frequency
    freq_min = modal_freq - freq_bandwidth
    freq_max = modal_freq + freq_bandwidth
    freq_mask = (freq >= freq_min) & (freq <= freq_max)
    
    # Create SDOF function around modal frequency
    sdof_psd = np.zeros_like(modal_psd)
    sdof_psd[freq_mask] = modal_psd[freq_mask]
    
    # Convert to time domain using inverse FFT
    n_time = len(modal_coordinate)
    correlation_func = np.fft.irfft(sdof_psd, n=n_time)
    
    # Fit exponential decay to correlation function
    dt = 1.0 / fs
    time_vector = np.arange(len(correlation_func)) * dt
    
    # Use envelope of correlation function for damping estimation
    correlation_envelope = np.abs(hilbert(correlation_func))
    
    # Fit exponential decay: A * exp(-zeta * omega_n * t)
    try:
        # Focus on first part of decay for stable fitting
        n_fit = min(1000, len(correlation_envelope) // 4)
        time_fit = time_vector[:n_fit]
        envelope_fit = correlation_envelope[:n_fit]
        
        # Remove very small values to avoid log issues
        valid_mask = envelope_fit > np.max(envelope_fit) * 0.01
        time_fit = time_fit[valid_mask]
        envelope_fit = envelope_fit[valid_mask]
        
        if len(time_fit) > 10:
            # Linear fit in log domain: log(envelope) = log(A) - zeta*omega_n*t
            log_envelope = np.log(envelope_fit)
            coeffs = np.polyfit(time_fit, log_envelope, 1)
            decay_rate = -coeffs[0]  # zeta * omega_n
            
            # Calculate damping ratio
            omega_n = 2 * np.pi * modal_freq
            damping_ratio = decay_rate / omega_n
            
            # Ensure physically reasonable damping
            damping_ratio = max(0.001, min(0.2, damping_ratio))
            
        else:
            damping_ratio = 0.02  # Default assumption
            
    except:
        damping_ratio = 0.02  # Default if fitting fails
    
    return damping_ratio, correlation_func

def generate_multi_channel_bridge_data(n_channels=6, duration=120, fs=100, 
                                     bridge_length=50):
    """
    Generate multi-channel bridge acceleration data with realistic modal characteristics
    
    Parameters:
    -----------
    n_channels : int
        Number of measurement points along bridge
    duration : float
        Signal duration in seconds
    fs : float
        Sampling frequency
    bridge_length : float
        Bridge length in meters
        
    Returns:
    --------
    t : ndarray
        Time vector
    responses : ndarray
        Multi-channel response matrix
    true_modes : dict
        True modal parameters for validation
    """
    
    t = np.linspace(0, duration, int(fs * duration))
    responses = np.zeros((len(t), n_channels))
    
    # Measurement point locations along bridge span
    locations = np.linspace(0, bridge_length, n_channels)
    
    # Define bridge modes with realistic properties
    true_modes = {
        'frequencies': [1.85, 3.42, 5.78, 7.95],
        'damping_ratios': [0.02, 0.025, 0.03, 0.028],
        'mode_shapes': []
    }
    
    # Generate mode shapes for simply supported beam
    for mode_num in range(len(true_modes['frequencies'])):
        # Sine wave mode shapes: sin(n*pi*x/L)
        mode_shape = np.sin((mode_num + 1) * np.pi * locations / bridge_length)
        true_modes['mode_shapes'].append(mode_shape)
        
        freq = true_modes['frequencies'][mode_num]
        damping = true_modes['damping_ratios'][mode_num]
        
        # Generate modal response with ambient excitation
        modal_amplitude = 0.1 / (mode_num + 1)  # Higher modes have lower amplitude
        
        # Random ambient excitation for this mode
        modal_excitation = np.random.normal(0, 1, len(t))
        modal_excitation = signal.filtfilt(*signal.butter(4, freq/10, fs=fs), 
                                          modal_excitation)
        
        # Apply modal response to each channel
        for ch in range(n_channels):
            modal_response = mode_shape[ch] * modal_amplitude * modal_excitation
            
            # Add damping
            damped_response = signal.lfilter(*signal.butter(2, [freq*0.8, freq*1.2], 
                                                           btype='band', fs=fs),
                                           modal_response)
            
            responses[:, ch] += damped_response
    
    # Add uncorrelated noise to each channel
    noise_level = 0.01
    for ch in range(n_channels):
        responses[:, ch] += np.random.normal(0, noise_level, len(t))
    
    return t, responses, true_modes

# Generate multi-channel bridge data
print("Generating multi-channel bridge data for modal identification...")
t_modal, responses_modal, true_modes = generate_multi_channel_bridge_data(
    n_channels=6, duration=120, fs=100)

# Perform FDD modal identification
freq_fdd, sv1, mode_shapes_fdd, modal_freqs_fdd = frequency_domain_decomposition(
    responses_modal, fs=100)

# Estimate damping ratios using EFDD
damping_ratios = []
correlation_functions = []

print("\nPerforming Enhanced FDD damping estimation...")
for i, (modal_freq, mode_shape) in enumerate(zip(modal_freqs_fdd, mode_shapes_fdd.T)):
    damping, correlation = enhanced_fdd_damping_estimation(
        responses_modal, fs=100, modal_freq, mode_shape)
    damping_ratios.append(damping)
    correlation_functions.append(correlation[:2000])  # Keep first 2000 points

# Create comprehensive modal identification visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=['Multi-Channel Bridge Responses', 'FDD Singular Values',
                   'Identified vs True Mode Shapes', 'Modal Damping Estimation',
                   'Mode Shape Comparison (Mode 1)', 'Mode Shape Comparison (Mode 2)'],
    specs=[[{"colspan": 1}, {"colspan": 1}],
           [{"colspan": 1}, {"colspan": 1}],
           [{"colspan": 1}, {"colspan": 1}]]
)

# Multi-channel time series (show subset for clarity)
colors = px.colors.qualitative.Set1
for ch in range(min(4, responses_modal.shape[1])):
    fig.add_trace(go.Scatter(x=t_modal[:2000], y=responses_modal[:2000, ch],
                            name=f'Channel {ch+1}',
                            line=dict(color=colors[ch], width=1)),
                  row=1, col=1)

# FDD Singular values with identified peaks
fig.add_trace(go.Scatter(x=freq_fdd, y=sv1,
                        name='First Singular Value',
                        line=dict(color='blue', width=2)),
              row=1, col=2)

# Mark identified modal frequencies
fig.add_trace(go.Scatter(x=modal_freqs_fdd, 
                        y=[sv1[np.argmin(np.abs(freq_fdd - f))] for f in modal_freqs_fdd],
                        mode='markers', name='Identified Modes',
                        marker=dict(size=10, color='red', symbol='diamond')),
              row=1, col=2)

# Mode shape comparison
n_modes_compare = min(3, len(modal_freqs_fdd), len(true_modes['frequencies']))
locations = np.linspace(0, 50, 6)  # Bridge span locations

for mode_idx in range(n_modes_compare):
    # True mode shapes
    fig.add_trace(go.Scatter(x=locations, y=true_modes['mode_shapes'][mode_idx],
                            name=f'True Mode {mode_idx+1}',
                            line=dict(color=colors[mode_idx], width=3, dash='solid'),
                            mode='lines+markers'),
                  row=2, col=1)
    
    # Identified mode shapes (normalize for comparison)
    identified_shape = mode_shapes_fdd[:, mode_idx]
    identified_shape = identified_shape / np.max(np.abs(identified_shape))
    
    # Match sign convention
    if np.dot(identified_shape, true_modes['mode_shapes'][mode_idx]) < 0:
        identified_shape = -identified_shape
    
    fig.add_trace(go.Scatter(x=locations, y=identified_shape,
                            name=f'FDD Mode {mode_idx+1}',
                            line=dict(color=colors[mode_idx], width=2, dash='dash'),
                            mode='lines+markers'),
                  row=2, col=1)

# Damping estimation visualization (correlation functions)
time_corr = np.arange(len(correlation_functions[0])) / 100  # Time vector for correlation
for i, corr_func in enumerate(correlation_functions[:3]):
    envelope = np.abs(hilbert(corr_func))
    fig.add_trace(go.Scatter(x=time_corr, y=envelope,
                            name=f'Mode {i+1} Envelope (ζ={damping_ratios[i]:.3f})',
                            line=dict(color=colors[i], width=2)),
                  row=2, col=2)

# Detailed mode shape comparisons
if len(modal_freqs_fdd) >= 1:
    # Mode 1 comparison
    fig.add_trace(go.Scatter(x=locations, y=true_modes['mode_shapes'][0],
                            name='True Mode 1',
                            line=dict(color='blue', width=3),
                            mode='lines+markers'),
                  row=3, col=1)
    
    shape_1 = mode_shapes_fdd[:, 0]
    shape_1 = shape_1 / np.max(np.abs(shape_1))
    if np.dot(shape_1, true_modes['mode_shapes'][0]) < 0:
        shape_1 = -shape_1
        
    fig.add_trace(go.Scatter(x=locations, y=shape_1,
                            name='Identified Mode 1',
                            line=dict(color='red', width=2, dash='dash'),
                            mode='lines+markers'),
                  row=3, col=1)

if len(modal_freqs_fdd) >= 2:
    # Mode 2 comparison
    fig.add_trace(go.Scatter(x=locations, y=true_modes['mode_shapes'][1],
                            name='True Mode 2',
                            line=dict(color='green', width=3),
                            mode='lines+markers'),
                  row=3, col=2)
    
    shape_2 = mode_shapes_fdd[:, 1]
    shape_2 = shape_2 / np.max(np.abs(shape_2))
    if np.dot(shape_2, true_modes['mode_shapes'][1]) < 0:
        shape_2 = -shape_2
        
    fig.add_trace(go.Scatter(x=locations, y=shape_2,
                            name='Identified Mode 2',
                            line=dict(color='orange', width=2, dash='dash'),
                            mode='lines+markers'),
                  row=3, col=2)

# Update layout
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
fig.update_xaxes(title_text="Position along Bridge (m)", row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=2)
fig.update_xaxes(title_text="Position along Bridge (m)", row=3, col=1)
fig.update_xaxes(title_text="Position along Bridge (m)", row=3, col=2)

fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
fig.update_yaxes(title_text="Singular Value", type="log", row=1, col=2)
fig.update_yaxes(title_text="Normalized Amplitude", row=2, col=1)
fig.update_yaxes(title_text="Correlation Envelope", type="log", row=2, col=2)
fig.update_yaxes(title_text="Normalized Amplitude", row=3, col=1)
fig.update_yaxes(title_text="Normalized Amplitude", row=3, col=2)

fig.update_layout(height=1000, title_text="Non-Parametric Modal Identification using FDD/EFDD",
                 showlegend=True, template="plotly_white")

fig.show()

# Quantitative assessment of identification accuracy
print("\nModal Identification Results:")
print("="*50)
print(f"{'Mode':<6} {'True f (Hz)':<12} {'FDD f (Hz)':<12} {'Error (%)':<10} {'True ζ':<8} {'EFDD ζ':<8}")
print("-"*50)

for i in range(min(len(modal_freqs_fdd), len(true_modes['frequencies']))):
    true_freq = true_modes['frequencies'][i]
    identified_freq = modal_freqs_fdd[i]
    freq_error = 100 * abs(true_freq - identified_freq) / true_freq
    
    true_damping = true_modes['damping_ratios'][i]
    identified_damping = damping_ratios[i] if i < len(damping_ratios) else 0
    
    print(f"{i+1:<6} {true_freq:<12.3f} {identified_freq:<12.3f} {freq_error:<10.2f} "
          f"{true_damping:<8.3f} {identified_damping:<8.3f}")

# Calculate Modal Assurance Criterion (MAC) for mode shape comparison
def calculate_mac(mode1, mode2):
    """Calculate Modal Assurance Criterion between two mode shapes"""
    numerator = abs(np.dot(mode1, mode2))**2
    denominator = np.dot(mode1, mode1) * np.dot(mode2, mode2)
    return numerator / denominator if denominator > 0 else 0

print(f"\nModal Assurance Criterion (MAC):")
print("-"*30)
for i in range(min(len(modal_freqs_fdd), len(true_modes['mode_shapes']))):
    if i < mode_shapes_fdd.shape[1]:
        identified_shape = mode_shapes_fdd[:, i]
        true_shape = true_modes['mode_shapes'][i]
        
        # Normalize and ensure consistent sign
        identified_shape = identified_shape / np.linalg.norm(identified_shape)
        true_shape = true_shape / np.linalg.norm(true_shape)
        
        if np.dot(identified_shape, true_shape) < 0:
            identified_shape = -identified_shape
            
        mac_value = calculate_mac(true_shape, identified_shape)
        print(f"Mode {i+1}: MAC = {mac_value:.4f}")
```

This comprehensive modal identification implementation demonstrates:

1. **FDD algorithm**: SVD-based decomposition of multi-channel PSD matrix
2. **EFDD enhancement**: Damping estimation through correlation function analysis
3. **Mode shape extraction**: Spatial patterns from singular vectors
4. **Validation metrics**: MAC values and frequency accuracy assessment
5. **Realistic bridge modeling**: Physics-based mode shapes and multi-channel responses

---

## 5.9 Advanced Applications in Bridge SHM

### 5.9.1 Operational Modal Analysis Under Traffic Loading

Modern bridge monitoring systems process vibration data in the frequency domain and use machine learning to detect anomalies, demonstrating the practical implementation of these techniques in real-world infrastructure.

Real bridge monitoring faces unique challenges:
- **Non-stationary excitation**: Traffic loading varies in intensity and frequency content
- **Environmental effects**: Temperature and humidity affect modal parameters
- **Operational conditions**: Different loading patterns throughout the day

### 5.9.2 Integration with Machine Learning

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def extract_frequency_domain_features(signal_data, fs, feature_types='all'):
    """
    Extract comprehensive frequency-domain features for ML applications
    
    Parameters:
    -----------
    signal_data : ndarray
        Input acceleration signal
    fs : float
        Sampling frequency
    feature_types : str or list
        Types of features to extract ('spectral', 'modal', 'statistical', 'all')
        
    Returns:
    --------
    features : dict
        Dictionary containing extracted features
    feature_vector : ndarray
        Concatenated feature vector for ML algorithms
    """
    
    # Compute basic frequency domain representations
    freq, psd = signal.welch(signal_data, fs, nperseg=2048)
    fft_mag = np.abs(np.fft.fft(signal_data))[:len(signal_data)//2]
    freq_fft = np.fft.fftfreq(len(signal_data), 1/fs)[:len(signal_data)//2]
    
    features = {}
    
    if feature_types == 'all' or 'spectral' in feature_types:
        # Spectral features
        features['spectral'] = {
            'peak_frequency': freq[np.argmax(psd)],
            'dominant_freq_amplitude': np.max(psd),
            'spectral_centroid': np.sum(freq * psd) / np.sum(psd),
            'spectral_rolloff': freq[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]],
            'spectral_bandwidth': np.sqrt(np.sum(((freq - features.get('spectral', {}).get('spectral_centroid', 0))**2) * psd) / np.sum(psd)),
            'spectral_flatness': np.exp(np.mean(np.log(psd + 1e-12))) / np.mean(psd),
        }
    
    if feature_types == 'all' or 'modal' in feature_types:
        # Modal-specific features for SHM
        structural_freq_mask = (freq >= 0.5) & (freq <= 20)  # Typical bridge frequency range
        structural_psd = psd[structural_freq_mask]
        structural_freq = freq[structural_freq_mask]
        
        # Find modal peaks
        peaks, _ = find_peaks(structural_psd, height=np.max(structural_psd)*0.1, distance=10)
        modal_freqs = structural_freq[peaks] if len(peaks) > 0 else [0]
        modal_amps = structural_psd[peaks] if len(peaks) > 0 else [0]
        
        features['modal'] = {
            'n_modes': len(peaks),
            'first_mode_freq': modal_freqs[0] if len(modal_freqs) > 0 else 0,
            'first_mode_amplitude': modal_amps[0] if len(modal_amps) > 0 else 0,
            'modal_frequency_ratio': modal_freqs[1]/modal_freqs[0] if len(modal_freqs) > 1 else 0,
            'modal_amplitude_ratio': modal_amps[1]/modal_amps[0] if len(modal_amps) > 1 else 0,
        }
    
    if feature_types == 'all' or 'statistical' in feature_types:
        # Statistical features of frequency content
        features['statistical'] = {
            'psd_mean': np.mean(psd),
            'psd_std': np.std(psd),
            'psd_skewness': np.mean(((psd - np.mean(psd)) / np.std(psd))**3),
            'psd_kurtosis': np.mean(((psd - np.mean(psd)) / np.std(psd))**4),
            'energy_ratio_low': np.sum(psd[freq <= 5]) / np.sum(psd),
            'energy_ratio_high': np.sum(psd[freq >= 10]) / np.sum(psd),
        }
    
    # Concatenate all features into a single vector
    feature_vector = []
    for category in features:
        for feature_name, value in features[category].items():
            feature_vector.append(value)
    
    return features, np.array(feature_vector)

def bridge_health_monitoring_ml_workflow(response_data, fs, time_windows=None):
    """
    Complete ML workflow for bridge health monitoring using frequency-domain features
    
    Parameters:
    -----------
    response_data : ndarray
        Long-term bridge response data
    fs : float
        Sampling frequency
    time_windows : list
        Time windows for analysis (if None, uses sliding windows)
        
    Returns:
    --------
    health_indicators : dict
        Computed health indicators and anomaly detection results
    """
    
    # Sliding window analysis if time_windows not specified
    if time_windows is None:
        window_length = 30 * fs  # 30 second windows
        overlap = window_length // 2
        time_windows = []
        
        for start in range(0, len(response_data) - window_length, overlap):
            end = start + window_length
            time_windows.append((start, end))
    
    # Extract features from each window
    all_features = []
    feature_names = None
    
    print("Extracting frequency-domain features from time windows...")
    for i, (start, end) in enumerate(time_windows):
        window_data = response_data[start:end]
        features, feature_vector = extract_frequency_domain_features(window_data, fs)
        
        if feature_names is None:
            # Store feature names for later reference
            feature_names = []
            for category in features:
                for feature_name in features[category]:
                    feature_names.append(f"{category}_{feature_name}")
        
        all_features.append(feature_vector)
    
    feature_matrix = np.array(all_features)
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(feature_matrix)
    
    # Principal Component Analysis for dimensionality reduction
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    features_pca = pca.fit_transform(features_normalized)
    
    # K-means clustering for normal operation characterization
    n_clusters = 3  # Assume 3 operational states (e.g., light, medium, heavy traffic)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_pca)
    
    # Anomaly detection using distance from cluster centers
    distances_to_centers = []
    for i, point in enumerate(features_pca):
        cluster_center = kmeans.cluster_centers_[cluster_labels[i]]
        distance = np.linalg.norm(point - cluster_center)
        distances_to_centers.append(distance)
    
    distances_to_centers = np.array(distances_to_centers)
    
    # Define anomaly threshold (e.g., 95th percentile)
    anomaly_threshold = np.percentile(distances_to_centers, 95)
    anomalies = distances_to_centers > anomaly_threshold
    
    # Calculate health indicators
    health_indicators = {
        'feature_matrix': feature_matrix,
        'feature_names': feature_names,
        'cluster_labels': cluster_labels,
        'anomaly_flags': anomalies,
        'anomaly_scores': distances_to_centers,
        'anomaly_threshold': anomaly_threshold,
        'pca_components': pca.components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'n_anomalies': np.sum(anomalies),
        'anomaly_percentage': 100 * np.sum(anomalies) / len(anomalies)
    }
    
    return health_indicators

# Generate long-term bridge monitoring data with various operational conditions
def generate_longterm_bridge_data(duration_hours=24, fs=100):
    """Generate 24-hour bridge monitoring data with varying traffic conditions"""
    
    total_samples = int(duration_hours * 3600 * fs)
    t = np.linspace(0, duration_hours * 3600, total_samples)
    acceleration = np.zeros_like(t)
    
    # Simulate daily traffic pattern
    for i, time_val in enumerate(t):
        hour_of_day = (time_val / 3600) % 24
        
        # Traffic intensity varies throughout day
        if 6 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19:
            traffic_intensity = 1.0  # Rush hour
        elif 9 < hour_of_day < 17:
            traffic_intensity = 0.6  # Daytime
        elif 19 < hour_of_day <= 23 or 5 <= hour_of_day < 6:
            traffic_intensity = 0.3  # Evening/early morning
        else:
            traffic_intensity = 0.1  # Night time
        
        # Base modal response (first mode)
        base_freq = 1.85
        modal_amplitude = 0.05 * traffic_intensity
        acceleration[i] += modal_amplitude * np.sin(2 * np.pi * base_freq * time_val)
        
        # Second mode
        second_freq = 3.42
        acceleration[i] += 0.02 * traffic_intensity * np.sin(2 * np.pi * second_freq * time_val)
        
        # Add some damage progression (very gradual frequency reduction)
        damage_factor = 1 - 0.001 * (time_val / (3600 * 24))  # 0.1% reduction over 24 hours
        if hour_of_day > 12:  # Damage becomes apparent after noon
            acceleration[i] *= damage_factor
        
        # Random vehicle events
        if np.random.random() < traffic_intensity * 0.001:  # Higher probability during high traffic
            vehicle_impact = 0.2 * np.random.random() * np.sin(2 * np.pi * 25 * time_val)
            acceleration[i] += vehicle_impact
    
    # Add ambient noise
    acceleration += np.random.normal(0, 0.01, len(t))
    
    return t, acceleration

# Generate and analyze long-term monitoring data
print("Generating 24-hour bridge monitoring data...")
t_longterm, accel_longterm = generate_longterm_bridge_data(duration_hours=6, fs=100)  # 6 hours for demo

# Perform ML-based health monitoring analysis
health_results = bridge_health_monitoring_ml_workflow(accel_longterm, fs=100)

# Create comprehensive ML analysis visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=['Long-term Bridge Response', 'Feature Evolution (PCA)',
                   'Cluster Analysis', 'Anomaly Detection',
                   'Feature Importance', 'Health Indicator Timeline'],
    specs=[[{"colspan": 2}, None],
           [{"colspan": 1}, {"colspan": 1}],
           [{"colspan": 1}, {"colspan": 1}]]
)

# Time series plot (show subset for clarity)
plot_duration = 3600 * 2  # 2 hours
plot_samples = int(plot_duration * 100)
time_hours = t_longterm[:plot_samples] / 3600

fig.add_trace(go.Scatter(x=time_hours, y=accel_longterm[:plot_samples],
                        name='Bridge Response',
                        line=dict(color='#2E86AB', width=1)),
              row=1, col=1)

# PCA feature space
pca_features = PCA(n_components=2).fit_transform(
    StandardScaler().fit_transform(health_results['feature_matrix']))

colors_cluster = ['red', 'blue', 'green', 'orange', 'purple']
for cluster_id in range(max(health_results['cluster_labels']) + 1):
    mask = health_results['cluster_labels'] == cluster_id
    fig.add_trace(go.Scatter(x=pca_features[mask, 0], y=pca_features[mask, 1],
                            mode='markers', name=f'Cluster {cluster_id+1}',
                            marker=dict(color=colors_cluster[cluster_id], size=8)),
                  row=2, col=1)

# Cluster centers
kmeans_2d = KMeans(n_clusters=3, random_state=42).fit(pca_features)
fig.add_trace(go.Scatter(x=kmeans_2d.cluster_centers_[:, 0], 
                        y=kmeans_2d.cluster_centers_[:, 1],
                        mode='markers', name='Cluster Centers',
                        marker=dict(color='black', size=15, symbol='x')),
              row=2, col=1)

# Anomaly detection plot
window_times = np.arange(len(health_results['anomaly_scores'])) * 0.25  # 15-min windows
normal_mask = ~health_results['anomaly_flags']
anomaly_mask = health_results['anomaly_flags']

fig.add_trace(go.Scatter(x=window_times[normal_mask], 
                        y=health_results['anomaly_scores'][normal_mask],
                        mode='markers', name='Normal Operation',
                        marker=dict(color='blue', size=6)),
              row=2, col=2)

fig.add_trace(go.Scatter(x=window_times[anomaly_mask], 
                        y=health_results['anomaly_scores'][anomaly_mask],
                        mode='markers', name='Anomalies',
                        marker=dict(color='red', size=8)),
              row=2, col=2)

# Threshold line
fig.add_hline(y=health_results['anomaly_threshold'], line_dash="dash", 
              line_color="red", annotation_text="Anomaly Threshold", row=2, col=2)

# Feature importance (based on PCA loadings)
feature_importance = np.abs(health_results['pca_components'][0])  # First PC
top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features

fig.add_trace(go.Bar(x=[health_results['feature_names'][i] for i in top_features_idx],
                    y=feature_importance[top_features_idx],
                    name='Feature Importance',
                    marker_color='green'),
              row=3, col=1)

# Health indicator timeline
moving_avg_window = 10
if len(health_results['anomaly_scores']) >= moving_avg_window:
    health_indicator = pd.Series(health_results['anomaly_scores']).rolling(
        window=moving_avg_window).mean().values
    
    fig.add_trace(go.Scatter(x=window_times, y=health_indicator,
                            name='Health Indicator (Moving Avg)',
                            line=dict(color='purple', width=3)),
                  row=3, col=2)
    
    fig.add_trace(go.Scatter(x=window_times, y=health_results['anomaly_scores'],
                            name='Raw Anomaly Score',
                            line=dict(color='lightgray', width=1)),
                  row=3, col=2)

# Update layout
fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
fig.update_xaxes(title_text="PC1", row=2, col=1)
fig.update_xaxes(title_text="Time (hours)", row=2, col=2)
fig.update_xaxes(title_text="Features", row=3, col=1)
fig.update_xaxes(title_text="Time (hours)", row=3, col=2)

fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
fig.update_yaxes(title_text="PC2", row=2, col=1)
fig.update_yaxes(title_text="Anomaly Score", row=2, col=2)
fig.update_yaxes(title_text="Importance", row=3, col=1)
fig.update_yaxes(title_text="Health Indicator", row=3, col=2)

# Rotate x-axis labels for feature names
fig.update_layout(height=1000, 
                 title_text="Machine Learning-Based Bridge Health Monitoring",
                 showlegend=True, template="plotly_white")
fig.update_xaxes(tickangle=45, row=3, col=1)

fig.show()

# Summary statistics
print("\nMachine Learning Analysis Results:")
print("="*40)
print(f"Total analysis windows: {len(health_results['anomaly_scores'])}")
print(f"Detected anomalies: {health_results['n_anomalies']}")
print(f"Anomaly rate: {health_results['anomaly_percentage']:.2f}%")
print(f"PCA explained variance (first 3 components): {health_results['explained_variance_ratio'][:3]}")

# Feature ranking
feature_ranking = np.argsort(np.abs(health_results['pca_components'][0]))[::-1]
print(f"\nTop 5 most important features:")
for i in range(5):
    feature_idx = feature_ranking[i]
    feature_name = health_results['feature_names'][feature_idx]
    importance = np.abs(health_results['pca_components'][0][feature_idx])
    print(f"{i+1}. {feature_name}: {importance:.3f}")
```

This advanced implementation demonstrates the integration of frequency-domain analysis with machine learning for practical bridge health monitoring applications.

---

## 5.10 Summary and Key Takeaways

Frequency-domain analysis provides the foundation for modern structural health monitoring systems, offering robust and sensitive techniques for extracting meaningful information from bridge response data. This chapter has covered the essential mathematical transforms and their practical implementations:

**Core Techniques:**
- **FFT Analysis**: Fundamental tool for converting time-domain signals to frequency domain
- **Power Spectral Density**: Gold standard for random vibration analysis and modal identification
- **STFT**: Time-frequency analysis for non-stationary signals and event detection
- **Wavelet Transform**: Multi-resolution analysis for transient detection and damage progression monitoring
- **Non-parametric Modal ID**: FDD and EFDD methods for operational modal analysis

**Key Applications in Bridge SHM:**
- Modal parameter identification (frequencies, mode shapes, damping)
- Damage detection through frequency changes
- Operational condition monitoring under varying traffic loads
- Integration with machine learning for automated health assessment

**Practical Considerations:**
- Proper windowing and averaging techniques reduce spectral leakage and noise
- PSDT provides operational invariant features less sensitive to loading conditions
- Multi-resolution approaches capture both global structural behavior and local damage events
- Feature extraction enables automated monitoring and anomaly detection

---

## 5.11 Exercises

### Exercise 5.1: FFT Implementation and Analysis

**Problem:** A bridge monitoring system records acceleration data at 200 Hz sampling rate. Implement an FFT analysis to identify the first three modal frequencies from a 60-second recording.

**Tasks:**
a) Generate realistic bridge acceleration data with three modes at 1.2 Hz, 2.8 Hz, and 4.5 Hz
b) Apply appropriate windowing and implement FFT analysis
c) Identify modal frequencies using peak detection
d) Compare results with different window functions (Hanning, Hamming, Blackman)
e) Analyze the effect of zero-padding on frequency resolution

### Exercise 5.2: Power Spectral Density Transmissibility

**Problem:** Calculate PSDT between two measurement points on a bridge to create a damage-sensitive feature that is less affected by traffic loading variations.

**Tasks:**
a) Generate multi-point bridge data with realistic modal characteristics
b) Implement PSDT calculation between reference and response points
c) Simulate damage by reducing stiffness (frequency reduction) 
d) Compare sensitivity of PSDT vs regular PSD to damage
e) Analyze robustness to varying excitation levels

### Exercise 5.3: STFT Analysis of Vehicle Crossing Events

**Problem:** Use STFT to analyze the time-frequency characteristics of bridge response during vehicle crossing events.

**Tasks:**
a) Generate bridge data during vehicle crossing with time-varying modal excitation
b) Implement STFT with different time-frequency resolution settings
c) Track modal frequency evolution during the crossing event
d) Detect transient events (e.g., vehicles hitting expansion joints)
e) Quantify frequency variations and their relationship to vehicle speed

### Exercise 5.4: Wavelet-Based Damage Detection

**Problem:** Implement continuous wavelet transform for detecting sudden damage events in bridge structures.

**Tasks:**
a) Generate bridge data with gradual damage progression and sudden damage events
b) Apply CWT using appropriate mother wavelets (Morlet, Mexican hat)
c) Develop automated damage event detection algorithm
d) Compare wavelet analysis with traditional FFT approach
e) Assess sensitivity to different types of damage (gradual vs sudden)

### Exercise 5.5: Non-Parametric Modal Identification

**Problem:** Implement Enhanced Frequency Domain Decomposition (EFDD) for complete modal identification of a multi-span bridge.

**Tasks:**
a) Generate multi-channel bridge data with realistic mode shapes
b) Implement FDD algorithm using SVD of PSD matrix
c) Extract modal frequencies, mode shapes, and damping ratios
d) Validate results against known modal parameters
e) Calculate Modal Assurance Criterion (MAC) for mode shape accuracy

---

## 5.12 Exercise Solutions

### Solution 5.1: FFT Implementation and Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def solution_5_1():
    """Solution for Exercise 5.1: FFT Implementation and Analysis"""
    
    # Parameters
    fs = 200  # Hz
    duration = 60  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # a) Generate realistic bridge data with three modes
    modal_freqs = [1.2, 2.8, 4.5]  # Hz
    modal_amps = [0.1, 0.06, 0.03]  # m/s²
    modal_damping = [0.02, 0.025, 0.03]
    
    acceleration = np.zeros_like(t)
    for i, (freq, amp, damp) in enumerate(zip(modal_freqs, modal_amps, modal_damping)):
        # Damped sinusoidal response with random phase
        phase = np.random.uniform(0, 2*np.pi)
        modal_response = amp * np.exp(-damp * 2 * np.pi * freq * t) * \
                        np.sin(2 * np.pi * freq * t + phase)
        acceleration += modal_response
    
    # Add noise
    acceleration += np.random.normal(0, 0.01, len(t))
    
    # b) Apply different window functions and FFT analysis
    windows = ['hann', 'hamming', 'blackman', 'boxcar']
    nperseg = 8192
    
    results = {}
    for window in windows:
        freq, psd = signal.welch(acceleration, fs, window=window, nperseg=nperseg)
        
        # c) Peak detection for modal identification
        peaks, properties = find_peaks(psd, height=np.max(psd)*0.1, distance=20)
        identified_freqs = freq[peaks]
        
        results[window] = {
            'freq': freq,
            'psd': psd,
            'peaks': peaks,
            'identified_freqs': identified_freqs[:3]  # First 3 modes
        }
    
    # d) Compare window functions
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=[f'{w.capitalize()} Window' for w in windows])
    
    positions = [(1,1), (1,2), (2,1), (2,2)]
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (window, color) in enumerate(zip(windows, colors)):
        row, col = positions[i]
        fig.add_trace(go.Scatter(x=results[window]['freq'], y=results[window]['psd'],
                                name=f'{window} PSD', line=dict(color=color)),
                     row=row, col=col)
        
        # Mark identified peaks
        peak_freqs = results[window]['freq'][results[window]['peaks']]
        peak_psds = results[window]['psd'][results[window]['peaks']]
        fig.add_trace(go.Scatter(x=peak_freqs, y=peak_psds, mode='markers',
                                name=f'{window} Peaks', 
                                marker=dict(color=color, size=10, symbol='diamond')),
                     row=row, col=col)
    
    fig.update_layout(height=600, title_text="Window Function Comparison")
    fig.show()
    
    # e) Zero-padding analysis
    zero_pad_factors = [1, 2, 4, 8]
    fig_zp = go.Figure()
    
    for factor in zero_pad_factors:
        padded_signal = np.pad(acceleration, (0, len(acceleration)*(factor-1)), 'constant')
        freq_zp = np.fft.fftfreq(len(padded_signal), 1/fs)[:len(padded_signal)//2]
        fft_zp = np.abs(np.fft.fft(padded_signal))[:len(padded_signal)//2]
        
        fig_zp.add_trace(go.Scatter(x=freq_zp, y=fft_zp, 
                                   name=f'Zero-pad factor {factor}',
                                   line=dict(width=2)))
    
    fig_zp.update_layout(title="Effect of Zero-Padding on Frequency Resolution",
                        xaxis_title="Frequency (Hz)", yaxis_title="FFT Magnitude")
    fig_zp.show()
    
    # Results summary
    print("Exercise 5.1 Results:")
    print("="*30)
    print(f"True modal frequencies: {modal_freqs} Hz")
    
    for window in windows:
        identified = results[window]['identified_freqs']
        errors = [abs(true - id) for true, id in zip(modal_freqs[:len(identified)], identified)]
        print(f"{window.capitalize()} window identified: {identified} Hz")
        print(f"  Errors: {errors} Hz")

# Run solution
solution_5_1()
```

### Solution 5.2: Power Spectral Density Transmissibility

```python
def solution_5_2():
    """Solution for Exercise 5.2: PSDT Analysis"""
    
    # Generate multi-point bridge data
    fs = 100
    duration = 120
    t = np.linspace(0, duration, int(fs * duration))
    n_points = 5
    
    # Bridge parameters
    bridge_modes = [(1.8, 0.02), (3.4, 0.025), (5.2, 0.03)]
    responses = np.zeros((len(t), n_points))
    
    # Generate spatial mode shapes
    locations = np.linspace(0, 1, n_points)
    
    for mode_idx, (freq, damping) in enumerate(bridge_modes):
        mode_shape = np.sin((mode_idx + 1) * np.pi * locations)
        
        # Time-varying excitation
        excitation = np.random.normal(0, 1, len(t))
        excitation = signal.filtfilt(*signal.butter(4, freq/5, fs=fs), excitation)
        
        # Apply to each point
        for point in range(n_points):
            modal_amplitude = 0.1 / (mode_idx + 1)
            responses[:, point] += mode_shape[point] * modal_amplitude * excitation
    
    # Add noise
    for point in range(n_points):
        responses[:, point] += np.random.normal(0, 0.01, len(t))
    
    # Simulate damage (reduce frequency of first mode by 5%)
    damaged_responses = responses.copy()
    damage_factor = 0.95
    
    for mode_idx, (freq, damping) in enumerate(bridge_modes[:1]):  # Only first mode
        if mode_idx == 0:  # First mode only
            mode_shape = np.sin((mode_idx + 1) * np.pi * locations)
            
            # Damaged excitation with reduced frequency
            damaged_freq = freq * damage_factor
            excitation = np.random.normal(0, 1, len(t))
            excitation = signal.filtfilt(*signal.butter(4, damaged_freq/5, fs=fs), excitation)
            
            # Replace first mode response
            for point in range(n_points):
                modal_amplitude = 0.1
                # Subtract original first mode
                orig_shape = np.sin(np.pi * locations)
                orig_excitation = np.random.normal(0, 1, len(t))
                orig_excitation = signal.filtfilt(*signal.butter(4, freq/5, fs=fs), orig_excitation)
                
                # Add damaged mode
                damaged_responses[:, point] += mode_shape[point] * modal_amplitude * \
                                             (excitation - orig_excitation)
    
    # Calculate PSDT for healthy and damaged states
    reference_point = 2  # Middle point as reference
    
    def calculate_psdt_comparison(responses_healthy, responses_damaged, ref_idx):
        freq, ref_psd_h = signal.welch(responses_healthy[:, ref_idx], fs)
        freq, ref_psd_d = signal.welch(responses_damaged[:, ref_idx], fs)
        
        psdt_healthy = []
        psdt_damaged = []
        psd_healthy = []
        psd_damaged = []
        
        for point in range(n_points):
            # Healthy state
            _, cross_psd_h = signal.csd(responses_healthy[:, ref_idx], 
                                       responses_healthy[:, point], fs)
            psdt_h = cross_psd_h / (ref_psd_h + 1e-12)
            psdt_healthy.append(np.abs(psdt_h))
            
            _, psd_h = signal.welch(responses_healthy[:, point], fs)
            psd_healthy.append(psd_h)
            
            # Damaged state
            _, cross_psd_d = signal.csd(responses_damaged[:, ref_idx], 
                                       responses_damaged[:, point], fs)
            psdt_d = cross_psd_d / (ref_psd_d + 1e-12)
            psdt_damaged.append(np.abs(psdt_d))
            
            _, psd_d = signal.welch(responses_damaged[:, point], fs)
            psd_damaged.append(psd_d)
        
        return freq, psdt_healthy, psdt_damaged, psd_healthy, psd_damaged
    
    freq, psdt_h, psdt_d, psd_h, psd_d = calculate_psdt_comparison(
        responses, damaged_responses, reference_point)
    
    # Visualization
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=['PSDT Comparison', 'PSD Comparison',
                                      'PSDT Damage Sensitivity', 'PSD Damage Sensitivity'])
    
    colors = px.colors.qualitative.Set1
    
    # PSDT comparison
    for point in range(n_points):
        if point != reference_point:
            fig.add_trace(go.Scatter(x=freq, y=psdt_h[point], 
                                    name=f'PSDT Point {point+1} (Healthy)',
                                    line=dict(color=colors[point], width=2)),
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=freq, y=psdt_d[point], 
                                    name=f'PSDT Point {point+1} (Damaged)',
                                    line=dict(color=colors[point], width=2, dash='dash')),
                         row=1, col=1)
    
    # PSD comparison
    for point in range(n_points):
        fig.add_trace(go.Scatter(x=freq, y=psd_h[point], 
                                name=f'PSD Point {point+1} (Healthy)',
                                line=dict(color=colors[point], width=2)),
                     row=1, col=2)
        fig.add_trace(go.Scatter(x=freq, y=psd_d[point], 
                                name=f'PSD Point {point+1} (Damaged)',
                                line=dict(color=colors[point], width=2, dash='dash')),
                     row=1, col=2)
    
    # Damage sensitivity analysis
    target_point = 1  # Analyze sensitivity at this point
    psdt_change = (psdt_d[target_point] - psdt_h[target_point]) / psdt_h[target_point] * 100
    psd_change = (psd_d[target_point] - psd_h[target_point]) / psd_h[target_point] * 100
    
    fig.add_trace(go.Scatter(x=freq, y=psdt_change, name='PSDT % Change',
                            line=dict(color='red', width=3)),
                 row=2, col=1)
    
    fig.add_trace(go.Scatter(x=freq, y=psd_change, name='PSD % Change',
                            line=dict(color='blue', width=3)),
                 row=2, col=2)
    
    fig.update_layout(height=800, title_text="PSDT vs PSD Damage Sensitivity Analysis")
    fig.show()
    
    # Quantitative analysis
    first_mode_freq_idx = np.argmin(np.abs(freq - 1.8))
    psdt_sensitivity = np.abs(psdt_change[first_mode_freq_idx])
    psd_sensitivity = np.abs(psd_change[first_mode_freq_idx])
    
    print("Exercise 5.2 Results:")
    print("="*30)
    print(f"Damage applied: {(1-damage_factor)*100:.1f}% frequency reduction")
    print(f"PSDT sensitivity at first mode: {psdt_sensitivity:.2f}%")
    print(f"PSD sensitivity at first mode: {psd_sensitivity:.2f}%")
    print(f"PSDT advantage factor: {psdt_sensitivity/psd_sensitivity:.2f}x")

solution_5_2()
```

### Solution 5.3: STFT Analysis of Vehicle Crossing Events

```python
def solution_5_3():
    """Solution for Exercise 5.3: STFT Vehicle Crossing Analysis"""
    
    # Generate vehicle crossing event
    fs = 100
    duration = 40  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Vehicle parameters
    vehicle_speed = 20  # m/s
    bridge_length = 60  # m
    crossing_duration = bridge_length / vehicle_speed
    crossing_start = 10  # seconds
    crossing_end = crossing_start + crossing_duration
    
    acceleration = np.zeros_like(t)
    
    # Bridge modal properties
    base_freq1 = 1.8  # Hz
    base_freq2 = 3.5  # Hz
    
    for i, time_val in enumerate(t):
        if crossing_start <= time_val <= crossing_end:
            # Vehicle on bridge - time-varying modal response
            position = (time_val - crossing_start) / crossing_duration
            
            # First mode with position-dependent amplitude and frequency variation
            amplitude1 = 0.15 * np.sin(np.pi * position)  # Max at mid-span
            freq_variation1 = base_freq1 * (1 + 0.02 * np.sin(2 * np.pi * position))
            acceleration[i] += amplitude1 * np.sin(2 * np.pi * freq_variation1 * time_val)
            
            # Second mode
            amplitude2 = 0.08 * np.sin(2 * np.pi * position)
            freq_variation2 = base_freq2 * (1 + 0.015 * np.cos(np.pi * position))
            acceleration[i] += amplitude2 * np.sin(2 * np.pi * freq_variation2 * time_val)
            
        else:
            # Ambient vibration
            acceleration[i] += 0.02 * np.sin(2 * np.pi * base_freq1 * time_val)
            acceleration[i] += 0.01 * np.sin(2 * np.pi * base_freq2 * time_val)
    
    # Add vehicle impact events (expansion joint hits)
    impact_times = [crossing_start - 0.5, crossing_end + 0.5]  # Just before/after crossing
    for impact_time in impact_times:
        impact_idx = int(impact_time * fs)
        if 0 <= impact_idx < len(acceleration):
            # High-frequency transient
            for j in range(int(0.2 * fs)):  # 0.2 second duration
                if impact_idx + j < len(acceleration):
                    decay = np.exp(-j / (0.05 * fs))
                    acceleration[impact_idx + j] += 0.3 * decay * \
                        np.sin(2 * np.pi * 25 * t[impact_idx + j])
    
    # Add noise
    acceleration += np.random.normal(0, 0.005, len(t))
    
    # STFT analysis with different resolutions
    stft_configs = [
        {'nperseg': 256, 'name': 'High Time Resolution'},
        {'nperseg': 1024, 'name': 'High Frequency Resolution'},
        {'nperseg': 512, 'name': 'Balanced Resolution'}
    ]
    
    stft_results = {}
    for config in stft_configs:
        freq, time_stft, stft_complex = signal.stft(
            acceleration, fs, nperseg=config['nperseg'], noverlap=config['nperseg']//2)
        stft_results[config['name']] = {
            'freq': freq,
            'time': time_stft,
            'magnitude': np.abs(stft_complex)
        }
    
    # Modal frequency tracking
    def track_modal_frequencies(freq, stft_mag, target_freqs):
        tracked_freqs = {f'mode_{i+1}': [] for i in range(len(target_freqs))}
        
        for time_idx in range(stft_mag.shape[1]):
            for i, target_freq in enumerate(target_freqs):
                # Find frequency range around target
                freq_range = 0.3  # Hz
                freq_mask = (freq >= target_freq - freq_range) & (freq <= target_freq + freq_range)
                
                if np.any(freq_mask):
                    freq_subset = freq[freq_mask]
                    mag_subset = stft_mag[freq_mask, time_idx]
                    
                    # Find peak in this range
                    if len(mag_subset) > 0:
                        peak_idx = np.argmax(mag_subset)
                        tracked_freqs[f'mode_{i+1}'].append(freq_subset[peak_idx])
                    else:
                        tracked_freqs[f'mode_{i+1}'].append(target_freq)
                else:
                    tracked_freqs[f'mode_{i+1}'].append(target_freq)
        
        return tracked_freqs
    
    # Track frequencies using balanced resolution STFT
    balanced_result = stft_results['Balanced Resolution']
    tracked_frequencies = track_modal_frequencies(
        balanced_result['freq'], balanced_result['magnitude'], [base_freq1, base_freq2])
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Vehicle Crossing Signal', 'High Time Resolution STFT',
                       'High Frequency Resolution STFT', 'Modal Frequency Tracking',
                       'Impact Event Detection', 'Frequency Variation Analysis'],
        specs=[[{"colspan": 2}, None],
               [{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}]]
    )
    
    # Time domain signal
    fig.add_trace(go.Scatter(x=t, y=acceleration, name='Bridge Response',
                            line=dict(color='blue', width=1.5)),
                 row=1, col=1)
    
    # Mark crossing period
    fig.add_vrect(x0=crossing_start, x1=crossing_end, fillcolor="red", 
                  opacity=0.2, annotation_text="Vehicle Crossing", row=1, col=1)
    
    # STFT spectrograms
    fig.add_trace(go.Heatmap(
        x=stft_results['High Time Resolution']['time'],
        y=stft_results['High Time Resolution']['freq'],
        z=20*np.log10(stft_results['High Time Resolution']['magnitude'] + 1e-12),
        colorscale='Viridis', name='High Time Res',
        colorbar=dict(title="Magnitude (dB)", x=0.48)),
        row=2, col=1)
    
    fig.add_trace(go.Heatmap(
        x=stft_results['High Frequency Resolution']['time'],
        y=stft_results['High Frequency Resolution']['freq'],
        z=20*np.log10(stft_results['High Frequency Resolution']['magnitude'] + 1e-12),
        colorscale='Plasma', name='High Freq Res',
        colorbar=dict(title="Magnitude (dB)", x=0.98)),
        row=2, col=2)
    
    # Modal frequency tracking
    time_track = balanced_result['time']
    fig.add_trace(go.Scatter(x=time_track, y=tracked_frequencies['mode_1'],
                            name='First Mode', line=dict(color='red', width=3)),
                 row=3, col=1)
    fig.add_trace(go.Scatter(x=time_track, y=tracked_frequencies['mode_2'],
                            name='Second Mode', line=dict(color='blue', width=3)),
                 row=3, col=1)
    
    # Add baseline frequencies
    fig.add_hline(y=base_freq1, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=base_freq2, line_dash="dash", line_color="blue", row=3, col=1)
    
    # Impact detection (high frequency content)
    impact_freq_range = (20, 30)  # Hz
    freq_mask = (balanced_result['freq'] >= impact_freq_range[0]) & \
                (balanced_result['freq'] <= impact_freq_range[1])
    impact_energy = np.sum(balanced_result['magnitude'][freq_mask, :], axis=0)
    
    fig.add_trace(go.Scatter(x=time_track, y=impact_energy,
                            name='Impact Energy (20-30 Hz)',
                            line=dict(color='orange', width=2)),
                 row=3, col=2)
    
    # Update layout
    fig.update_layout(height=1000, title_text="STFT Analysis of Vehicle Crossing Event")
    fig.show()
    
    # Quantitative analysis
    crossing_mask = (time_track >= crossing_start) & (time_track <= crossing_end)
    
    mode1_baseline = np.mean([f for f in tracked_frequencies['mode_1'] if f != base_freq1])
    mode1_crossing = np.array(tracked_frequencies['mode_1'])[crossing_mask]
    mode1_variation = np.max(mode1_crossing) - np.min(mode1_crossing)
    
    print("Exercise 5.3 Results:")
    print("="*35)
    print(f"Vehicle crossing duration: {crossing_duration:.1f} seconds")
    print(f"First mode baseline frequency: {mode1_baseline:.3f} Hz")
    print(f"Maximum frequency variation during crossing: {mode1_variation:.3f} Hz")
    print(f"Relative variation: {100*mode1_variation/base_freq1:.2f}%")
    
    # Impact detection summary
    impact_threshold = np.mean(impact_energy) + 2*np.std(impact_energy)
    impact_events = time_track[impact_energy > impact_threshold]
    print(f"Detected impact events at times: {impact_events}")

solution_5_3()
```

### Solution 5.4: Wavelet-Based Damage Detection

```python
def solution_5_4():
    """Solution for Exercise 5.4: Wavelet Damage Detection"""
    
    import pywt
    from scipy.signal import hilbert
    
    # Generate bridge data with damage progression
    fs = 100
    duration = 300  # 5 minutes
    t = np.linspace(0, duration, int(fs * duration))
    
    # Damage scenario: gradual degradation + sudden events
    initial_freq = 2.0  # Hz
    degradation_rate = 0.0002  # per second
    
    acceleration = np.zeros_like(t)
    instantaneous_freq = np.zeros_like(t)
    
    # Sudden damage events at specific times
    damage_events = [120, 240]  # seconds
    damage_magnitudes = [0.05, 0.08]  # frequency drops
    
    cumulative_damage = 0
    
    for i, time_val in enumerate(t):
        # Check for sudden damage events
        for event_time, damage_mag in zip(damage_events, damage_magnitudes):
            if abs(time_val - event_time) < 0.01:  # Within 0.01 seconds
                cumulative_damage += damage_mag
        
        # Calculate current frequency with gradual and sudden damage
        gradual_damage = degradation_rate * time_val
        current_freq = initial_freq * (1 - gradual_damage - cumulative_damage)
        instantaneous_freq[i] = current_freq
        
        # Generate modal response
        acceleration[i] = 0.1 * np.sin(2 * np.pi * current_freq * time_val)
        
        # Add sudden damage transients
        for event_time in damage_events:
            if abs(time_val - event_time) < 2.0:  # 2-second transient
                transient_decay = np.exp(-abs(time_val - event_time) / 0.5)
                transient_freq = 15  # High frequency transient
                acceleration[i] += 0.2 * transient_decay * \
                    np.sin(2 * np.pi * transient_freq * time_val)
    
    # Add noise
    acceleration += np.random.normal(0, 0.01, len(t))
    
    # Continuous Wavelet Transform analysis
    scales = np.arange(1, 128)
    wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet
    
    # Compute CWT
    cwt_coefficients, frequencies = pywt.cwt(acceleration, scales, wavelet, 1/fs)
    
    # Energy analysis for damage detection
    cwt_energy = np.abs(cwt_coefficients)**2
    
    # Focus on structural frequency range (0.5-10 Hz)
    struct_freq_mask = (frequencies >= 0.5) & (frequencies <= 10)
    struct_cwt_energy = cwt_energy[struct_freq_mask, :]
    struct_frequencies = frequencies[struct_freq_mask]
    
    # Damage event detection algorithm
    def detect_damage_events_wavelet(energy, frequencies, times, threshold_factor=3):
        # Calculate baseline energy statistics
        baseline_window = int(30 * fs)  # First 30 seconds
        baseline_energy = energy[:, :baseline_window]
        baseline_mean = np.mean(baseline_energy, axis=1, keepdims=True)
        baseline_std = np.std(baseline_energy, axis=1, keepdims=True)
        
        # Anomaly detection threshold
        threshold = baseline_mean + threshold_factor * baseline_std
        
        # Find anomalous events
        anomalies = energy > threshold
        
        # Group anomalies into events
        events = []
        for time_idx in range(energy.shape[1]):
            if np.any(anomalies[:, time_idx]):
                # Find dominant frequency of anomaly
                anomaly_freqs = frequencies[anomalies[:, time_idx]]
                anomaly_energies = energy[anomalies[:, time_idx], time_idx]
                
                if len(anomaly_freqs) > 0:
                    dominant_freq_idx = np.argmax(anomaly_energies)
                    events.append({
                        'time': times[time_idx],
                        'frequency': anomaly_freqs[dominant_freq_idx],
                        'energy': anomaly_energies[dominant_freq_idx],
                        'severity': np.max(anomaly_energies) / np.max(baseline_mean)
                    })
        
        return events
    
    detected_events = detect_damage_events_wavelet(
        struct_cwt_energy, struct_frequencies, t)
    
    # Ridge extraction for frequency tracking
    def extract_frequency_ridge(cwt_coefficients, frequencies, target_freq=2.0):
        ridge_freqs = []
        ridge_energies = []
        
        for time_idx in range(cwt_coefficients.shape[1]):
            # Find frequency closest to target
            freq_idx = np.argmin(np.abs(frequencies - target_freq))
            
            # Search in neighborhood for peak
            search_range = 5
            start_idx = max(0, freq_idx - search_range)
            end_idx = min(len(frequencies), freq_idx + search_range + 1)
            
            local_energies = np.abs(cwt_coefficients[start_idx:end_idx, time_idx])**2
            peak_idx = np.argmax(local_energies)
            
            ridge_freqs.append(frequencies[start_idx + peak_idx])
            ridge_energies.append(local_energies[peak_idx])
        
        return np.array(ridge_freqs), np.array(ridge_energies)
    
    tracked_freq, tracked_energy = extract_frequency_ridge(
        cwt_coefficients[struct_freq_mask, :], struct_frequencies)
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Signal with Damage Events', 'Wavelet Scalogram',
                       'Frequency Tracking', 'Damage Event Detection',
                       'Energy Distribution Analysis', 'Damage Progression'],
        specs=[[{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}]]
    )
    
    # Time domain signal
    fig.add_trace(go.Scatter(x=t, y=acceleration, name='Acceleration',
                            line=dict(color='blue', width=1)),
                 row=1, col=1)
    
    # Mark damage events
    for event_time in damage_events:
        fig.add_vline(x=event_time, line_dash="dash", line_color="red",
                      annotation_text="Damage Event", row=1, col=1)
    
    # Wavelet scalogram
    fig.add_trace(go.Heatmap(x=t, y=frequencies, 
                            z=20*np.log10(np.abs(cwt_coefficients) + 1e-12),
                            colorscale='Viridis', name='CWT Magnitude',
                            colorbar=dict(title="Magnitude (dB)")),
                 row=1, col=2)
    
    # Frequency tracking
    fig.add_trace(go.Scatter(x=t, y=tracked_freq, name='Tracked Frequency',
                            line=dict(color='red', width=3)),
                 row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=instantaneous_freq, name='True Frequency',
                            line=dict(color='black', width=2, dash='dash')),
                 row=2, col=1)
    
    # Detected events
    if detected_events:
        event_times = [e['time'] for e in detected_events[:50]]  # Limit for clarity
        event_freqs = [e['frequency'] for e in detected_events[:50]]
        event_severities = [e['severity'] for e in detected_events[:50]]
        
        fig.add_trace(go.Scatter(x=event_times, y=event_freqs, mode='markers',
                                name='Detected Events',
                                marker=dict(size=[min(20, s*2) for s in event_severities],
                                          color=event_severities,
                                          colorscale='Reds')),
                     row=2, col=2)
    
    # Energy distribution
    total_energy_time = np.sum(struct_cwt_energy, axis=0)
    fig.add_trace(go.Scatter(x=t, y=total_energy_time, name='Total Energy',
                            line=dict(color='green', width=2)),
                 row=3, col=1)
    
    # Damage progression analysis
    window_size = 30 * fs  # 30-second windows
    freq_degradation = []
    window_times = []
    
    for start in range(0, len(tracked_freq) - window_size, window_size//2):
        end = start + window_size
        window_mean_freq = np.mean(tracked_freq[start:end])
        freq_degradation.append(window_mean_freq)
        window_times.append(t[start + window_size//2])
    
    fig.add_trace(go.Scatter(x=window_times, y=freq_degradation,
                            name='Frequency Degradation',
                            line=dict(color='purple', width=3),
                            mode='lines+markers'),
                 row=3, col=2)
    
    # Update layout
    fig.update_layout(height=1000, title_text="Wavelet-Based Damage Detection Analysis")
    fig.show()
    
    # Quantitative results
    initial_tracked_freq = np.mean(tracked_freq[:10*fs])
    final_tracked_freq = np.mean(tracked_freq[-10*fs:])
    total_degradation = (initial_tracked_freq - final_tracked_freq) / initial_tracked_freq * 100
    
    print("Exercise 5.4 Results:")
    print("="*35)
    print(f"Initial frequency: {initial_tracked_freq:.4f} Hz")
    print(f"Final frequency: {final_tracked_freq:.4f} Hz")
    print(f"Total frequency degradation: {total_degradation:.2f}%")
    print(f"Number of detected anomalous events: {len(detected_events)}")
    
    # Event timing analysis
    detected_event_times = [e['time'] for e in detected_events]
    for true_event in damage_events:
        nearby_detections = [t for t in detected_event_times if abs(t - true_event) < 10]
        print(f"Events near t={true_event}s: {len(nearby_detections)} detections")

solution_5_4()
```

### Solution 5.5: Non-Parametric Modal Identification

```python
def solution_5_5():
    """Solution for Exercise 5.5: Enhanced FDD Implementation"""
    
    # Generate multi-channel bridge data
    fs = 100
    duration = 180  # 3 minutes
    t = np.linspace(0, duration, int(fs * duration))
    n_channels = 8
    bridge_length = 80  # meters
    
    # Define true modal parameters for a multi-span bridge
    true_modal_params = {
        'frequencies': [0.95, 1.75, 2.85, 4.20, 5.65],  # Hz
        'damping_ratios': [0.015, 0.020, 0.025, 0.030, 0.035],
        'mode_shapes': []
    }
    
    # Generate realistic mode shapes for multi-span bridge
    sensor_locations = np.linspace(0, bridge_length, n_channels)
    
    # Mode shapes for multi-span bridge (more complex than simply supported)
    for mode_num, freq in enumerate(true_modal_params['frequencies']):
        if mode_num == 0:  # First mode - fundamental bending
            mode_shape = np.sin(np.pi * sensor_locations / bridge_length)
        elif mode_num == 1:  # Second mode - second bending
            mode_shape = np.sin(2 * np.pi * sensor_locations / bridge_length)
        elif mode_num == 2:  # Third mode - third bending
            mode_shape = np.sin(3 * np.pi * sensor_locations / bridge_length)
        elif mode_num == 3:  # First torsional mode
            mode_shape = np.cos(np.pi * sensor_locations / bridge_length) * \
                        np.sin(np.pi * sensor_locations / (bridge_length/2))
        else:  # Higher order mode
            mode_shape = np.sin((mode_num + 1) * np.pi * sensor_locations / bridge_length)
        
        # Normalize mode shape
        mode_shape = mode_shape / np.max(np.abs(mode_shape))
        true_modal_params['mode_shapes'].append(mode_shape)
    
    # Generate multi-channel response data
    responses = np.zeros((len(t), n_channels))
    
    for mode_idx, freq in enumerate(true_modal_params['frequencies']):
        damping = true_modal_params['damping_ratios'][mode_idx]
        mode_shape = true_modal_params['mode_shapes'][mode_idx]
        
        # Generate modal excitation (ambient loading)
        modal_amplitude = 0.1 / (mode_idx + 1)  # Higher modes have lower amplitude
        
        # Band-limited white noise excitation
        excitation = np.random.normal(0, 1, len(t))
        # Filter around modal frequency
        sos = signal.butter(4, [freq*0.7, freq*1.3], btype='band', fs=fs, output='sos')
        filtered_excitation = signal.sosfilt(sos, excitation)
        
        # Apply to each channel according to mode shape
        for ch in range(n_channels):
            modal_response = mode_shape[ch] * modal_amplitude * filtered_excitation
            
            # Add light damping
            damped_response = modal_response * np.exp(-damping * 2 * np.pi * freq * t)
            responses[:, ch] += damped_response
    
    # Add uncorrelated noise
    noise_level = 0.005
    for ch in range(n_channels):
        responses[:, ch] += np.random.normal(0, noise_level, len(t))
    
    # Enhanced FDD Implementation
    def enhanced_fdd_complete(response_matrix, fs, freq_range=(0.1, 10)):
        """Complete Enhanced FDD implementation"""
        
        n_channels = response_matrix.shape[1]
        nperseg = min(4096, response_matrix.shape[0] // 8)
        
        # Step 1: Compute cross-PSD matrix
        freq, psd_matrix = compute_cross_psd_matrix(response_matrix, fs, nperseg)
        
        # Step 2: SVD at each frequency
        singular_values, singular_vectors = perform_svd_decomposition(psd_matrix)
        
        # Step 3: Peak picking on first singular value
        modal_frequencies, peak_indices = identify_modal_frequencies(
            freq, singular_values[:, 0], freq_range)
        
        # Step 4: Extract mode shapes
        mode_shapes = extract_mode_shapes(singular_vectors, peak_indices)
        
        # Step 5: Enhanced damping estimation
        damping_ratios = []
        for i, modal_freq in enumerate(modal_frequencies):
            if i < mode_shapes.shape[1]:
                damping = estimate_modal_damping_efdd(
                    response_matrix, fs, modal_freq, mode_shapes[:, i])
                damping_ratios.append(damping)
        
        return {
            'frequencies': freq,
            'singular_values': singular_values,
            'modal_frequencies': modal_frequencies,
            'mode_shapes': mode_shapes,
            'damping_ratios': damping_ratios,
            'psd_matrix': psd_matrix
        }
    
    def compute_cross_psd_matrix(response_matrix, fs, nperseg):
        """Compute full cross-PSD matrix"""
        n_channels = response_matrix.shape[1]
        
        # Get frequency vector
        freq, _ = signal.welch(response_matrix[:, 0], fs, nperseg=nperseg)
        n_freq = len(freq)
        
        # Initialize PSD matrix
        psd_matrix = np.zeros((n_freq, n_channels, n_channels), dtype=complex)
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    # Auto-PSD
                    freq, psd_matrix[:, i, j] = signal.welch(
                        response_matrix[:, i], fs, nperseg=nperseg)
                else:
                    # Cross-PSD
                    freq, psd_matrix[:, i, j] = signal.csd(
                        response_matrix[:, i], response_matrix[:, j], fs, nperseg=nperseg)
        
        return freq, psd_matrix
    
    def perform_svd_decomposition(psd_matrix):
        """SVD decomposition at each frequency"""
        n_freq, n_channels, _ = psd_matrix.shape
        
        singular_values = np.zeros((n_freq, n_channels))
        singular_vectors = np.zeros((n_freq, n_channels, n_channels), dtype=complex)
        
        for i in range(n_freq):
            try:
                U, S, Vh = np.linalg.svd(psd_matrix[i, :, :])
                singular_values[i, :] = S
                singular_vectors[i, :, :] = U
            except np.linalg.LinAlgError:
                # Handle singular matrices
                singular_values[i, :] = 0
                singular_vectors[i, :, :] = np.eye(n_channels)
        
        return singular_values, singular_vectors
    
    def identify_modal_frequencies(freq, first_singular_value, freq_range):
        """Identify modal frequencies using peak picking"""
        # Apply frequency range filter
        freq_mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
        freq_subset = freq[freq_mask]
        sv1_subset = first_singular_value[freq_mask]
        
        # Smooth for better peak detection
        sv1_smooth = signal.savgol_filter(sv1_subset, 
                                         min(21, len(sv1_subset)//10), 2)
        
        # Peak detection
        min_distance = len(freq_subset) // 30  # Minimum distance between peaks
        peaks, properties = find_peaks(sv1_smooth, 
                                      height=np.max(sv1_smooth)*0.05,
                                      distance=min_distance)
        
        modal_frequencies = freq_subset[peaks]
        
        # Convert peak indices back to full frequency array
        full_peak_indices = np.where(freq_mask)[0][peaks]
        
        return modal_frequencies, full_peak_indices
    
    def extract_mode_shapes(singular_vectors, peak_indices):
        """Extract mode shapes from singular vectors at modal frequencies"""
        n_modes = len(peak_indices)
        n_channels = singular_vectors.shape[1]
        
        mode_shapes = np.zeros((n_channels, n_modes))
        
        for i, peak_idx in enumerate(peak_indices):
            # First singular vector at modal frequency
            mode_vector = singular_vectors[peak_idx, :, 0]
            
            # Take real part and normalize
            mode_shape = np.real(mode_vector)
            mode_shape = mode_shape / np.max(np.abs(mode_shape))
            
            mode_shapes[:, i] = mode_shape
        
        return mode_shapes
    
    def estimate_modal_damping_efdd(response_matrix, fs, modal_freq, mode_shape):
        """Enhanced FDD damping estimation"""
        # Project multi-channel response onto mode shape
        modal_coordinate = np.dot(response_matrix, mode_shape)
        
        # Frequency domain approach
        freq, modal_psd = signal.welch(modal_coordinate, fs, nperseg=2048)
        
        # Extract frequency band around modal frequency
        bandwidth = 0.2  # Hz
        freq_mask = (freq >= modal_freq - bandwidth) & (freq <= modal_freq + bandwidth)
        
        # Create SDOF function
        sdof_psd = np.zeros_like(modal_psd)
        sdof_psd[freq_mask] = modal_psd[freq_mask]
        
        # Convert to time domain correlation function
        correlation_func = np.fft.irfft(sdof_psd, n=len(modal_coordinate))
        
        # Extract envelope using Hilbert transform
        correlation_envelope = np.abs(hilbert(correlation_func))
        
        # Fit exponential decay for damping estimation
        try:
            dt = 1.0 / fs
            time_vector = np.arange(len(correlation_envelope)) * dt
            
            # Use first part of decay for fitting
            n_fit = min(500, len(correlation_envelope) // 4)
            time_fit = time_vector[:n_fit]
            envelope_fit = correlation_envelope[:n_fit]
            
            # Remove very small values
            valid_mask = envelope_fit > np.max(envelope_fit) * 0.01
            time_fit = time_fit[valid_mask]
            envelope_fit = envelope_fit[valid_mask]
            
            if len(time_fit) > 20:
                # Linear fit in log domain
                log_envelope = np.log(envelope_fit)
                coeffs = np.polyfit(time_fit, log_envelope, 1)
                decay_rate = -coeffs[0]
                
                # Calculate damping ratio
                omega_n = 2 * np.pi * modal_freq
                damping_ratio = decay_rate / omega_n
                
                # Ensure reasonable bounds
                damping_ratio = np.clip(damping_ratio, 0.005, 0.15)
            else:
                damping_ratio = 0.02  # Default
                
        except:
            damping_ratio = 0.02  # Default if fitting fails
        
        return damping_ratio
    
    # Perform Enhanced FDD analysis
    print("Performing Enhanced FDD modal identification...")
    fdd_results = enhanced_fdd_complete(responses, fs)
    
    # Calculate Modal Assurance Criterion (MAC)
    def calculate_mac_matrix(true_modes, identified_modes):
        """Calculate MAC matrix between true and identified mode shapes"""
        n_true = len(true_modes)
        n_identified = identified_modes.shape[1]
        
        mac_matrix = np.zeros((n_true, n_identified))
        
        for i in range(n_true):
            for j in range(n_identified):
                true_mode = true_modes[i]
                identified_mode = identified_modes[:, j]
                
                # Ensure same sign convention
                if np.dot(true_mode, identified_mode) < 0:
                    identified_mode = -identified_mode
                
                # Calculate MAC
                numerator = abs(np.dot(true_mode, identified_mode))**2
                denominator = np.dot(true_mode, true_mode) * np.dot(identified_mode, identified_mode)
                mac_matrix[i, j] = numerator / denominator if denominator > 0 else 0
        
        return mac_matrix
    
    # Calculate MAC matrix
    mac_matrix = calculate_mac_matrix(true_modal_params['mode_shapes'], 
                                     fdd_results['mode_shapes'])
    
    # Create comprehensive visualization
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['Multi-Channel Bridge Responses', 'FDD Singular Values',
                       'Mode Shape Comparison', 'MAC Matrix',
                       'Frequency Accuracy', 'Damping Estimation'],
        specs=[[{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}]]
    )
    
    # Multi-channel responses (subset for clarity)
    colors = px.colors.qualitative.Set1
    for ch in range(min(4, n_channels)):
        fig.add_trace(go.Scatter(x=t[:2000], y=responses[:2000, ch],
                                name=f'Ch {ch+1}', line=dict(color=colors[ch])),
                     row=1, col=1)
    
    # FDD singular values
    fig.add_trace(go.Scatter(x=fdd_results['frequencies'], 
                            y=fdd_results['singular_values'][:, 0],
                            name='1st Singular Value', line=dict(color='blue', width=2)),
                 row=1, col=2)
    
    # Mark identified frequencies
    sv1_at_peaks = [fdd_results['singular_values'][
        np.argmin(np.abs(fdd_results['frequencies'] - f)), 0] 
        for f in fdd_results['modal_frequencies']]
    
    fig.add_trace(go.Scatter(x=fdd_results['modal_frequencies'], y=sv1_at_peaks,
                            mode='markers', name='Identified Modes',
                            marker=dict(color='red', size=10, symbol='diamond')),
                 row=1, col=2)
    
    # Mode shape comparison
    n_compare = min(3, len(true_modal_params['mode_shapes']), 
                   fdd_results['mode_shapes'].shape[1])
    
    for i in range(n_compare):
        # True mode shapes
        fig.add_trace(go.Scatter(x=sensor_locations, y=true_modal_params['mode_shapes'][i],
                                name=f'True Mode {i+1}', 
                                line=dict(color=colors[i], width=3),
                                mode='lines+markers'),
                     row=2, col=1)
        
        # Identified mode shapes
        identified_shape = fdd_results['mode_shapes'][:, i]
        # Match sign convention
        if np.dot(identified_shape, true_modal_params['mode_shapes'][i]) < 0:
            identified_shape = -identified_shape
            
        fig.add_trace(go.Scatter(x=sensor_locations, y=identified_shape,
                                name=f'FDD Mode {i+1}',
                                line=dict(color=colors[i], width=2, dash='dash'),
                                mode='lines+markers'),
                     row=2, col=1)
    
    # MAC matrix heatmap
    fig.add_trace(go.Heatmap(z=mac_matrix, x=[f'ID{i+1}' for i in range(mac_matrix.shape[1])],
                            y=[f'True{i+1}' for i in range(mac_matrix.shape[0])],
                            colorscale='RdYlBu_r', zmin=0, zmax=1,
                            colorbar=dict(title="MAC Value")),
                 row=2, col=2)
    
    # Frequency accuracy comparison
    n_freq_compare = min(len(true_modal_params['frequencies']), 
                        len(fdd_results['modal_frequencies']))
    
    fig.add_trace(go.Scatter(x=list(range(1, n_freq_compare+1)),
                            y=true_modal_params['frequencies'][:n_freq_compare],
                            name='True Frequencies', mode='markers+lines',
                            marker=dict(color='blue', size=10)),
                 row=3, col=1)
    
    fig.add_trace(go.Scatter(x=list(range(1, n_freq_compare+1)),
                            y=fdd_results['modal_frequencies'][:n_freq_compare],
                            name='FDD Frequencies', mode='markers+lines',
                            marker=dict(color='red', size=10)),
                 row=3, col=1)
    
    # Damping comparison
    fig.add_trace(go.Scatter(x=list(range(1, n_freq_compare+1)),
                            y=true_modal_params['damping_ratios'][:n_freq_compare],
                            name='True Damping', mode='markers+lines',
                            marker=dict(color='green', size=10)),
                 row=3, col=2)
    
    if len(fdd_results['damping_ratios']) > 0:
        fig.add_trace(go.Scatter(x=list(range(1, min(n_freq_compare, len(fdd_results['damping_ratios']))+1)),
                                y=fdd_results['damping_ratios'][:min(n_freq_compare, len(fdd_results['damping_ratios']))],
                                name='EFDD Damping', mode='markers+lines',
                                marker=dict(color='orange', size=10)),
                     row=3, col=2)
    
    fig.update_layout(height=1000, title_text="Enhanced FDD Modal Identification Results")
    fig.show()
    
    # Quantitative assessment
    print("Exercise 5.5 Results:")
    print("="*50)
    print(f"{'Mode':<6} {'True f':<10} {'FDD f':<10} {'Error %':<10} {'True ζ':<10} {'EFDD ζ':<10} {'MAC':<8}")
    print("-"*50)
    
    for i in range(n_freq_compare):
        true_freq = true_modal_params['frequencies'][i]
        fdd_freq = fdd_results['modal_frequencies'][i]
        freq_error = abs(true_freq - fdd_freq) / true_freq * 100
        
        true_damp = true_modal_params['damping_ratios'][i]
        fdd_damp = fdd_results['damping_ratios'][i] if i < len(fdd_results['damping_ratios']) else 0
        
        # Find best MAC value for this mode
        best_mac = np.max(mac_matrix[i, :]) if i < mac_matrix.shape[0] else 0
        
        print(f"{i+1:<6} {true_freq:<10.3f} {fdd_freq:<10.3f} {freq_error:<10.2f} "
              f"{true_damp:<10.3f} {fdd_damp:<10.3f} {best_mac:<8.3f}")
    
    # Overall assessment
    avg_freq_error = np.mean([abs(t-f)/t*100 for t,f in zip(
        true_modal_params['frequencies'][:n_freq_compare],
        fdd_results['modal_frequencies'][:n_freq_compare])])
    
    avg_mac = np.mean([np.max(mac_matrix[i, :]) for i in range(min(mac_matrix.shape[0], n_freq_compare))])
    
    print(f"\nOverall Performance:")
    print(f"Average frequency error: {avg_freq_error:.2f}%")
    print(f"Average MAC value: {avg_mac:.3f}")

solution_5_5()
```

---

## 5.13 References

1. **Brincker, R., Zhang, L., & Andersen, P.** (2001). Modal identification of output-only systems using frequency domain decomposition. *Smart Materials and Structures*, 10(3), 441-445.

2. **Qin, S., Zhang, Q., & Wei, X.** (2024). Sensitive properties of power spectral density transmissibility (PSDT) to moving vehicles and structural states in bridge health monitoring. *Structural Control and Health Monitoring*, Article 4695910.

3. **Bel-Hadj, Y., Weijtjens, W., & Devriendt, C.** (2025). Structural health monitoring in a population of similar structures with self-supervised learning: A two-stage approach for enhanced damage detection and model tuning. *Structural Health Monitoring*, Article 1324194.

4. **Kuok, S. C., & Yuen, K. V.** (2020). Application of wavelet transform in structural health monitoring. *Earthquake Engineering and Engineering Vibration*, 19(3), 515-532.

5. **Pioldi, F., Ferrari, R., & Rizzi, E.** (2017). Seismic FDD modal identification and monitoring of building properties from real strong-motion structural response signals. *Structural Control and Health Monitoring*, 24(11), e1982.

6. **Au, S. K.** (2011). Fast Bayesian FFT method for ambient modal identification with separated modes. *Journal of Engineering Mechanics*, 137(3), 214-226.

7. **Tarinejad, R., & Damadipour, M.** (2014). Modal identification of structures by a novel approach based on FDD-wavelet method. *Journal of Sound and Vibration*, 333(3), 1024-1045.

8. **Amezquita-Sanchez, J. P., Park, H. S., & Adeli, H.** (2017). A novel methodology for modal parameters identification of large smart structures using MUSIC, empirical wavelet transform, and Hilbert transform. *Engineering Structures*, 147, 48-59.

9. **An, Y., & Chatzi, E.** (2019). Recent progress and future trends on damage identification methods for bridge structures. *Structural Control and Health Monitoring*, 26(10), e2416.

10. **Devriendt, C., & Guillaume, P.** (2008). Identification of modal parameters from transmissibility measurements. *Journal of Sound and Vibration*, 314(1), 343-356.

---

**End of Chapter 5**

*This chapter has provided a comprehensive foundation in frequency-domain analysis techniques essential for modern structural health monitoring. The combination of theoretical understanding, practical implementation, and real-world applications prepares students to apply these powerful tools in bridge engineering and infrastructure monitoring contexts.*