# Method: Unsupervised Sleep-Stage Clustering and Dynamical Analysis

## 1. Overview

We developed an unsupervised analysis pipeline for sleep-stage discovery from EEG and optional EMG recordings. The workflow was designed to identify recurrent latent states, compare clustering strategies, characterize transition dynamics, and extract interpretable feature thresholds from the discovered states. The complete pipeline is modular and includes signal preprocessing, window-based feature extraction, unsupervised clustering, state alignment, Hidden Markov Model analysis, low-dimensional manifold learning, and rule-based explainability analysis.

## 2. Input Data

The pipeline accepts continuous EEG signals as the primary input and optional EMG signals as an auxiliary modality. Each recording is associated with its sampling frequency. The implementation supports `EDF`, `CSV`, and `MAT` files, and internally converts all supported formats into a unified signal representation for downstream analysis.

## 3. Signal Preprocessing

EEG and EMG signals are preprocessed before feature extraction. EEG is band-pass filtered to preserve physiologically relevant low- and mid-frequency components while suppressing slow drift and high-frequency noise. EMG is filtered in a higher frequency band appropriate for muscle activity. Optional notch filtering is applied to suppress power-line interference. After filtering, each channel is standardized to reduce inter-recording amplitude variability and to stabilize subsequent clustering.

## 4. Window Segmentation

The preprocessed signals are divided into fixed-length windows with optional overlap. The pipeline supports second-based windows, sample-based windows, and a notebook-compatible automatic windowing rule. Each window is treated as the fundamental unit for feature extraction and state assignment. Window metadata, including start time and end time, are retained to support temporal visualization and state transition analysis.

## 5. Feature Extraction

For each window, the system extracts both spectral and time-domain descriptors.

### 5.1 EEG Features

EEG features include spectral power estimates derived from the windowed frequency spectrum, total energy, dominant frequency, spectral entropy, and canonical band-power summaries such as delta, theta, alpha, sigma, and beta power. Relative band-power ratios are also computed to capture spectral composition.

### 5.2 EMG Features

When EMG is available, the pipeline extracts energy-related and time-domain features from each EMG window, allowing muscle-tone information to contribute to state separation.

### 5.3 Time-Domain Features

For both EEG and EMG windows, the following time-domain features can be computed:

- peak-to-peak amplitude
- zero-crossing rate
- peak count
- energy or integrated squared amplitude

The feature extractor also supports a notebook-compatible legacy mode that reproduces a reduced four-feature representation consisting of peak-to-peak amplitude, zero-crossing rate, peak count, and `iEMG²`.

## 6. Feature Scaling

Extracted features are normalized before clustering. The pipeline supports both z-score normalization and min-max scaling. Z-score normalization is recommended for the full multifeature pipeline, whereas min-max scaling is available for compatibility with prior notebook-based analyses.

## 7. Unsupervised Clustering

Three clustering strategies are applied independently to the normalized feature matrix:

- KMeans
- Gaussian Mixture Model with diagonal covariance
- Agglomerative Hierarchical Clustering

Each method produces a state-label sequence across windows. Cluster quality is assessed using internal metrics such as silhouette score, Davies-Bouldin index, cluster-balance entropy, and, for the Gaussian mixture model, information criteria including AIC and BIC.

## 8. State Alignment Across Clustering Methods

Because independent clustering methods may assign different numeric labels to similar latent states, the resulting labels are aligned to a common reference. Label alignment is performed using a confusion-matrix-based matching strategy, allowing direct comparison of temporal state sequences, state-specific spectra, transition matrices, and explainability outputs across methods.

## 9. Hidden Markov Model and State Dynamics

To characterize temporal organization directly from the continuous feature trajectory, the window-level feature time series is modeled using a Gaussian Hidden Markov Model with continuous emissions. In the current implementation, Gaussian emissions are parameterized with diagonal covariance matrices for numerical stability and efficient estimation. The HMM is fitted directly to the normalized feature sequence rather than to discrete cluster labels.

Model comparison is performed across three candidate state-space sizes:

- 3-state Gaussian HMM
- 4-state Gaussian HMM
- 5-state Gaussian HMM

For each candidate model, the pipeline estimates:

- hidden-state sequence
- state transition matrix
- initial-state distribution
- stationary-state distribution
- state-specific Gaussian means and variances
- run lengths and state-duration distributions
- log likelihood
- Bayesian Information Criterion (BIC)

The hidden-state sequence is inferred from the fitted model and used as the primary output of the HMM module. Log likelihood and BIC are then used to compare the 3-state, 4-state, and 5-state solutions and to assess the trade-off between model fit and complexity.

## 10. Low-Dimensional Manifold Learning

To visualize the geometry of the extracted feature space and its temporal evolution, the high-dimensional window-level feature representation is projected into a low-dimensional embedding using one of the following approaches:

- Principal Component Analysis
- UMAP
- Diffusion Map

The resulting embedding is used to visualize state-dependent point clouds, continuous trajectories over time, and short-step transitions between neighboring windows. This representation helps assess whether discovered states occupy separable regions and whether state trajectories evolve continuously over time.

## 11. Explainability and Threshold Extraction

To improve interpretability, the aligned cluster labels are treated as pseudo-targets and modeled using a shallow decision-tree rule extractor built on the full input feature set. This stage yields:

- cluster-associated feature thresholds
- feature importance ranking
- human-readable decision rules
- fidelity between extracted rules and original cluster labels

Because thresholds are derived from normalized features but mapped back to the original feature scale, the resulting rules remain interpretable in physiologically meaningful units.

## 12. Visualization and Outputs

The pipeline automatically exports intermediate data tables, figures, and reports. Representative outputs include:

- raw versus filtered signal traces
- signal segmentation plots with window boundaries
- single-window spectra and spectral heatmaps
- cluster timelines and cluster-wise feature distributions
- mean spectra for each cluster
- confusion matrices before and after label alignment
- HMM transition-matrix heatmaps and duration histograms
- manifold scatter plots and temporal trajectories
- decision-tree visualizations and threshold heatmaps
- integrated markdown reports summarizing clustering quality and explainability

## 13. GPU Acceleration and Multi-Platform Compatibility

The implementation supports optional GPU acceleration for computationally intensive stages. On Apple Silicon devices such as the MacBook Air M3, acceleration is available through `PyTorch + MPS`. On Windows systems equipped with NVIDIA GPUs such as the RTX 5070 Ti, acceleration is available through `PyTorch + CUDA`. In environments without GPU support or without an installed PyTorch runtime, the pipeline falls back automatically to a NumPy-based CPU implementation. GPU acceleration currently targets batched feature computation, KMeans, Gaussian mixture fitting, Gaussian HMM fitting, and selected manifold operations, while signal filtering, hierarchical clustering, and figure generation remain CPU-based in the current version.

## 14. Reproducibility

The pipeline exposes configuration parameters for filtering, window size, overlap, cluster number, random seed, feature scaling mode, dimensionality-reduction method, and acceleration backend. All major intermediate outputs are written to disk, including feature matrices, labels, transition matrices, explainability tables, and configuration metadata, enabling full analysis traceability and repeated experiments under controlled settings.

## 15. Suggested Concise Method Summary

The analysis pipeline first preprocesses continuous EEG and optional EMG recordings using band-pass and notch filtering followed by signal normalization. The filtered signals are segmented into fixed windows, from which spectral and time-domain features are extracted. Unsupervised state discovery is then performed using KMeans, Gaussian mixture modeling, and hierarchical clustering. In parallel, the continuous feature time series is modeled using Gaussian Hidden Markov Models with 3, 4, and 5 hidden states, and the competing models are compared using log likelihood and BIC. To visualize latent organization, the feature space is embedded into two or three dimensions using PCA, UMAP, or Diffusion Map. Finally, a shallow decision-tree rule extractor is trained on the full feature set to obtain cluster-specific thresholds and feature-importance estimates, thereby linking discovered latent states to interpretable physiological signatures.
