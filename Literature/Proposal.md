# Bachelor Thesis Proposal

## Title

**Neural Network-Based Time Series Analysis for Axle Detection Using Bridge
Weigh-In-Motion (B-WIM) Systems**

## 1. Introduction

Bridge Weigh-In-Motion (B-WIM) systems are widely used to estimate vehicle loads by
measuring the dynamic response of bridges as vehicles pass over them. A key step in
B-WIM analysis is accurate axle detection, which involves identifying the number of axles
and their time locations from bridge response signals such as strain or acceleration.
Traditional axle detection methods rely on thresholding, peak detection, or rule-based signal
processing techniques. While effective under ideal conditions, these approaches often
struggle with noise, varying vehicle speeds, overlapping axles, and environmental effects.
Bridge response signals are inherently time series data that can exhibit nonlinearity and
non-stationarity, making axle detection a challenging task.
Recent developments in neural networks offer new opportunities for analyzing such time
series data. By learning patterns directly from measured signals, neural networks can
potentially improve the accuracy and robustness of axle detection in B-WIM systems. This
bachelor thesis proposes the application of neural network-based time series analysis
methods for axle detection using B-WIM data.

## 2. Problem Statement

Accurate axle detection is critical for reliable load estimation in B-WIM systems. Existing
signal processing-based methods may fail when signals are noisy, vehicle speeds vary, or
multiple axles are closely spaced. There is a need for data-driven methods that can adapt to
complex signal characteristics without extensive manual tuning.
The problem addressed in this thesis is how neural networks can be used to analyze bridge
response time series and automatically detect axle events in a reliable and efficient manner.

## 3. Objectives

The main objectives of this bachelor thesis are:

1. To study the fundamentals of B-WIM systems and axle detection techniques.
2. To review basic neural network models used for time series analysis.


3. To develop a neural network-based approach for axle detection using B-WIM
    response signals.
4. To compare the performance of the proposed method with a conventional axle
    detection technique.
5. To evaluate the robustness of the neural network model under different noise and
    speed conditions.

## 4. Research Questions

This study aims to answer the following questions:

1. Can neural networks accurately detect axle locations from B-WIM time series data?
2. How does the neural network-based approach compare to traditional signal
    processing methods?
3. What type of neural network architecture is suitable for bachelor-level implementation
    and limited data?

## 5. Literature Review (Brief Overview)

B-WIM systems estimate axle weights by analyzing bridge responses caused by moving
vehicles. Axle detection is typically performed using peak detection, wavelet analysis, or
influence line methods. Neural networks have been increasingly applied to time series
problems such as pattern recognition and event detection, including applications in vibration
analysis and structural monitoring.
This thesis will review key studies on B-WIM axle detection and introductory neural network
approaches for time series analysis, focusing on methods suitable for undergraduate-level
research.

## 6. Methodology

### 6.1 Data Description

```
● Use of simulated or experimental B-WIM data (e.g., bridge strain or acceleration
signals)
● Time series corresponding to vehicle crossings
● Ground truth axle positions obtained from known vehicle configurations or reference
sensors
```
### 6.2 Data Preprocessing

```
● Signal filtering and normalization
● Segmentation of time series into sliding windows
● Labeling windows as axle or non-axle events
```
### 6.3 Neural Network Model


```
● Selection of a simple and interpretable model such as:
○ Feedforward Neural Network (MLP), or
○ 1D Convolutional Neural Network (CNN)
● Input: short time windows of bridge response signals
● Output: probability of axle presence
```
### 6.4 Baseline Method

```
● Implementation of a traditional axle detection approach (e.g., peak detection)
● Use as a benchmark for comparison
```
### 6.5 Evaluation

```
● Performance metrics: detection accuracy, precision, recall
● Comparison between neural network and baseline method
```
## 7. Expected Results

The expected outcomes of this thesis are:

1. A working neural network model capable of detecting axles from B-WIM time series
    data.
2. Demonstration of improved robustness compared to a simple threshold-based
    method.
3. Insights into the advantages and limitations of neural networks for B-WIM axle
    detection.

## 8. Scope and Limitations

This is a bachelor-level study with limited time and computational resources. The focus will
be on simple neural network architectures and a single application scenario. Advanced
models and large-scale deployment are outside the scope of this work.

## 9. Proposed Timeline

```
Phase Activities Duration
Phase 1 Literature review and background study Weeks 1–
Phase 2 Data preparation and preprocessing Weeks 2–
```

```
Phase 3 Model development Weeks 3–
Phase 4 Testing and evaluation Weeks 7–
Phase 5 Thesis writing and submission Weeks 9–
```
## 10. Significance of the Study

This thesis will demonstrate how modern data-driven techniques can be applied to a
practical civil engineering problem. The results may contribute to improving axle detection in
B-WIM systems and provide a foundation for future research at postgraduate level.