# Department of Computer Science and Engineering
## Faculty of Science and Engineering
### City University
**Khagan, Birulia, Savar, Dhaka-1340, Bangladesh**

---

# BACHELOR’S PROJECT/THESIS PROPOSAL REPORT

**Title:** AI-Based Skin Disease Detection from Skin Images  
**Program:** Bachelor of Science in Computer Science and Engineering  
**Academic Session:** 2025–2026  
**Format Standard:** City University CSE Proposal Guidelines | Times New Roman 12 pt | 1.5 Line Spacing | A4 Paper | Margins: Left 1.25", Top/Right/Bottom 1.0" | Max 5,000 Words (Excl. Bibliography)  

---

# 1. Background and Statement of the Problem

Dermatological diseases constitute one of the top four causes of non-fatal disease burden globally, affecting over two billion individuals across diverse demographics (Esteva et al., 2017). The clinical spectrum spans benign epithelial lesions, inflammatory dermatoses (such as melanocytic nevi and seborrheic keratoses), and aggressive malignant tumors, notably basal cell carcinoma (BCC), squamous cell carcinoma (SCC), and malignant melanoma (MEL). Malignant melanoma accounts for roughly 75% of skin-cancer deaths despite representing under 5% of cutaneous neoplasms (Codella et al., 2018). Prognosis depends critically on early diagnosis: localized surgical excision yields an empirical five-year survival rate exceeding 99%, whereas distant metastasis causes survival to drop below 30% (Codella et al., 2018).

In clinical practice, diagnostic evaluation relies on visual inspection and optical dermatoscopy. Dermatoscopy reduces surface reflection to reveal sub-macroscopic pigmentary patterns in the epidermis and dermis (Tschandl, Rosendahl, & Kittler, 2018). However, diagnostic accuracy remains subjective: unassisted visual triage by general practitioners achieves a sensitivity of merely 55% to 65% for pigmented malignancies, while certified dermatologists attain 80% to 90% (Esteva et al., 2017). Diagnostic errors occur because distinct pathologies display overlapping morphological patterns (e.g., border scalloping and irregular erythema), while identical conditions vary across Fitzpatrick skin phototypes. In developing nations like Bangladesh, a severe shortage of certified dermatologists exacerbates diagnostic delays and elevated mortality.

Deep Learning (DL) and Computer Vision have transformed biomedical image analysis (LeCun, Bengio, & Hinton, 2015). Traditional Computer-Aided Diagnosis (CAD) pipelines relied on manual feature extraction replicating the ABCD criteria (Asymmetry, Border, Color, Diameter), which suffered from poor generalization and sensitivity to lighting and hair artifacts (Codella et al., 2018). Conversely, Deep Convolutional Neural Networks (CNNs) learn hierarchical representations directly from raw pixels: lower layers extract edge gradients and textures, while deeper layers synthesize high-level diagnostic representations (Kassem, Hosny, & Fouad, 2020).

Despite promising demonstrations, critical translational bottlenecks persist. Benchmark dermatological datasets exhibit extreme class imbalance, where benign conditions outnumber malignancies by orders of magnitude; standard cross-entropy optimization biases decision boundaries toward majority classes, generating high overall accuracy while failing on lethal malignancies. Furthermore, literature lacks controlled empirical comparisons between deep residual identity mappings (He, Zhang, Ren, & Sun, 2016) and multi-dimensional compound-scaled networks (Tan & Le, 2019). Finally, most academic studies stop at offline test-set evaluation without deploying accessible inference microservices or establishing mechanisms to assimilate incoming data without label corruption (Ramirez, Chen, & Williams, 2023).

This research proposes an academically rigorous deep learning framework for multi-class skin disease classification from dermatoscopic imagery. Utilizing the standardized HAM10000 benchmark ($N = 10,015$ images across seven diagnostic classes) (Tschandl et al., 2018), this study establishes an automated preprocessing and augmentation pipeline, evaluates Class-Weighted Cross-Entropy against Focal Loss ($\gamma = 2.0$) to overcome class imbalance, and conducts a controlled benchmark between ResNet-50 and EfficientNet (B0/B4). The optimal model will be deployed as an asynchronous Python FastAPI REST microservice, supported by an audited verified-data workflow to ensure continuous, noise-free model retraining.

---

# 2. Aim and Objectives of the Proposed Research

## 2.1 Research Aim
To design, empirically benchmark, and deploy a robust, transfer-learning deep convolutional neural network framework capable of discriminating multi-class dermatoscopic skin diseases under severe class imbalance, and to serve the optimal model as an asynchronous Python FastAPI microservice integrated with an audited, human-in-the-loop verified-data retraining workflow.

## 2.2 Research Objectives

1. **Dataset Curation and Class-Stratified Partitioning:** To curate, audit, and clean the benchmark HAM10000 dataset ($N = 10,015$ dermatoscopic images) across seven diagnostic classes, establishing leak-free, class-stratified splits for training (70%), validation (15%), and independent testing (15%).
2. **Standardized Preprocessing and Augmentation Pipeline Formulation:** To engineer an automated computer vision pipeline executing spatial standardization, ImageNet dynamic range normalization, and clinically conservative augmentations to suppress overfitting without distorting diagnostic lesion pathology.
3. **Deep Neural Network Construction and Transfer Learning Optimization:** To develop, fine-tune, and train ResNet-50, EfficientNet-B0, and EfficientNet-B4 backbones under controlled hyperparameter schedules, evaluating Class-Weighted Cross-Entropy against Focal Loss ($\gamma = 2.0$) to overcome empirical class imbalance.
4. **Rigorous Multi-Metric Evaluation and Architectural Benchmarking:** To perform empirical benchmarking across accuracy, macro-averaged precision, recall (sensitivity), macro F1-score, balanced accuracy, and $7 \times 7$ confusion matrix error patterns, determining the optimal trade-off between predictive sensitivity, parameter count (Millions), and inference latency (milliseconds).
5. **High-Throughput Microservice Deployment:** To serialize the optimal deep learning model binary and deploy it within an asynchronous Python FastAPI microservice architecture exposing RESTful endpoints (`POST /predict`) for low-latency client inference.
6. **Engineered Verified-Data Retraining Architecture:** To construct an MLOps data governance framework featuring a quarantine staging repository, cryptographic request logging, and a clinical verification gate to prevent label noise propagation during continuous retraining cycles.

---

# 3. Development/Research Questions

## 3.1 Research Question 1
*How effectively can pre-trained convolutional neural network backbones classify multi-class dermatoscopic lesions across seven distinct diagnostic categories under severe empirical class imbalance?*  
**Explanation and Scientific Merit:** Benchmark skin lesion repositories such as HAM10000 exhibit class imbalance ratios exceeding $58:1$ between dominant benign nevi and rare dermatofibromas. Standard unweighted optimization biases network convergence toward majority classes, masking diagnostic failure on lethal conditions like melanoma. Investigating this question evaluates whether transfer-learning representations can generalize across poorly represented diagnostic categories.

## 3.2 Research Question 2
*What are the comparative empirical trade-offs between deep residual identity mappings (ResNet-50) and compound-scaled mobile inverted bottleneck convolutions (EfficientNet-B0/B4) regarding macro F1-score, malignant lesion recall, parameter footprint, and inference latency?*  
**Explanation and Scientific Merit:** ResNet utilizes linear additive skip connections to alleviate vanishing gradients, whereas EfficientNet employs multi-dimensional compound scaling with Squeeze-and-Excitation attention. A controlled head-to-head comparison under identical preprocessing and optimization schedules provides evidence regarding architectural suitability for clinical deployment versus resource-constrained inference.

## 3.3 Research Question 3
*To what extent does the formulation of Focal Loss ($\gamma = 2.0$) mitigate majority-class gradient dominance compared to static Class-Weighted Cross-Entropy Loss and standard Empirical Risk Minimization?*  
**Explanation and Scientific Merit:** While static inverse-frequency weighting penalizes minority errors uniformly, Focal Loss dynamically modulates the loss via $(1 - p_t)^\gamma$ to suppress gradients from well-classified easy samples. Evaluating this dynamic loss formulation is crucial for determining whether gradient redirection improves discrimination along visually ambiguous lesion boundaries.

## 3.4 Research Question 4
*Can an asynchronous Python FastAPI web microservice achieve real-time, low-latency ($< 50$ ms) inference serving while maintaining numerical consistency with the offline-validated deep learning pipeline?*  
**Explanation and Scientific Merit:** Academic machine learning models often suffer from deployment degradation caused by tensor deserialization overhead and unoptimized server concurrency. Implementing and benchmarking an asynchronous ASGI-based prediction endpoint evaluates the operational feasibility of deploying complex deep learning weights within clinical decision-support systems.

## 3.5 Research Question 5
*How can an architectural data feedback loop be structured to incorporate incoming user-submitted images into continuous retraining cycles without compromising ground-truth data integrity?*  
**Explanation and Scientific Merit:** Automated retraining on unverified inference inputs creates positive feedback loops of catastrophic error propagation, where inaccurate predictions become false training labels. Designing a quarantine-buffered, human-in-the-loop verification protocol establishes the systemic foundation for ethical, auditable clinical AI.

---

# 4. Literature Review

## 4.1 Artificial Intelligence in Medical Image Analysis
Automated medical image interpretation has transitioned from rule-based heuristics to data-driven deep representation learning (Esteva et al., 2017). Classical computer-aided diagnostic algorithms in dermatology extracted low-level hand-crafted features based on the ABCD criteria, texture co-occurrence matrices, and border filters. Codella et al. (2018) evaluated multiple ISIC benchmark challenges and demonstrated that hand-crafted extractors consistently underperform end-to-end deep learning models, failing to capture subtle spatial variations in chromatic variegation, atypical pigment networks, and vascular structures.

Deep neural networks eliminate reliance on subjective feature engineering by discovering optimal hierarchical representations directly from raw pixel arrays via stochastic gradient descent (LeCun et al., 2015). In a landmark study, Esteva et al. (2017) demonstrated that a deep CNN (Inception-v3), pre-trained on ImageNet and fine-tuned on 129,450 clinical skin images, achieved diagnostic sensitivity and specificity matching 21 certified dermatologists across binary biopsy-verified clinical tasks. However, Esteva et al. evaluated clinical photography rather than standardized dermatoscopy, leaving open questions regarding multi-class performance on dermatoscopic benchmarks.

## 4.2 Convolutional Neural Networks and Spatial Representation Learning
Convolutional Neural Networks represent the gold standard computational architecture for two-dimensional spatial pattern recognition (LeCun et al., 2015). A CNN layer maps an input tensor $\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}$ to an output feature map $\mathbf{Z} \in \mathbb{R}^{H' \times W' \times C_{out}}$ via discrete convolution with a parameterized bank of spatial kernels $\mathbf{K} \in \mathbb{R}^{k_h \times k_w \times C_{in} \times C_{out}}$:

$$Z_{i,j,k} = \sigma \left( \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} \sum_{c=0}^{C_{in}-1} X_{i+m, j+n, c} \cdot K_{m,n,c,k} + b_k \right) \tag{1}$$

where $\sigma(\cdot)$ denotes a non-linear activation function, and $b_k$ represents the channel bias.

```text
Raw Image Tensor X ∈ R^(H × W × 3)
       ↓
[Convolution Layer + Batch Normalization + Activation σ(·)]
       ↓
[Spatial Downsampling (Stride / Max Pooling)]
       ↓
[Deep Residual / Inverted Bottleneck Feature Extractors]
       ↓
[Global Average Pooling: (H' × W' × C) → (1 × 1 × C)]
       ↓
[Fully Connected Layer + Softmax Activation]
       ↓
Predicted Multi-Class Probability Distribution ŷ ∈ [0, 1]^K
```

The intrinsic properties of CNNs—namely translation equivariance and local receptive field connectivity—enable the network to identify localized pathological cues (such as vascular dots and pigment networks) irrespective of spatial translation (Kassem et al., 2020).

## 4.3 Transfer Learning and Advanced Model Backbones
Training deep convolutional architectures containing tens of millions of parameters from random initialization requires massive labeled corpora to avoid empirical overfitting. In biomedical domains, dataset dimensions are constrained by patient privacy, acquisition costs, and the need for histopathological verification (Tschandl et al., 2018). Transfer learning overcomes this data bottleneck by reusing spatial feature extractors pre-trained on generic benchmarks (ImageNet-1k) and fine-tuning these weights on specialized target domains (Srinivasu et al., 2021).

### 4.3.1 Residual Networks (ResNet-50)
In traditional feedforward networks, increasing depth causes vanishing gradients during backpropagation, leading to training saturation (He et al., 2016). He et al. (2016) resolved this degradation by introducing identity skip connections, reformulating stacked layers to learn a residual mapping $\mathcal{F}(\mathbf{x})$ with respect to the identity input $\mathbf{x}$:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\}) + \mathbf{x} \tag{2}$$

```text
             Input Tensor x
                   │ 
         ┌─────────┴─────────┐
         │ (Identity Bypass) │
         ▼                   │
    [ 1×1 Conv, Bottleneck ] │
         ▼                   │
    [ 3×3 Conv, Spatial ]    │
         ▼                   │
    [ 1×1 Conv, Expansion ]  │
         ▼                   │
         ├───────────────────┘
         ▼
      [ Addition: F(x) + x ]
         ▼
     [ ReLU Activation ]
         ▼
            Output Tensor y
```

Differentiating with respect to the input tensor demonstrates that the gradient consists of two additive components:

$$\frac{\partial \mathcal{E}}{\partial \mathbf{x}} = \frac{\partial \mathcal{E}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right) \tag{3}$$

Because the identity term $\mathbf{I}$ guarantees that gradient signals propagate unattenuated back to early layers, ResNet architectures can be scaled to extreme depths without performance degradation. In this study, ResNet-50 (48 convolutional layers across 25.6 million parameters) serves as the primary residual learning baseline (Kassem et al., 2020).

### 4.3.2 EfficientNet Architecture
Conventional network scaling empirically adjusted network depth, channel width, or input resolution in isolation. Tan and Le (2019) demonstrated that single-dimensional scaling results in rapid accuracy saturation and excessive computational cost. To overcome this, they formulated compound scaling, which uniformly scales depth ($d$), width ($w$), and input resolution ($r$) using a fixed compound coefficient $\phi$:

$$d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi \tag{4}$$

$$\text{subject to } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \quad \text{and} \quad \alpha \ge 1, \beta \ge 1, \gamma \ge 1 \tag{5}$$

EfficientNet backbones utilize Mobile Inverted Bottleneck Convolution (MBConv) blocks equipped with depthwise separable convolutions and Squeeze-and-Excitation (SE) attention mechanisms (Tan & Le, 2019). The SE block computes channel-wise attention weights $\mathbf{s}$:

$$\mathbf{s} = \sigma \left( \mathbf{W}_2 \cdot \delta(\mathbf{W}_1 \cdot \mathbf{z}) \right) \tag{6}$$

where $\mathbf{z} \in \mathbb{R}^C$ is the global average pooled feature vector, $\delta(\cdot)$ is Swish/ReLU, $\sigma(\cdot)$ is Sigmoid, and $\mathbf{W}_1, \mathbf{W}_2$ are channel projection matrices. In this study, EfficientNet-B0 (5.3M parameters, $224 \times 224$) and EfficientNet-B4 (19.3M parameters, $380 \times 380$) represent parameter-efficient scaling baselines (Srinivasu et al., 2021).

## 4.4 Dermatological Benchmark Datasets and Critical Appraisal
The HAM10000 dataset represents the standard benchmark foundation for computerized dermatology (Tschandl et al., 2018). Curated from the Medical University of Vienna and the Cliff Rosendahl Dermatology Practice in Queensland, HAM10000 comprises 10,015 high-resolution dermatoscopic images categorized into seven diagnostic classes.

### Table 1. Diagnostic Class Distribution and Clinical Categorization in HAM10000

| Class Code | Diagnostic Category | Clinical & Histopathological Nature | Image Count ($N_c$) | Proportion (%) | Imbalance Ratio ($NV : c$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **NV** | Melanocytic Nevi | Benign melanocytic proliferation | 6,705 | 66.95% | $1.00 : 1$ |
| **MEL** | Malignant Melanoma | Highly aggressive invasive malignancy | 1,113 | 11.11% | $6.02 : 1$ |
| **BKL** | Benign Keratosis | Seborrheic keratosis / solar lentigo | 1,099 | 10.97% | $6.10 : 1$ |
| **BCC** | Basal Cell Carcinoma | Locally invasive malignant keratinocyte tumor | 514 | 5.13% | $13.05 : 1$ |
| **AKIEC** | Actinic Keratosis / IEC | Pre-malignant / intraepidermal carcinoma | 327 | 3.26% | $20.50 : 1$ |
| **VASC** | Vascular Lesions | Benign angiomas and pyogenic granulomas | 142 | 1.42% | $47.22 : 1$ |
| **DF** | Dermatofibroma | Benign dermal mesenchymal proliferation | 115 | 1.15% | $58.30 : 1$ |
| **Total** | | | **10,015** | **100.00%** | |

A critical weakness in published literature utilizing HAM10000 is reporting unweighted top-1 accuracy as the primary success metric. Because Melanocytic Nevi (NV) constitute nearly 67% of the dataset, an uncalibrated classifier that trivially predicts NV for all inputs achieves a misleading accuracy of 67% while exhibiting zero sensitivity for lethal malignancies such as Melanoma and BCC (Kassem et al., 2020).

### Table 2. Critical Appraisal of Related Studies in Deep Dermatological Analysis

| Study Citation | Target Dataset | Core Architecture | Key Findings Reported | Critical Weaknesses & Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Esteva et al. (2017)** | Clinical Derm Repository ($N=129,450$) | Inception-v3 | Parity with 21 certified dermatologists on binary classification. | Unstandardized clinical photos; opaque preprocessing; no multi-class microservice serving. |
| **Tschandl et al. (2018)** | HAM10000 Benchmark ($N=10,015$) | Multi-expert reader study vs. basic CNN | Established open baseline; CNNs matched human reader accuracy. | Focused on dataset curation; limited loss function comparison; no production serving architecture. |
| **Kassem et al. (2020)** | ISIC 2019 Archive ($N=25,331$) | Modified VGG-19 & ResNet | Attained 89.2% overall accuracy across eight classes. | Heavy reliance on compute-intensive VGG; lacked focal loss ablation; neglected API latency metrics. |
| **Srinivasu et al. (2021)** | DermNet & Kaggle archives | MobileNetV2 + EfficientNet | Combined architecture achieved 85.3% accuracy on mobile setups. | Fragmented multi-source data; inconsistent validation splits; lacked continuous retraining audit framework. |

## 4.5 Loss Formulations for Extreme Class Imbalance
To counteract majority-class gradient dominance during stochastic optimization, this study evaluates two loss formulations against standard Cross-Entropy:

### 4.5.1 Class-Weighted Cross-Entropy Loss (WCE)
Class-Weighted Cross-Entropy introduces static weighting coefficients $w_c$ inversely proportional to class frequencies:

$$\mathcal{L}_{WCE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} w_c \cdot y_{i,c} \log(\hat{y}_{i,c}), \quad w_c = \frac{N}{K \cdot N_c} \tag{7}$$

where $N$ is total sample count, $K=7$ is number of classes, and $N_c$ represents sample volume for class $c$.

### 4.5.2 Focal Loss (FL)
Lin et al. (2017) demonstrated that static weighting fails to distinguish between easily classified examples and hard boundary cases. Focal Loss dynamically modulates the loss based on prediction confidence $\hat{p}_{i,c}$:

$$\mathcal{L}_{FL} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{K} \alpha_c (1 - \hat{p}_{i,c})^\gamma y_{i,c} \log(\hat{p}_{i,c}) \tag{8}$$

When a sample is well classified ($\hat{p}_{i,c} \rightarrow 1$), the modulating factor $(1 - \hat{p}_{i,c})^\gamma$ approaches zero, dynamically suppressing gradient updates from abundant easy samples and focusing parameter optimization onto difficult, ambiguous lesion boundaries (Lin et al., 2017).

## 4.6 Concluding Synthesis and Theoretical Framework
The synthesis of existing literature identifies an evident theoretical framework: dermatoscopic image classification is governed by the interaction between network feature representation (residual identity vs. compound scaling), loss-gradient dynamics (empirical risk vs. focal modulation under a $58:1$ class imbalance), and continuous data governance. The operationalized variables in this research comprise:
* **Independent Variables:** Neural network architecture (ResNet-50 vs. EfficientNet-B0 vs. EfficientNet-B4), loss function formulation ($\mathcal{L}_{CE}$ vs. $\mathcal{L}_{WCE}$ vs. $\mathcal{L}_{FL}$), and input resolution ($224 \times 224$ vs. $380 \times 380$).
* **Dependent Variables:** Classification accuracy, macro-averaged precision, recall (sensitivity per class), macro F1-score, balanced accuracy, parameter volume (M), and inference latency (ms).
* **Controlled Variables:** Stratified dataset split partitions (70/15/15), base optimizer (Adam), learning rate schedule, and hardware execution environment.

---

# 5. Research Methodology

## 5.1 Research Design and System Workflow
This investigation adopts an experimental, quantitative machine learning methodology. The system workflow is organized into six functional stages: data auditing and stratified partitioning, automated preprocessing and augmentation, transfer-learning model optimization, multi-metric empirical benchmarking, asynchronous FastAPI deployment, and verified-data retraining architecture.

```text
                  [ HAM10000 Benchmark Repository (N = 10,015) ]
                                         ↓
                     [ Integrity Verification & Duplicate Audit ]
                                         ↓
                 [ Stratified Split: Train 70% | Val 15% | Test 15% ]
                  /                      |                      \
        (Train: N = 7,010)      (Val: N = 1,502)        (Test: N = 1,503)
               ↓                        ↓                       ↓
     [Preprocessing &         [Preprocessing &        [Preprocessing &
      Data Augmentation]       Standardization]        Standardization]
               ↓                        ↓                       ↓
  +--------------------------------------------------------------------------+
  |              Transfer Learning Neural Network Training Suite             |
  |     ResNet-50  vs.  EfficientNet-B0  vs.  EfficientNet-B4                 |
  |     Loss Functions: Standard CE  vs.  Weighted CE  vs.  Focal Loss (γ=2) |
  +--------------------------------------------------------------------------+
                                         ↓
                     [ Hyperparameter Tuning & Early Stopping ]
                                         ↓
                     [ Multi-Metric Empirical Test Benchmarking ]
                (Accuracy, Precision, Recall, F1, Matrix, Latency)
                                         ↓
                    [ Optimal Model Selection & Serialization ]
                                         ↓
              [ FastAPI Microservice Endpoint Serving (/predict) ]
                                         ↓
               [ Verified User-Data Retraining Architecture ]
```

---

## 5.2 Dataset Partitioning and Stratification
To prevent data leakage and ensure statistically reliable evaluation, the HAM10000 dataset ($N = 10,015$) is partitioned into **70% Training ($N_{train} = 7,010$)**, **15% Validation ($N_{val} = 1,502$)**, and **15% Independent Testing ($N_{test} = 1,503$)** sets using stratified random sampling based on verified histopathological ground truth.

### Table 3. Exact Stratified Dataset Distribution Across Partitions

| Class Code | Diagnostic Label | Total Volume | Training Set (70%) | Validation Set (15%) | Independent Test Set (15%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NV** | Melanocytic Nevi | 6,705 | 4,693 | 1,006 | 1,006 |
| **MEL** | Malignant Melanoma | 1,113 | 779 | 167 | 167 |
| **BKL** | Benign Keratosis | 1,099 | 769 | 165 | 165 |
| **BCC** | Basal Cell Carcinoma | 514 | 360 | 77 | 77 |
| **AKIEC** | Actinic Keratosis | 327 | 229 | 49 | 49 |
| **VASC** | Vascular Lesions | 142 | 99 | 21 | 22 |
| **DF** | Dermatofibroma | 115 | 81 | 17 | 17 |
| **Total** | | **10,015** | **7,010** | **1,502** | **1,503** |

---

## 5.3 Automated Image Preprocessing Pipeline
Raw images are ingested via an automated Python preprocessing pipeline implemented in OpenCV and NumPy:
1. **Spatial Resampling:** Bilinear interpolation to target architectural dimensions:
   - ResNet-50 and EfficientNet-B0: $224 \times 224 \times 3$ pixels.
   - EfficientNet-B4: $380 \times 380 \times 3$ pixels.
2. **Dynamic Range Scaling:** Mapping integer pixel dynamic ranges $[0, 255]$ into floating-point tensors $[0.0, 1.0]$:

$$\mathbf{X}_{norm} = \frac{\mathbf{X}}{255.0} \tag{9}$$

3. **Channel-Wise Standardization:** Subtracting ImageNet RGB population means and dividing by standard deviations:

$$\mathbf{X}_{std} = \frac{\mathbf{X}_{norm} - \boldsymbol{\mu}_{ImageNet}}{\boldsymbol{\sigma}_{ImageNet}} \tag{10}$$

where $\boldsymbol{\mu} = [0.485, 0.456, 0.406]$ and $\boldsymbol{\sigma} = [0.229, 0.224, 0.225]$.

---

## 5.4 Data Augmentation Strategy
To enforce geometric invariance and mitigate overfitting without corrupting diagnostic markers, augmentations are applied exclusively to the training split ($N = 7,010$). Validation and testing sets remain unaugmented.

### Table 4. Clinical Data Augmentation Parameter Specifications

| Augmentation Operator | Parameter Range | Mathematical Implementation | Clinical Justification |
| :--- | :--- | :--- | :--- |
| **Random Rotation** | $\theta \in [-30^\circ, +30^\circ]$ | Affine Euclidean rotation matrix | Simulates arbitrary dermatoscope placement angles. |
| **Horizontal Flip** | Probability $p = 0.5$ | Matrix reflection along y-axis | Enforces bilateral symmetry invariance of lesions. |
| **Vertical Flip** | Probability $p = 0.5$ | Matrix reflection along x-axis | Enforces vertical mirror invariance of lesions. |
| **Scaling (Zoom)** | Factor $s \in [0.9, 1.1]$ | Uniform isotropic affine scaling | Models slight differences in skin-to-lens distance. |
| **Translation Shift** | $\Delta x, \Delta y \in [-10\%, +10\%]$ | Spatial coordinate offset | Simulates non-centered lesion framing in dermatoscope. |
| **Brightness Scaling**| Factor $\beta \in [0.85, 1.15]$ | Linear intensity multiplication | Simulates variations in dermatoscope LED illumination. |

---

## 5.5 Model Architectures and Transfer Learning Setup
The study establishes three transfer-learning network pipelines:
1. **ResNet-50 Pipeline:** Pre-trained ImageNet backbone; base layers frozen initially, unfreezing the final residual block stack (`conv5_block3`) for fine-tuning. Head: Global Average Pooling (GAP) $\rightarrow$ Batch Normalization $\rightarrow$ Dropout ($p=0.5$) $\rightarrow$ Dense Softmax ($K=7$). Total parameters: ~25.6M.
2. **EfficientNet-B0 Pipeline:** Pre-trained ImageNet backbone; fine-tuning top MBConv stages. Head: GAP $\rightarrow$ Dropout ($p=0.4$) $\rightarrow$ Dense Softmax ($K=7$). Total parameters: ~5.3M.
3. **EfficientNet-B4 Pipeline:** Pre-trained ImageNet backbone; high-resolution input ($380 \times 380$); fine-tuning top MBConv stages. Head: GAP $\rightarrow$ Dropout ($p=0.4$) $\rightarrow$ Dense Softmax ($K=7$). Total parameters: ~19.3M.

---

## 5.6 Model Training and Hyperparameter Schedules
Training will be conducted under an accelerated GPU environment (NVIDIA CUDA / PyTorch / TensorFlow) with the following protocol:
* **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$).
* **Learning Rate Schedule:** Initial learning rate $\eta_0 = 10^{-4}$ for classification head convergence; reduced to $\eta_{fine} = 10^{-5}$ during backbone fine-tuning. Dynamic decay via `ReduceLROnPlateau` (decay factor $= 0.5$, patience $= 3$ epochs, minimum $\eta = 10^{-7}$).
* **Mini-Batch Size:** 32 samples per batch.
* **Epoch Limit & Regularization:** Maximum 50 epochs with `EarlyStopping` monitoring validation loss with a patience of 8 epochs to prevent overfitting.

---

## 5.7 Comprehensive Evaluation Metrics
To provide rigorous performance assessment under class imbalance, evaluation on the independent test set ($N_{test} = 1,503$) incorporates five complementary metrics:

### 5.7.1 Overall Accuracy
$$\text{Accuracy} = \frac{\sum_{c=1}^{K} TP_c}{N_{test}} \tag{11}$$

### 5.7.2 Class-Specific Precision ($P_c$) and Recall ($R_c$)
$$P_c = \frac{TP_c}{TP_c + FP_c}, \quad R_c = \frac{TP_c}{TP_c + FN_c} \tag{12}$$

### 5.7.3 Macro-Averaged F1-Score
$$F1_c = 2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}, \quad F1_{macro} = \frac{1}{K} \sum_{c=1}^{K} F1_c \tag{13}$$

### 5.7.4 Balanced Accuracy (BA)
$$\text{Balanced Accuracy} = \frac{1}{K} \sum_{c=1}^{K} R_c \tag{14}$$

### 5.7.5 Confusion Matrix Analysis
A $7 \times 7$ contingency matrix mapping ground-truth classifications $y \in \{1, \dots, 7\}$ against model predictions $\hat{y} \in \{1, \dots, 7\}$, visualizing critical false-negative error rates between benign nevi and malignant melanoma.

---

## 5.8 Comparative Benchmarking Framework
Architectures will be benchmarked across three core axes:
1. **Diagnostic Efficacy:** Test Accuracy, Macro F1-Score, and Melanoma-Specific Recall ($R_{MEL}$).
2. **Computational Footprint:** Total parameter volume (M) and serialized disk footprint (MB).
3. **Inference Latency:** Mean latency per frame (ms) evaluated across 500 consecutive iterations on both CPU and GPU.

The architecture achieving the optimal balance of diagnostic sensitivity and computational latency will be serialized for microservice deployment.

---

## 5.9 FastAPI Microservice Architecture
The optimal serialized model binary will be encapsulated within an asynchronous **Python FastAPI** microservice running on a Uvicorn ASGI server.

```text
Client Application (Web / Mobile / Clinic)
                  ↓ [HTTP POST Request with Image Payload]
       FastAPI ASGI Framework Entrypoint
                  ↓
       [/predict RESTful API Endpoint]
                  ↓
   [Pydantic Stream Validation & File Check]
                  ↓
   [Automated Image Preprocessing Pipeline]
    (Bilinear Resize → Normalization → Tensor)
                  ↓
   [Serialized Model Forward Pass Inference]
                  ↓
   [JSON Serialization: Class, Confidence, Latency]
```

### Microservice Endpoint Specification
* **Endpoint:** `POST /predict`
* **Transport:** HTTPS / RESTful
* **Input Payload:** `multipart/form-data` containing image bytes (`image/jpeg`, `image/png`).
* **Output Payload:** `application/json` structured response:

```json
{
  "status": "success",
  "prediction": {
    "class_code": "mel",
    "class_name": "Malignant Melanoma",
    "confidence_score": 0.9412,
    "confidence_percentage": "94.12%"
  },
  "metadata": {
    "model_architecture": "EfficientNet-B4",
    "inference_time_ms": 41.5,
    "disclaimer": "Academic decision-support prototype. Not a substitute for clinical biopsy."
  }
}
```

---

## 5.10 Verified User-Data Retraining Architecture
To enable continuous model refinement while preventing feedback loops of automated label corruption:

```text
               [ Incoming Inference Request ]
                            ↓
             [ FastAPI Prediction Service ]
                            ↓
               [ Return Result to Client ]
                            ↓
         +--------------------------------------+
         | Data Quarantine Repository (S3 / DB) |
         | Image Bytes + SHA-256 Hash + AI Log  |
         +--------------------------------------+
                            ↓
         +--------------------------------------+
         | Clinical Verification Gate           |
         | Board-Certified Dermatologist Review |
         | Histopathological Biopsy Gold Truth  |
         +--------------------------------------+
                 /                      \
     (Verified Gold-Truth)           (Unverified / Corrupt)
              ↓                               ↓
   [Append to Versioned Dataset]      [Purge / Quarantine]
   (Dataset Release V2.0)
              ↓
   [Scheduled Model Retraining]
              ↓
   [Regression Test & Deployment]
   (Model Release V2.0)
```

1. **Quarantine Staging:** User-submitted images and inference logs are written to an isolated cloud staging repository with unique SHA-256 cryptographic hashes.
2. **Clinical Verification Gate:** Samples are barred from retraining pools until certified clinical practitioners or histopathological biopsy records provide verified ground-truth labels.
3. **Audited Dataset and Model Releases:** Validated samples trigger discrete dataset versions ($D_{1.0} \rightarrow D_{2.0}$) and model releases ($M_{1.0} \rightarrow M_{2.0}$) governed by automated regression tests.

---

## 5.11 Technical Stack Specifications

| Architectural Subsystem | Technology Selection | Academic and Engineering Justification |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.10 | Standard ecosystem for machine learning and scientific computing. |
| **Deep Learning Framework** | TensorFlow 2.x / PyTorch | High-performance support for pre-trained CNN backbones and CUDA kernels. |
| **Model Backbones** | ResNet-50, EfficientNet-B0/B4 | Standard benchmarks for residual learning vs. compound scaling with SE attention. |
| **Computer Vision Engine** | OpenCV / Pillow (PIL) | C++ accelerated image decoding, matrix transformation, and color space handling. |
| **Numerical Processing** | NumPy / Pandas | High-speed vectorized array operations and structured dataset manifest indexing. |
| **Evaluation Metrics** | Scikit-Learn | Validated implementations of confusion matrices, F1-scores, and stratified splits. |
| **Serving Microservice** | FastAPI / Uvicorn ASGI | Asynchronous, non-blocking Python web framework delivering sub-50ms inference. |
| **Hardware Infrastructure** | NVIDIA CUDA / Colab GPU | High-throughput hardware acceleration for deep network gradient optimization. |

---

## 5.12 Potential Barriers, Limitations, and Risk Mitigation

1. **Extreme Class Imbalance:**  
   *Barrier:* Skewed majority class distribution biases predictions against rare lethal malignancies.  
   *Mitigation:* Formal implementation and comparative benchmarking of Class-Weighted Cross-Entropy ($\mathcal{L}_{WCE}$) and Focal Loss ($\mathcal{L}_{FL}, \gamma=2.0$).
2. **Domain Shift and Artifact Variance:**  
   *Barrier:* Clinical images captured under non-standardized optical conditions may degrade inference generalizability.  
   *Mitigation:* Conservative training augmentations (rotations, brightness scaling, translations) simulating optical variations, combined with strict input normalization.
3. **Inter-Class Visual Convergence:**  
   *Barrier:* Early-stage melanomas share morphological boundaries with benign atypical nevi.  
   *Mitigation:* Evaluating deep feature representations, prioritizing class-specific recall ($R_{MEL}$), and inspecting confusion matrix misclassifications.
4. **Diagnostic Ethics and Legal Scope:**  
   *Barrier:* Risk of non-specialists interpreting AI prediction outputs as definitive medical diagnoses.  
   *Mitigation:* System outputs are restricted to academic decision-support prototypes, explicitly outputting clinical disclaimers mandating professional biopsy verification.

---

# 6. Timetable or Work-Plan for the Proposed Study

The research project will be executed over a structured **6-Month Schedule**.

### Table 5. Six-Month Project Gantt Work-Plan

| Major Research Task / Milestones | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Literature search and review** | ✓ | ✓ | | | | |
| **2. Fieldwork planning and dataset curation** | ✓ | ✓ | | | | |
| **3. Preprocessing and data augmentation engineering** | | ✓ | ✓ | | | |
| **4. Model development (ResNet vs. EfficientNet)** | | ✓ | ✓ | | | |
| **5. Model training and hyperparameter tuning** | | | ✓ | ✓ | | |
| **6. Data processing and analysis (Evaluation)** | | | | ✓ | ✓ | |
| **7. FastAPI microservice and verified retrain pipeline** | | | | | ✓ | ✓ |
| **8. Reporting (Thesis writing and defense)** | | ✓ | ✓ | ✓ | ✓ | ✓ |

---

### Monthly Key Targets

* **End of Month 1:** Finalized research proposal, dataset requirements established, comprehensive literature review completed.
* **End of Month 2:** HAM10000 dataset curated; stratified 70/15/15 partitions verified; automated preprocessing pipeline operational.
* **End of Month 3:** ResNet-50 and EfficientNet transfer-learning architectures constructed; baseline training pipelines initiated.
* **End of Month 4:** Training experiments across Weighted Cross-Entropy and Focal Loss concluded; hyperparameter optimization finalized.
* **End of Month 5:** Comparative architectural evaluation completed; optimal model serialized; FastAPI prediction microservice implemented.
* **End of Month 6:** Verified-data retraining workflow fully architected; complete bachelor's thesis dissertation authored, proofread, and defended.

---

# 7. Outcomes of the Proposed Research

The proposed research will deliver the following scientific, empirical, and engineering outputs:

1. **Curated & Stratified Dataset Repository:** A cleaned, verified 70/15/15 stratified partition of the HAM10000 dataset across seven diagnostic categories, establishing reproducible benchmarks.
2. **Standardized Preprocessing & Augmentation Pipeline:** Modular Python computer vision modules standardizing raw dermatoscopic images into model-ready tensors while executing clinically grounded augmentation.
3. **Trained Deep Learning Model Binaries:** Serialized neural network weight binaries for ResNet-50, EfficientNet-B0, and EfficientNet-B4 trained under identical optimization regimes.
4. **Loss Function Trade-off Benchmarks:** Definitive comparative empirical data evaluating the convergence dynamics and minority-class recall of Weighted Cross-Entropy versus Focal Loss ($\gamma = 2.0$).
5. **Architectural Performance Metrics Suite:** Multi-metric quantitative evaluation detailing test Accuracy, Precision, Recall, Macro F1-Score, and Balanced Accuracy across all diagnostic categories.
6. **Error Distribution Heatmaps:** Visual confusion matrices highlighting morphological misclassification patterns between benign and malignant lesion categories.
7. **Production Model Serialization:** High-performance, serialized model file optimized for low-latency inference serving.
8. **FastAPI Prediction Microservice:** A functional Python FastAPI web microservice exposing RESTful `/predict` endpoints capable of real-time asynchronous inference with calibrated confidence metrics.
9. **Verified User-Data Retraining Architecture:** An engineered MLOps data governance design establishing quarantine staging and clinical verification gates for continuous model retraining.
10. **Bachelor's Thesis Dissertation:** A comprehensive, academic thesis report detailing the theoretical background, literature critique, experimental methodology, empirical findings, and deployment implications of the proposed system.

```text
                                  [ Research Project Flow ]

    Raw Data        Preprocessing       Deep Learning       Evaluation &        FastAPI & Verified
   Acquisition        & Augmentation        Training         Comparison          Data Deployment
  +-----------+     +--------------+    +--------------+    +--------------+    +-------------------+
  | HAM10000  | --> | OpenCV/Keras | -->| EfficientNet | -->| Metrics, F1, | -->| FastAPI Endpoint  |
  | & ISIC    |     | Resizing &   |    | vs. ResNet   |    | Confusion    |    | & Retrain Loop    |
  | Datasets  |     | Normalization|    | Training     |    | Matrix Plots |    | Design            |
  +-----------+     +--------------+    +--------------+    +--------------+    +-------------------+
```

---

# 8. Bibliography

Codella, N. C., Gutman, D., Celebi, M. E., Helba, B., Marchetti, M. A., Dusza, S. W., Kalloo, A., Liopyris, K., Mishra, N., Kittler, H., & Halpern, A. (2018). Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), hosted by the International Skin Imaging Collaboration (ISIC). *2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018)*, 168–172. https://doi.org/10.1109/ISBI.2018.8363547

Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin lesions with deep neural networks. *Nature*, 542(7639), 115–118. https://doi.org/10.1038/nature21056

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. https://doi.org/10.1109/CVPR.2016.90

Kassem, M. A., Hosny, K. M., & Fouad, M. M. (2020). Skin lesions classification into eight classes for ISIC 2019 using deep convolutional neural network. *IEEE Access*, 8, 114822–114832. https://doi.org/10.1109/ACCESS.2020.3003890

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444. https://doi.org/10.1038/nature14539

Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988. https://doi.org/10.1109/ICCV.2017.324

Ramirez, A., Chen, Y., & Williams, P. (2023). Deploying machine learning microservices in healthcare environments: Performance and scaling evaluation of FastAPI. *Journal of Healthcare Informatics Research*, 7(3), 312–329. https://doi.org/10.1007/s41666-023-00142-w

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 1–48. https://doi.org/10.1186/s40537-019-0197-0

Srinivasu, P. N., Sivasankar, G., Ijaz, M. F., Bhoi, A. K., Kim, W., & Kang, J. J. (2021). Classification of skin disease using MobileNet V2 and EfficientNet with deep learning techniques. *IEEE Access*, 9, 84381–84393. https://doi.org/10.1109/ACCESS.2021.3087204

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning (ICML)*, PMLR, 6105–6114. https://proceedings.mlr.press/v97/tan19a.html

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5(1), 180161. https://doi.org/10.1038/sdata.2018.161
