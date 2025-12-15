# **Architectural Synthesis for Multi-Modal Dermatological Diagnosis: A Comprehensive Strategy for the MILK10k Benchmark**

## **1\. Introduction: The Paradigm Shift in Automated Dermatopathology**

The domain of computational dermatology has undergone a profound transformation over the last decade. Early initiatives, typified by the International Skin Imaging Collaboration (ISIC) challenges of 2016 through 2018, focused predominantly on the classification of dermoscopic images—highly standardized, polarized, and magnified views of skin lesions. While these benchmarks advanced the state of Convolutional Neural Networks (CNNs) in identifying morphological patterns such as pigment networks and globules, they largely ignored the macroscopic clinical context in which dermatologists operate. A clinician does not diagnose a lesion in a vacuum; they integrate the "ugly duckling" sign from the patient's total body map, the textural cues from a clinical close-up, and the patient’s metadata.

The MILK10k challenge represents the culmination of this shift towards **holistic, multi-modal diagnosis**. By mandating the integration of paired clinical and dermatoscopic imagery with high-fidelity metadata—specifically the semantic probability scores derived from the MONET framework—this benchmark moves beyond pattern recognition into the realm of semantic reasoning.1 The task is no longer simply to classify a texture but to synthesize disparate signals: the microscopic architecture of a melanocyte nest, the macroscopic irregularity of the lesion's border on the patient's arm, and the semantic probability that the lesion contains ulceration or distinct vascular structures.

Furthermore, the introduction of the **Macro F1 Score** as the primary evaluation metric fundamentally alters the optimization landscape. Unlike Accuracy or ROC-AUC, which can be maximized by performing well on majority classes (such as Nevi or Benign Keratosis), Macro F1 treats the rarest class (e.g., Dermatofibroma or Vascular Lesion) with equal importance to the most common. This imposes a strict penalty on models that neglect the "long tail" of the diagnostic distribution, necessitating advanced loss engineering and sampling strategies.

This report leverages the availability of an NVIDIA A100 GPU with 80GB of VRAM to propose a computational strategy that transcends the limitations of resource-constrained environments. We are not bound by the need to use lightweight models like MobileNet or ResNet-18. Instead, we can deploy massive, pre-trained Foundation Models such as **PanDerm** (and its successor **DermLIP**), fuse them via computationally expensive cross-attention mechanisms, and train them with large batch sizes that stabilize volatile metrics like the Soft F1 loss. The following analysis deconstructs the MILK10k challenge into its constituent modalities, evaluates the theoretical underpinnings of candidate architectures, and prescribes a unified "Tri-Modal PanDerm Fusion" system designed to maximize the Macro F1 Score.

## ---

**2\. Deconstructing the MILK10k Multi-Modal Signal**

To engineer an optimal architecture, one must first understand the distinct statistical and semantic properties of the input data. The MILK10k dataset is unique in its provision of a tripartite input vector for each of the 5,240 lesions: the clinical view, the dermoscopic view, and the semantic metadata.1

### **2.1 The Dichotomy of Dermatological Imagery**

The distinction between clinical close-ups and dermatoscopic images is not merely one of magnification; it is a difference in physics and information content.

**Dermatoscopic Imagery** is obtained using a dermoscope, which typically employs polarized light and fluid immersion to eliminate surface reflection (glare) from the stratum corneum. This allows the observer—and the neural network—to visualize subsurface structures located in the epidermis and the papillary dermis.3 The visual features relevant here are high-frequency textures: pigment networks, dots/globules, streaks, and blue-white veils. From a Deep Learning perspective, these images are translationally invariant and heavily texture-dependent. Convolutional Neural Networks (CNNs) like **EfficientNetV2** have historically excelled here because their inductive bias (local connectivity and weight sharing) aligns perfectly with the detection of these local textures.4

**Clinical Close-up Imagery**, conversely, is a standard macroscopic photograph. It captures the lesion's 3D topography (elevation, nodularity), its relationship to the surrounding skin (erythema halo), and surface features like crusting or scaling that might be flattened by a dermoscope's contact plate. However, clinical images are fraught with noise: variable lighting conditions, shadows, background artifacts (clothing, rulers), and varying distances from the camera. A model trained on clinical images must be robust to **domain shift** and capable of extracting global context rather than just local texture. Vision Transformers (ViTs), with their global attention mechanisms, are theoretically superior for this modality as they can integrate information from the entire field of view to determine context, mitigating the impact of local artifacts.5

The architectural implication is immediate: a "Early Fusion" strategy, where clinical and dermoscopic images are concatenated into a 6-channel tensor and passed through a single backbone, is suboptimal. The statistical distribution of pixel intensities and the nature of the features (texture vs. shape) are too divergent. The network would struggle to learn a unified set of filters that apply equally well to both. Instead, a **Siamese or Dual-Stream architecture** is required, where each modality is processed by a dedicated encoder initialized with domain-specific weights.

### **2.2 The MONET Framework: "Privileged" Semantic Information**

The most novel aspect of the MILK10k challenge is the inclusion of metadata derived from the **MONET (Medical cONcept rETriever)** framework.6 MONET is a foundation model trained on over 100,000 dermatology image-text pairs derived from medical literature. It learns to associate visual patterns with high-level semantic concepts.

The dataset provides probability scores for concepts such as:

* **Ulceration/Crust:** A strong predictor of malignancy in Melanoma and Basal Cell Carcinoma.  
* **Vasculature/Vessels:** Critical for distinguishing vascular lesions and identifying neo-angiogenesis in tumors.  
* **Erythema:** Indicates inflammation, relevant for diagnosing Lichen Planus or differentiating irritated nevi.  
* **Pigmentation:** Distinguishes melanocytic from non-melanocytic lesions.  
* **Artifacts (Gel, Pen Ink):** Noise indicators that the model should ideally learn to ignore.

In traditional deep learning pipelines, metadata is often treated as a second-class citizen, appended to the final feature vector just before the softmax classifier. This "Late Fusion" approach is insufficient for the MONET scores. These scores are not just attributes; they are **attention guides**. If the MONET score for "Ulceration" is 0.9, the visual encoder should ideally shift its attention to focus on regions of the image that exhibit crusting or erosion. If the "Pen Ink" score is high, the model should down-weight features associated with purple/blue artificial pigments.

This suggests an architectural requirement for **Cross-Modal Attention**, where the semantic embedding of the MONET scores is used to query the visual features, effectively gating the information flow based on the semantic prior. The **SkinM2Former** architecture 8 provides a blueprint for this interaction, utilizing a Tri-Modal Cross-Attention Transformer (TMCT) to facilitate deep interaction between image features and metadata tokens.

## ---

**3\. Foundation Model Analysis: The PanDerm Ecosystem**

Given the availability of an A100 GPU, training a model from scratch is an inefficient use of resources. The sheer volume of unlabeled dermatological data available globally has given rise to Domain-Specific Foundation Models. Among these, **PanDerm** represents the current zenith of pre-training sophistication.

### **3.1 The Architecture of PanDerm**

PanDerm is a multi-modal foundation model pre-trained on a massive corpus of over 2 million skin disease images sourced from 11 clinical institutions.9 Its significance lies in its training methodology and data diversity. Unlike models pre-trained on ImageNet, which learn to distinguish dogs from cats, PanDerm has learned the subtle differentiation between a dysplastic nevus and a melanoma in situ.

The core architecture of PanDerm typically utilizes a **ViT-Large (Vision Transformer)** backbone.9 The choice of a Transformer over a CNN is pivotal. Transformers, which rely on Self-Attention mechanisms ($\\text{Attention}(Q, K, V) \= \\text{softmax}(\\frac{QK^T}{\\sqrt{d\_k}})V$), have a global receptive field from the very first layer. This allows PanDerm to learn long-range dependencies in the image—for example, relating a satellite metastasis at the edge of an image to the primary tumor in the center—which CNNs can only achieve deep in the network hierarchy.

PanDerm's pre-training objective involves **Self-Supervised Learning (SSL)**, specifically a combination of Masked Latent Modeling and CLIP-based feature alignment.11

* **Masked Latent Modeling:** The model learns to reconstruct missing parts of the image representation, forcing it to internalize the statistical regularities of skin lesions.  
* **CLIP Alignment:** It aligns the image embeddings with text embeddings derived from diagnostic captions. This is crucial for the MILK10k task because it means the PanDerm weights are already primed to understand the "language" of dermatology, making the integration of MONET scores (which are text-derived concepts) far more natural.

### **3.2 DermLIP: The Vision-Language Successor**

Recent advancements have led to **DermLIP**, a refined iteration of the PanDerm lineage.12 DermLIP is explicitly designed as a Vision-Language Model, utilizing a PanDerm-Base vision encoder aligned with a **PubMedBERT** text encoder.

* **Hugging Face Integration:** The availability of weights such as redlessone/DermLIP\_PanDerm-base-w-PubMed-256 13 allows for direct integration into PyTorch pipelines.  
* **Performance Delta:** Benchmarks indicate that DermLIP outperforms standard BiomedCLIP and MONET on zero-shot classification tasks by significant margins (+14.7% accuracy).12  
* **Relevance to MILK10k:** By using the DermLIP vision encoder weights, we are initializing our model with a feature extractor that has already been optimized to maximize the mutual information between visual patterns and medical concepts. This provides a "head start" in learning the mapping from pixels to the semantic categories implicit in the MONET scores.

### **3.3 The Baseline Competitor: EfficientNetV2**

While PanDerm offers semantic superiority, the **EfficientNetV2** family 4 remains a formidable baseline and a necessary component of any ensemble. EfficientNetV2 is a CNN optimized for training speed and parameter efficiency. It utilizes Fused-MBConv blocks to reduce memory access overhead.

* **The Inductive Bias Argument:** CNNs possess a strong inductive bias for locality. In dermoscopy, where the diagnostic signal often lies in high-frequency local textures (e.g., "peppering" or "angulated lines"), CNNs can sometimes outperform Transformers which have to learn these local relationships from scratch.  
* **Robustness:** Analysis of ISIC 2024 top solutions reveals that EfficientNet ensembles are incredibly robust and stable.14 They are less prone to catastrophic overfitting on small datasets compared to ViTs.  
* **Role in MILK10k:** EfficientNetV2 should serve as the "Texture Specialist" in our architectural strategy, complementing the "Semantic/Global Specialist" role of the PanDerm ViT.

## ---

**4\. Architectural Strategy: The Tri-Modal PanDerm Fusion**

We propose a bespoke architecture designed specifically to exploit the A100's capacity and the MILK10k's multi-modal structure. We term this the **Tri-Modal PanDerm Fusion Network**. This architecture fuses the Dual-Stream PanDerm backbones with the MONET concepts using a sophisticated cross-attention mechanism inspired by **SkinM2Former**.8

### **4.1 Backbone: Dual-Stream Foundation Encoders**

We instantiate two parallel encoders, $E\_{derm}$ and $E\_{clin}$, both initialized with the DermLIP\_PanDerm weights.

$$f\_{derm} \= E\_{derm}(I\_{derm}) \\in \\mathbb{R}^{L \\times D}$$

$$f\_{clin} \= E\_{clin}(I\_{clin}) \\in \\mathbb{R}^{L \\times D}$$  
where $I\_{derm}, I\_{clin} \\in \\mathbb{R}^{3 \\times 224 \\times 224}$ are the input images, $L$ is the number of tokens (197 for a standard ViT-B/16 including the CLS token), and $D$ is the embedding dimension (e.g., 768 or 1024 for ViT-L).

Crucial Implementation Detail:  
With 80GB VRAM, we do not need to freeze these backbones entirely. However, to preserve the foundation knowledge, we employ Layer-Wise Learning Rate Decay. The earliest layers (closer to the input) are assigned a very low learning rate ($10^{-6}$), while the deeper layers and the fusion module are assigned higher rates ($10^{-4}$). This allows the model to adapt its high-level semantic representation to the MILK10k classes without destroying the low-level Gabor-like filters learned during pre-training.

### **4.2 The Semantic Bridge: Encoding MONET Scores**

The 11 MONET probability scores are not treated as scalars but as **Concept Embeddings**. We pass the vector of scores $S\_{monet} \\in \\mathbb{R}^{11}$ through a specific projection network.

$$E\_{concept} \= \\text{MLP}\_{proj}(S\_{monet}) \\in \\mathbb{R}^{K \\times D}$$  
Here, we can project the scores into $K$ tokens (where $K$ could be 1, treating all scores as a single context vector, or 11, treating each score as a distinct token). Treating them as 11 distinct tokens allows the attention mechanism to attend to specific concepts (e.g., focusing solely on "Ulceration").

### **4.3 The Fusion Core: Tri-Modal Cross-Attention (TMCT)**

The fusion module is the heart of the architecture. We adapt the **Tri-Modal Cross-Attention Transformer (TMCT)** proposed in SkinM2Former 8 and the fusion concepts from **HybridSkinFormer**.5

The fusion occurs in three stages of interaction:

Stage 1: View Alignment (Spatial Correspondence)  
The clinical and dermoscopic images capture the same lesion but with different coordinate systems and deformations. We use Cross-Attention to align them.

$$Q \= f\_{derm}, \\quad K \= f\_{clin}, \\quad V \= f\_{clin}$$

$$f\_{derm \\leftarrow clin} \= \\text{MultiHeadAttention}(Q, K, V) \+ f\_{derm}$$

This operation enriches the dermoscopic features with clinical context. For instance, if the clinical image shows the lesion is on sun-damaged skin (solar elastosis), this context is injected into the dermoscopic representation.  
Stage 2: Semantic Gating (Concept Injection)  
We use the MONET embeddings to "gate" or highlight relevant visual features.

$$Q \= f\_{derm \\leftarrow clin}, \\quad K \= E\_{concept}, \\quad V \= E\_{concept}$$

$$f\_{fused} \= \\text{MultiHeadAttention}(Q, K, V) \+ f\_{derm \\leftarrow clin}$$

This step effectively asks the question: "Given that the MONET score says 'Ulceration' is likely, which visual patches support this?" The attention mechanism will assign high weights to patches that semantically correlate with the MONET concepts.  
Stage 3: Global Context Pooling  
Instead of a simple CLS token or Global Average Pooling, we utilize the Residual-Learnable Multi-head Attention module from HybridSkinFormer.5 This module applies a learnable query to summarize the entire sequence of fused tokens into a single global descriptor vector.

$$f\_{global} \= \\text{GlobalAttention}(f\_{fused})$$

### **4.4 Metadata Injection: The "Tabular-First" Insight**

While MONET scores are used for attention, the standard patient metadata (Age, Sex, Anatomical Site) acts as a prior. The winner of the ISIC 2024 challenge 14 demonstrated that Gradient Boosted Decision Trees (GBDTs) like XGBoost or CatBoost often handle this type of discrete/tabular data better than Neural Networks.

Therefore, we propose a **Hybrid Stacking Head**:

1. The Deep Learning model outputs a feature vector $f\_{global}$.  
2. We concatenate $f\_{global}$ with the raw metadata.  
3. This combined vector is passed to the final MLP classifier.  
4. *Simultaneously*, we train an XGBoost model on the metadata \+ the frozen logits of the DL model. The final prediction is a weighted average of the DL Head and the XGBoost Head.

## ---

**5\. Optimization for the Macro F1 Metric**

The choice of loss function is critical. The Macro F1 score is the "Great Equalizer"—it punishes performance on the majority class if the minority class is neglected. Standard Cross-Entropy (CE) loss is fundamentally misaligned with this goal because it optimizes for likelihood, which is dominated by frequent classes.

### **5.1 The "Soft F1" Loss Function**

The F1 score is defined as:

$$F1 \= \\frac{2TP}{2TP \+ FP \+ FN}$$

This is non-differentiable because $TP$ (True Positives), $FP$, and $FN$ are discrete counts derived from a thresholded step function. To make this differentiable for Backpropagation, we must smooth the step function into a sigmoid probability.16  
We define the Soft F1 Loss for a specific class $c$ over a batch of $B$ samples:  
$$ \\mathcal{L}{SoftF1}^{(c)} \= 1 \- \\frac{2 \\sum{i=1}^{B} p\_{i,c} y\_{i,c}}{\\sum\_{i=1}^{B} p\_{i,c} \+ \\sum\_{i=1}^{B} y\_{i,c} \+ \\epsilon} $$  
where $p\_{i,c}$ is the predicted probability and $y\_{i,c}$ is the ground truth (0 or 1).  
The total loss is the average over all classes (Macro):

$$\\mathcal{L}\_{MacroSoftF1} \= \\frac{1}{C} \\sum\_{c=1}^{C} \\mathcal{L}\_{SoftF1}^{(c)}$$  
**Hardware Implication:** The Soft F1 loss is a *batch-level* metric. It requires accurate statistics of $TP, FP, FN$ within the batch. If the batch size is too small (e.g., 8 or 16), the estimation of the F1 score is noisy and gradients fluctuate wildly. The A100 80GB allows us to push the batch size to **64, 96, or even 128** (using gradient accumulation or mixed precision). This large batch size is the single most important hardware-enabled advantage for optimizing this specific metric.

### **5.2 The Compound Loss Strategy**

Pure Soft F1 loss can be unstable, especially early in training when predictions are random. It can get stuck in local minima where the model predicts "all zeros" or "all ones". To stabilize training, we combine it with **Weighted Focal Loss**.

$$\\mathcal{L}\_{Total} \= \\lambda\_1 \\mathcal{L}\_{Focal} \+ \\lambda\_2 \\mathcal{L}\_{MacroSoftF1}$$  
Focal Loss 5 focuses on "hard" examples (those with low probability of the correct class):

$$\\mathcal{L}\_{Focal} \= \-\\alpha\_t (1 \- p\_t)^\\gamma \\log(p\_t)$$

We set $\\gamma \= 2.0$ to penalize hard misclassifications and $\\alpha\_t$ to be the inverse class frequency (weighting rare classes higher). This ensures that the model learns discriminative features (via Focal Loss) while simultaneously optimizing the target metric (via Soft F1).

### **5.3 Auxiliary Losses**

Following the 2nd place solution from ISIC 2024 15, we employ Auxiliary Loss Heads. We attach classifiers to the intermediate outputs of the Dermoscopy Encoder ($E\_{derm}$) and Clinical Encoder ($E\_{clin}$) before fusion.  
$$ \\mathcal{L}{Final} \= \\mathcal{L}{Total}(Fused) \+ 0.3 \\mathcal{L}{Total}(Derm) \+ 0.3 \\mathcal{L}{Total}(Clin) $$  
This "Deep Supervision" ensures that the individual encoders learn meaningful representations even before the complex fusion stage, facilitating faster convergence and acting as a regularizer against overfitting.

## ---

**6\. Implementation Strategy on the A100**

The NVIDIA A100 80GB is a distinct class of hardware that permits strategies impossible on consumer cards (like the RTX 3090/4090).

### **6.1 Memory-Centric Optimizations**

1\. Brain Floating Point (BF16):  
The A100 supports torch.bfloat16. Unlike standard FP16, BF16 has the same exponent range as FP32. This is crucial for training large Transformers like PanDerm ViT-L. In FP16, the gradients for ViT-L often underflow or overflow, requiring complex "Loss Scaling." BF16 avoids this instability entirely, allowing for robust training of massive models without numerical divergence.  
2\. Activation Checkpointing:  
Even with 80GB, two ViT-Large models \+ Fusion \+ Large Batch Size might pressure memory. We can use torch.utils.checkpoint to trade compute for memory. By not storing intermediate activations during the forward pass and recomputing them during the backward pass, we can effectively double the maximum batch size. Given the A100's massive compute throughput (Tensor Cores), the time penalty is negligible compared to the stability gain of larger batches for the Soft F1 loss.

### **6.2 Data Pipeline and Augmentation**

The CPU is often the bottleneck when the GPU is this fast. We must use **FFCV (Fast Forward Computer Vision)** or **NVIDIA DALI** to load and augment images directly on the GPU.

* **Dermoscopic Augmentation:** Rotation ($0-360^\\circ$), Vertical/Horizontal Flip (invariant features).  
* **Clinical Augmentation:** Color Jitter (Brightness/Contrast), Gaussian Blur (simulating out-of-focus shots), and Cutout/CoarseDropout (forcing the model to look at context, not just the lesion center).  
* **Test-Time Augmentation (TTA):** During inference, we perform 16x TTA (4 rotations $\\times$ 4 crops) for every image. The A100's throughput ensures this can be done within the inference time limits of the challenge.

## ---

**7\. Ablation and Ensemble Strategy**

To secure a top position, one relies on the "Wisdom of Crowds" (Ensembling). We construct three diverse models.

### **7.1 Model A: The Foundation (Tri-Modal PanDerm)**

* **Backbone:** DermLIP (ViT-L).  
* **Focus:** Semantic understanding, global context.  
* **Strength:** Best at handling complex/ambiguous cases where metadata is key.

### **7.2 Model B: The Texture Specialist (EfficientNetV2 Ensemble)**

* **Backbone:** EfficientNetV2-XL.4  
* **Focus:** High-frequency texture details.  
* **Input:** Concatenated images (Early Fusion).  
* **Strength:** Extremely robust on standard presentations; catches texture cues ViT might miss.

### **7.3 Model C: The Tabular Expert (XGBoost)**

* **Input:** Patient Metadata \+ MONET Scores \+ **Embeddings extracted from Model A**.  
* **Logic:** We extract the $D$-dimensional feature vector from the PanDerm model (frozen) and feed it into an XGBoost classifier.  
* **Strength:** GBDTs are mathematically superior at handling discrete/categorical variables and finding decision thresholds on tabular data.14

Final Prediction:

$$P\_{final} \= w\_1 P\_{PanDerm} \+ w\_2 P\_{EfficientNet} \+ w\_3 P\_{XGBoost}$$

The weights $w\_1, w\_2, w\_3$ are found via Nelder-Mead optimization on the validation set to maximize the Macro F1 score.

## ---

**8\. Conclusion**

The MILK10k challenge is a test of a system's ability to integrate discordant information channels. A naive approach treating it as a simple image classification task will fail due to the complexity of the input (paired multi-view images) and the strictness of the metric (Macro F1).

The proposed solution leverages the **A100 80GB** to deploy the most advanced tools available in 2024/2025:

1. **Foundation Intelligence:** Using **PanDerm/DermLIP** to initialize the visual system with millions of pre-learned dermatological concepts.  
2. **Semantic Fusion:** Implementing **Tri-Modal Cross-Attention** to allow MONET scores to actively guide visual processing, rather than passively sitting as appended features.  
3. **Metric-Aware Optimization:** Utilizing the A100's memory to support large-batch **Soft F1** training, directly attacking the competition metric.  
4. **Robust Ensembling:** combining the semantic depth of Transformers with the textural precision of CNNs and the tabular dominance of Gradient Boosting.

This architecture moves beyond "Black Box" diagnosis towards a system that aligns visual evidence with semantic medical concepts, offering the highest probability of maximizing the Macro F1 Score in the MILK10k benchmark.

### ---

**Data Tables and Comparisons**

**Table 1: Comparative Analysis of Architectural Candidates**

| Feature | EfficientNetV2 (Baseline) | Swin Transformer | PanDerm (Proposed) |
| :---- | :---- | :---- | :---- |
| **Inductive Bias** | Local (CNN). Good for texture. | Hierarchical (Window Attention). | Global (Self-Attention). Good for context. |
| **Pre-training** | ImageNet (Natural Images). | ImageNet-22k. | **2.1M Skin Images (PanDerm).** |
| **Multi-Modal Native** | No. Requires concatenation. | No. | **Yes.** Trained on Clinical \+ Dermoscopic. |
| **Metadata Integration** | Concatenation (Late Fusion). | Concatenation. | **Cross-Attention (Deep Fusion).** |
| **Macro F1 Potential** | Moderate. Struggles with rare classes. | High. | **Highest.** Semantic alignment aids rare classes. |
| **Hardware Fit** | Low VRAM. (Under-utilizes A100). | High VRAM. | **Max VRAM.** Fully utilizes 80GB capacity. |

**Table 2: Recommended Hyperparameters for A100 Training**

| Parameter | Value | Rationale |
| :---- | :---- | :---- |
| **Batch Size** | 96 \- 128 | Stabilizes Soft F1 Loss statistics; maximizes GPU saturation. |
| **Precision** | bfloat16 | Stability of FP32 with memory of FP16. No loss scaling needed. |
| **Optimizer** | AdamW | Standard for Transformers. |
| **Learning Rate** | $1e-5$ (Backbone), $1e-3$ (Head) | Layer-wise decay preserves foundation knowledge. |
| **Weight Decay** | 0.05 | Prevents overfitting in the massive parameter space of ViT-L. |
| **Loss Function** | Focal ($\\gamma=2$) \+ Soft F1 | Hybrid loss to target metric and handle imbalance. |
| **Augmentation** | RandAugment (N=2, M=9) | Heavy augmentation prevents memorization of the 5k samples. |

**Table 3: MONET Concept Mapping Strategy**

| MONET Concept | Diagnostic Relevance (Example) | Attention Mechanism Goal |
| :---- | :---- | :---- |
| **Ulceration** | Melanoma, BCC | Attend to: Crusting, erosion, bleeding. |
| **Blue/White Veil** | Invasive Melanoma | Attend to: Central blue-grey structures. |
| **Vessels** | BCC, Hemangioma | Attend to: Arborizing or comma-shaped vessels. |
| **Pigment Network** | Melanocytic Lesions | Attend to: Reticular patterns at periphery. |
| **Pen Ink / Gel** | Artifact | **Negative Attention:** Ignore these regions. |

## ---

**9\. Failure Mode Analysis and Mitigation**

Even with a state-of-the-art architecture, specific failure modes are predictable given the dataset constraints.

Failure Mode 1: The "Clinical Dominance" Trap.  
Deep Learning models often latch onto clinical images because they contain background cues (shortcuts). For example, if most malignant cases are photographed with a specific ruler or background color, the model will learn the ruler, not the lesion.

* *Mitigation:* **Modality Dropout.** During training, randomly zero-out the clinical image tensor (e.g., 20% of the time). This forces the model to rely on the dermoscopic view and the metadata to make the prediction, preventing over-reliance on clinical artifacts.

Failure Mode 2: The "Rare Class" Collapse.  
Classes like Dermatofibroma or Squamous Cell Carcinoma might be represented by fewer than 100 samples. The model might converge to a state where it simply never predicts these classes to minimize standard error.

* *Mitigation:* **Oversampling & Logit Adjustment.** Use a WeightedRandomSampler to ensure every batch contains these classes. Additionally, applying Post-Hoc Logit Adjustment (subtracting $\\log(P\_{class})$ from the logits) can rebalance the decision boundaries in favor of rare classes.

Failure Mode 3: Metadata Overfitting.  
The MONET scores are powerful, but if the model learns to rely solely on them (ignoring the images), it will fail on the test set if the MONET scores in the test set have slightly different distributions or calibration.

* *Mitigation:* **Concept Dropout.** Randomly zero-out specific MONET scores during training. This ensures the model learns to find the visual evidence itself, using the MONET score only as a confirmation or guide, rather than a crutch.

By proactively addressing these failure modes within the architectural design, the system becomes robust not just to the training data, but to the generalization gap inherent in the public-to-private leaderboard transition.

#### **Nguồn trích dẫn**

1. MILK10k Benchmark \- ISIC Challenge, truy cập vào tháng 12 8, 2025, [https://challenge.isic-archive.com/landing/milk10k/](https://challenge.isic-archive.com/landing/milk10k/)  
2. skinCancerMilk10k \- Kaggle, truy cập vào tháng 12 8, 2025, [https://www.kaggle.com/datasets/able23/skincancermilk10k](https://www.kaggle.com/datasets/able23/skincancermilk10k)  
3. Artificial Intelligence and New Technologies in Melanoma Diagnosis: A Narrative Review, truy cập vào tháng 12 8, 2025, [https://www.mdpi.com/2072-6694/17/24/3896](https://www.mdpi.com/2072-6694/17/24/3896)  
4. Overview of image category in the ISIC-2024 dataset. \- ResearchGate, truy cập vào tháng 12 8, 2025, [https://www.researchgate.net/figure/Overview-of-image-category-in-the-ISIC-2024-dataset\_fig1\_391247171](https://www.researchgate.net/figure/Overview-of-image-category-in-the-ISIC-2024-dataset_fig1_391247171)  
5. An Ingeniously Designed Skin Lesion Classification Model Across ..., truy cập vào tháng 12 8, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12385535/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12385535/)  
6. ConceptCLIP: Towards Trustworthy Medical AI via Concept-Enhanced Contrastive Langauge-Image Pre-training \- arXiv, truy cập vào tháng 12 8, 2025, [https://arxiv.org/html/2501.15579v1](https://arxiv.org/html/2501.15579v1)  
7. Transparent medical image AI via an image-text foundation model grounded in medical literature \- PubMed, truy cập vào tháng 12 8, 2025, [https://pubmed.ncbi.nlm.nih.gov/38627560/](https://pubmed.ncbi.nlm.nih.gov/38627560/)  
8. A Novel Perspective for Multi-Modal Multi-Label ... \- CVF Open Access, truy cập vào tháng 12 8, 2025, [https://openaccess.thecvf.com/content/WACV2025/papers/Zhang\_A\_Novel\_Perspective\_for\_Multi-Modal\_Multi-Label\_Skin\_Lesion\_Classification\_WACV\_2025\_paper.pdf](https://openaccess.thecvf.com/content/WACV2025/papers/Zhang_A_Novel_Perspective_for_Multi-Modal_Multi-Label_Skin_Lesion_Classification_WACV_2025_paper.pdf)  
9. A Multimodal Vision Foundation Model for Clinical Dermatology \- arXiv, truy cập vào tháng 12 8, 2025, [https://arxiv.org/html/2410.15038v2](https://arxiv.org/html/2410.15038v2)  
10. A multimodal vision foundation model for clinical dermatology \- ResearchGate, truy cập vào tháng 12 8, 2025, [https://www.researchgate.net/publication/392472819\_A\_multimodal\_vision\_foundation\_model\_for\_clinical\_dermatology](https://www.researchgate.net/publication/392472819_A_multimodal_vision_foundation_model_for_clinical_dermatology)  
11. A General-Purpose Multimodal Foundation Model for Dermatology \- arXiv, truy cập vào tháng 12 8, 2025, [https://arxiv.org/html/2410.15038v1](https://arxiv.org/html/2410.15038v1)  
12. Derm1M: A Million‑Scale Vision‑Language Dataset Aligned with Clinical Ontology Knowledge for Dermatology \- GitHub, truy cập vào tháng 12 8, 2025, [https://github.com/SiyuanYan1/Derm1M](https://github.com/SiyuanYan1/Derm1M)  
13. redlessone/DermLIP\_ViT-B-16 \- Hugging Face, truy cập vào tháng 12 8, 2025, [https://huggingface.co/redlessone/DermLIP\_ViT-B-16](https://huggingface.co/redlessone/DermLIP_ViT-B-16)  
14. 1st Place Solution | Kaggle, truy cập vào tháng 12 8, 2025, [https://www.kaggle.com/competitions/isic-2024-challenge/writeups/ilya-novoselskiy-1st-place-solution](https://www.kaggle.com/competitions/isic-2024-challenge/writeups/ilya-novoselskiy-1st-place-solution)  
15. 2nd Place Solution | Kaggle, truy cập vào tháng 12 8, 2025, [https://www.kaggle.com/competitions/isic-2024-challenge/writeups/yakiniku-2nd-place-solution](https://www.kaggle.com/competitions/isic-2024-challenge/writeups/yakiniku-2nd-place-solution)  
16. sigmoidF1: A Smooth F1 Score Surrogate Loss for Multi- label Classification \- arXiv, truy cập vào tháng 12 8, 2025, [https://arxiv.org/pdf/2108.10566](https://arxiv.org/pdf/2108.10566)