# Machine Unlearning on Spoken Language Understanding

## 📖 Overview
This repository contains the detailed experimental results for the paper **"Alexa, can you *forget* me?” Machine Unlearning Benchmark on Spoken Language Understanding**, accepted at INTERSPEECH 2025, and of **"UnSLU-BENCH+: Extended Machine Unlearning Benchmark for Spoken Language Understanding"**, published at IEEE Transactions on Audio, Speech and Language Processing (TASLP) in 2026. 

**CGUM update.** With respect to the initial version, the GUM metric has been updated in the journal paper, and the new CGUM implementation is available in the notebook [CGUM.ipynb](CGUM.ipynb). For future experimentation, we recommend using CGUM instead of GUM.

[![paper](https://img.shields.io/badge/Paper-Interspeech-green)](https://www.isca-archive.org/interspeech_2025/koudounas25c_interspeech.pdf)
[![paper TASLP](https://img.shields.io/badge/Paper-TASLP-green)](https://ieeexplore.ieee.org/abstract/document/11447421)
[![SLU-models-collection](https://img.shields.io/badge/SLU_Models_Collection-HuggingFace-red)](https://huggingface.co/collections/alkiskoudounas/slu-models-67bcb156245d12b6b08bf2f2)
[![UNLEARNING-models-collection](https://img.shields.io/badge/UnSLU_Models_Collection_(GOLD)-HuggingFace-yellow)](https://huggingface.co/collections/alkiskoudounas/unslu-bench-68304d8647fb9b0c72533066)

## 🔗 Table of Contents
- [⚙️ Experimental Setup](#⚙️-experimental-setup)
- [📊 Detailed Results](#📊-detailed-results)
  - [1. Comparison of unlearning methods](#1-comparison-of-unlearning-methods)
    - [A. FSC](#a-fsc)
    - [B. SLURP*](#b-slurp)
    - [C. ITALIC](#c-italic)
    - [D. SpeechMASSIVE (DE)](#d-speechmassive-de)
    - [E. SpeechMASSIVE (FR)](#e-speechmassive-fr)
  - [2. Best Learning Rate (LR) for Unlearning Methods](#2-best-learning-rate-lr-for-unlearning-methods)
- [📜 License](#📜-license)
- [📧 Contact](#📧-contact)
- [📄 Citation](#📄-citation)

## ⚙️ Experimental Setup

**Transformer models.** For English datasets (FSC and SLURP), we use [wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base) and [HuBERT](https://huggingface.co/facebook/facebook-base-ls960) models, while for ITALIC and SpeechMASSIVE we employ the multilingual [XLS-R-128](https://huggingface.co/facebook/wav2vec2-xls-r-300m) and XLS-R-53 models, with the latter ASR fine-tuned for the target language (i.e., [Italian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-italian), [German](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german),
[French](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french)).
On FSC, the models are trained for 2800 steps, with 8 batch size, 10% warmup steps, 4 gradient accumulation steps, 5e-4 initial learning rate, AdamW with weight decay, plateau scheduler, and early stopping criterion. SLURP follows the same configuration, but training lasts for 30000 steps.
On ITALIC and SpeechMASSIVE, both XLS-R-128 and specialized XLS-R-53 models are trained with the same configuration, but for 15000 total steps, 1e-4 initial learning rate, and 1 gradient accumulation step.

**Unlearning methods.** All unlearning methods were tested for a single epoch and with the hyperparameters recommended in the original papers. For example, in cf-k, the entire network was frozen except the last layer (cf-1). With SCRUB, a temperature T = 4.0 was used, and with Bad Teaching, the KL temperature is 1. In UNSIR, the learning rate of the noise was set to 0.01. AdamW was used as the optimizer for all methods to remain compliant with the training. 

**Datasets.** It is possible to obtain the datasets used in the experiments by running the relative notebooks present in the `unlearning_datasets_creation` folder. Once the datasets are created, they can be found in the `data_name` folder, where name is the name of the dataset. The datasets are **FSC** (`data_fsc`), **SLURP\*** (`data_slurp*`), **ITALIC** (`data_italic`), **SpeechMASSIVE** (`data_sm-de` and `data_sm-fr`).


All experiments are conducted on two NVIDIA A6000 48GB GPU. 


## 📊 Detailed Results

### 1. Comparison of unlearning methods

**Legend:**  
- **Bold** = best result  
- <u>Underlined</u> = second best  

#### A. FSC 

| **Method**        | **F1_Test** | **Acc_Test** | **F1_Forget** | **Acc_Forget** | **MIA** | **GUM** | **Speedup** | **F1_Test** | **Acc_Test** | **F1_Forget** | **Acc_Forget** | **MIA** | **GUM** | **Speedup** |
|-------------------|------------|--------------|---------------|----------------|--------|--------|------------|------------|--------------|---------------|----------------|--------|--------|------------|
|                   | **wav2vec 2.0** → ||||            |             || **HuBERT** → ||||||            |
| **Orig.**         | 0.994      | 0.994        | 1.000         | 1.000          | 0.508  | 0.000  | 1.00×      | 0.993      | 0.991        | 1.000         | 1.000          | 0.511  | 0.000  | 1.00×      |
| **Gold**          | 0.993      | 0.992        | 0.997         | 0.998          | 0.503  | 0.000  | 1.00×      | 0.991      | 0.990        | 0.996         | 0.998          | 0.507  | 0.000  | 1.00×      |
| FT                | **0.993**  | 0.993        | <u>0.999</u>  | 0.999          | **0.504** | 0.517  | 7.96×      | 0.979      | 0.979        | 0.993         | 0.992          | **0.508** | 0.514  | 7.69×      |
| NG                | 0.987      | 0.988        | 0.976         | 0.984          | <u>0.501</u> | **0.816** | **206.9×**   | **0.992**  | 0.990        | **0.996**     | 0.996          | 0.514  | 0.000  | **201.1×**   |
| NG+               | <u>0.994</u> | 0.994        | 0.994         | 0.996          | 0.493  | 0.000  | 4.03×      | 0.979      | 0.983        | 0.929         | 0.955          | 0.510  | 0.336  | 3.90×      |
| CF-k              | <u>0.994</u> | 0.994        | 1.000         | 1.000          | <u>0.501</u> | <u>0.606</u> | <u>16.97×</u> | <u>0.993</u> | 0.991        | 1.000         | 1.000          | <u>0.505</u> | **0.642** | <u>26.70×</u> |
| UNSIR             | 0.991      | 0.992        | 1.000         | 1.000          | 0.506  | 0.447  | 6.55×      | <u>0.994</u> | 0.992        | 0.998         | 0.998          | **0.508** | <u>0.484</u> | 6.38×      |
| BT                | **0.993**  | 0.994        | 1.000         | 1.000          | 0.508  | 0.000  | 4.78×      | <u>0.993</u> | 0.991        | 0.999         | 0.999          | 0.504  | 0.363  | 4.65×      |
| BT-L              | <u>0.994</u> | 0.994        | **0.996**     | 0.998          | 0.506  | 0.431  | 5.87×      | <u>0.993</u> | 0.991        | <u>0.997</u>  | 0.998          | **0.506** | 0.464  | 5.69×      |
| SCRUB             | <u>0.994</u> | 0.994        | 1.000         | 1.000          | 0.506  | 0.439  | 6.21×      | <u>0.993</u> | 0.992        | 0.998         | 0.999          | **0.508** | 0.479  | 6.22×      |


#### B. SLURP*

| **Method** | F1_Test | Acc_Test | F1_Forget | Acc_Forget | MIA   | GUM   | Speedup | F1_Test | Acc_Test | F1_Forget | Acc_Forget | MIA   | GUM   | Speedup |
|------------|---------|----------|-----------|------------|--------|--------|---------|---------|----------|-----------|------------|--------|--------|---------|
|            | **wav2vec 2.0** → ||||||| **HuBERT** → ||||||
| **Orig.**  | 0.689   | 0.815    | 1.000     | 0.999      | 0.628  | 0.000  | 1.000×  | 0.712   | 0.830    | 1.000     | 1.000      | 0.613  | 0.000  | 1.000×  |
| **Gold**   | 0.707   | 0.825    | 0.711     | 0.822      | 0.506  | 0.000  | 1.000×  | 0.704   | 0.826    | 0.715     | 0.821      | 0.492  | 0.000  | 1.000×  |
| FT         | 0.638   | 0.750    | **0.970** | 0.989      | 0.648  | 0.000  | 83.78×  | 0.734   | 0.827    | 1.000     | 1.000      | 0.611  | 0.088  | 79.00×  |
| NG         | 0.695   | 0.809    | <u>0.986</u> | 0.988  | <u>0.604</u> | **0.563** | **1748×** | 0.718   | 0.830    | 0.959     | 0.993      | 0.587  | **0.587** | **1654×** |
| NG+        | 0.701   | 0.810    | 0.995     | 0.993      | **0.603** | <u>0.446</u> | 41.63×  | 0.630   | 0.777    | **0.852** | 0.913      | **0.453** | <u>0.578</u> | 39.30×  |
| CF-k       | **0.709** | 0.818  | 1.000     | 1.000      | 0.626  | 0.089  | <u>291.9×</u> | 0.715   | 0.831    | 1.000     | 1.000      | 0.608  | 0.196  | <u>274.2×</u> |
| UNSIR      | 0.673   | 0.801    | 1.000     | 1.000      | 0.637  | 0.000  | 64.07×  | 0.722   | 0.832    | 1.000     | 1.000      | 0.613  | 0.000  | 60.44×  |
| BT         | <u>0.710</u> | 0.815 | 0.999     | 0.999      | 0.619  | 0.275  | 50.35×  | <u>0.711</u> | 0.830 | 1.000     | 1.000      | 0.613  | 0.000  | 47.42×  |
| BT-L       | 0.680   | 0.811    | 0.995     | 0.998      | 0.637  | 0.000  | 61.74×  | 0.685   | 0.792    | <u>0.907</u> | 0.954    | <u>0.558</u> | <u>0.578</u> | 58.11×  |
| SCRUB      | 0.697   | 0.824    | 0.999     | 0.999      | 0.608  | 0.429  | 64.82×  | **0.704** | 0.832    | 1.000     | 1.000      | 0.600  | 0.350  | 65.40×  |


#### C. ITALIC

| **Method** | F1_Test | Acc_Test | F1_Forget | Acc_Forget | MIA   | GUM   | Speedup | F1_Test | Acc_Test | F1_Forget | Acc_Forget | MIA   | GUM   | Speedup |
|------------|---------|----------|-----------|------------|--------|--------|---------|---------|----------|-----------|------------|--------|--------|---------|
|            | **XLS-R 128** → |||| ||| **XLS-R 53-IT** → ||||||
| **Orig.**  | 0.688   | 0.735    | 0.894     | 0.965      | 0.632  | 0.000  | 1.000×  | 0.778   | 0.837    | 1.000     | 1.000      | 0.615  | 0.000  | 1.000×  |
| **Gold**   | 0.643   | 0.709    | 0.568     | 0.631      | 0.532  | 0.000  | 1.000×  | 0.784   | 0.835    | 0.736     | 0.824      | 0.478  | 0.000  | 1.000×  |
| FT         | 0.638 | 0.686  | 0.671     | 0.739      | 0.555  | 0.590 | 30.80×  | 0.711   | 0.778    | 0.850     | 0.899      | 0.550 | 0.551 | 31.10×  |
| NG         | 0.679   | 0.728    | 0.868     | 0.933      | 0.603  | 0.646 | 613.4× | 0.590   | 0.688    | 0.621 | 0.767      | 0.525 | 0.766 | 623.0× |
| NG+        | 0.658   | 0.710    | 0.001     | 0.007      | 0.932  | 0.000  | 15.14×  | 0.743   | 0.800    | 0.936     | 0.950      | 0.582  | 0.418  | 15.37×  |
| CF-k       | 0.677   | 0.737    | 0.871     | 0.963      | 0.626  | 0.253  | 98.59× | 0.781 | 0.838    | 1.000     | 1.000      | 0.609  | 0.201  | 98.99× |
| UNSIR      | 0.636 | 0.701 | 0.830     | 0.925      | 0.621  | 0.328  | 22.01×  | 0.775 | 0.838    | 1.000     | 1.000      | 0.612  | 0.109  | 22.26×  |
| BT         | 0.683   | 0.734    | 0.639 | 0.720      | 0.481  | 0.504  | 17.90×  | 0.731   | 0.803    | 0.848 | 0.903      | 0.557  | 0.491  | 17.94×  |
| BT-L       | 0.686   | 0.734    | 0.651 | 0.749   | 0.518 | 0.558  | 22.02×  | 0.729   | 0.801    | 0.876     | 0.913      | 0.564  | 0.499  | 22.21×  |
| SCRUB      | 0.442   | 0.464    | 0.357     | 0.377      | 0.533 | 0.536  | 23.25×  | 0.770   | 0.809    | 0.990     | 0.982      | 0.610  | 0.164  | 22.66×  |



#### D. SpeechMASSIVE (DE)

| **Method** | F1_Test | Acc_Test | F1_Forget | Acc_Forget | MIA   | GUM   | Speedup | F1_Test | Acc_Test | F1_Forget | Acc_Forget | MIA   | GUM   | Speedup |
|------------|---------|----------|-----------|------------|--------|--------|---------|---------|----------|-----------|------------|--------|--------|---------|
|            | **XLS-R 128** → |||| ||| **XLS-R 53-DE** → |||||
| **Orig.**  | 0.584   | 0.681    | 0.841     | 0.938      | 0.621  | 0.000  | 1.000×  | 0.778   | 0.804    | 1.000     | 1.000      | 0.622  | 0.000  | 1.000×  |
| **Gold**   | 0.566   | 0.672    | 0.529     | 0.674      | 0.513  | 0.000  | 1.000×  | 0.745   | 0.795    | 0.706     | 0.818      | 0.493  | 0.000  | 1.000×  |
| FT         | 0.498   | 0.579    | **0.548** | 0.694      | <u>0.543</u> | <u>0.588</u> | 34.34×  | 0.661   | 0.729    | <u>0.905</u> | 0.938 | <u>0.585</u> | <u>0.464</u> | 17.79×  |
| NG         | <u>0.550</u> | 0.651 | 0.726     | 0.818      | 0.562  | **0.797** | **1078×** | 0.764   | 0.796    | 0.957     | 0.985      | 0.587  | **0.643** | **558.7×** |
| NG+        | 0.540   | 0.635    | <u>0.567</u> | 0.700   | **0.487** | 0.522  | 16.89×  | **0.759** | 0.789    | **0.878** | 0.944      | **0.568** | 0.431  | 8.770×  |
| CF-k       | 0.587   | 0.682    | 0.865     | 0.941      | 0.622  | 0.000  | <u>109.9×</u> | 0.777   | 0.803    | 1.000     | 1.000      | 0.616  | 0.208  | <u>56.93×</u> |
| UNSIR      | **0.565** | 0.649  | 0.788     | 0.924      | 0.616  | 0.197  | 27.46×  | 0.785   | 0.804    | 1.000     | 1.000      | 0.619  | 0.114  | 14.23×  |
| BT         | 0.584   | 0.682    | 0.789     | 0.912      | 0.582  | 0.489  | 20.02×  | 0.726   | 0.783    | 0.945     | 0.976      | <u>0.585</u> | 0.418  | 10.41×  |
| BT-L       | 0.584   | 0.681    | 0.786     | 0.912      | 0.576  | 0.523  | 24.87×  | <u>0.729</u> | 0.784 | 0.948     | 0.979      | 0.587  | 0.434  | 12.94×  |
| SCRUB      | 0.584   | 0.682    | 0.780     | 0.918      | 0.600  | 0.429  | 26.86×  | 0.781   | 0.800    | 1.000     | 1.000      | 0.615  | 0.211  | 13.43×  |


#### E. SpeechMASSIVE (FR)

| **Method** | F1_Test | Acc_Test | F1_Forget | Acc_Forget | MIA   | GUM   | Speedup | F1_Test | Acc_Test | F1_Forget | Acc_Forget | MIA   | GUM   | Speedup |
|------------|---------|----------|-----------|------------|--------|--------|---------|---------|----------|-----------|------------|--------|--------|---------|
|            | **XLS-R 128** → ||||||| **XLS-R 53-FR** → |||||
| **Orig.**  | 0.410   | 0.543    | 0.572     | 0.733      | 0.629  | 0.000  | 1.000×  | 0.756   | 0.815    | 1.000     | 1.000      | 0.635  | 0.000  | 1.000×  |
| **Gold**   | 0.469   | 0.618    | 0.460     | 0.580      | 0.509  | 0.000  | 1.000×  | 0.772   | 0.807    | 0.800     | 0.825      | 0.520  | 0.000  | 1.000×  |
| FT         | 0.400   | 0.527    | **0.465** | 0.589      | **0.539** | <u>0.545</u> | 18.12×  | 0.759   | 0.816    | 0.974     | 0.997      | 0.627  | 0.255  | 18.42×  |
| NG         | 0.317   | 0.438    | 0.349     | 0.491      | <u>0.564</u> | **0.749** | **597.3×** | 0.768   | 0.815    | **0.935** | 0.979      | **0.617** | **0.501** | **610.2×** |
| NG+        | 0.382   | 0.501    | 0.008     | 0.028      | 0.882  | 0.000  | 8.900×  | 0.759   | 0.807    | <u>0.943</u> | 0.982      | <u>0.620</u> | 0.317  | 9.230×  |
| CF-k       | **0.436** | 0.551  | 0.594     | 0.767      | 0.612  | 0.414  | <u>58.23×</u> | <u>0.770</u> | 0.815 | 1.000     | 1.000      | 0.624  | <u>0.338</u> | <u>58.86×</u> |
| UNSIR      | <u>0.420</u> | 0.548 | 0.591     | 0.755      | 0.620  | 0.259  | 14.67×  | 0.768   | 0.815    | 1.000     | 1.000      | 0.633  | 0.089  | 14.94×  |
| BT         | 0.411   | 0.544    | 0.583     | 0.742      | 0.597  | 0.409  | 10.60×  | **0.772** | 0.816 | 0.981     | 0.994      | 0.621  | 0.317  | 10.82×  |
| BT-L       | 0.412   | 0.543    | 0.574     | 0.739      | 0.591  | 0.447  | 13.18×  | 0.727   | 0.789    | 0.981     | 0.985      | 0.623  | 0.306  | 13.42×  |
| SCRUB      | 0.409   | 0.539    | <u>0.532</u> | 0.702   | 0.611  | 0.358  | 13.68×  | 0.769   | 0.814    | 1.000     | 1.000      | 0.633  | 0.089  | 13.94×  |

---

### 2. Best Learning Rate (LR) for Unlearning Methods

The following table shows the best LR value that maximizes the Global Unlearning Metric (GUM) proposed in the paper for each method, dataset, and model. 

| Model               | **fsc** HuBERT | **fsc** wav2vec 2.0 | **slurp** HuBERT | **slurp** wav2vec 2.0 | **italic** XLS-R 128 | **italic** XLS-R 53-IT | **sm-de** XLS-R 128- | **sm-de** XLS-R 53-DE | **sm-fr** XLS-R 128 | **sm-fr** XLS-R 53-FR |
|:-------------------|------------:|----------------:|-------------:|------------------:|-----------------:|-------------------:|-----------------:|------------------:|-----------------:|------------------:|
| `FT`           |     0.0001  |        1e-05    |      1e-05   |           0.0001  |         0.0001   |           0.0001   |         0.0001   |           0.0001  |         0.0001   |           1e-05    |
| `NG`            |     1e-06   |        5e-06    |      5e-06   |           5e-06   |         1e-06    |           5e-06    |         5e-06    |           5e-06   |         5e-06    |           5e-06    |
| `NG+`    |     1e-06   |         1e-06   |      5e-06   |           1e-06   |         1e-06    |           5e-07    |         5e-07    |           1e-06   |         1e-06    |           5e-07    |
| `CF-k`                |     5e-05   |        0.0001   |      0.0001  |           5e-05   |         0.0001   |           0.0001   |         0.0001   |           1e-05   |         0.0001   |           5e-05    |
| `UNSIR`              |     1e-05   |        0.0001   |      1e-05   |           0.0001  |         0.0001   |           1e-05    |         5e-05    |           1e-05   |         1e-05    |           1e-05    |
| `BT`       |     1e-06   |         1e-06   |      1e-06   |           5e-06   |         1e-06    |           1e-06    |         1e-06    |           1e-06   |         1e-06    |           5e-06    |
| `BT-L` |     1e-06   |         5e-07   |      1e-06   |           1e-06   |         1e-06    |           1e-06    |         1e-06    |           1e-06   |         1e-06    |           1e-06    |
| `SCRUB`              |     5e-06   |        1e-06    |      5e-06   |           5e-06   |         5e-06    |           5e-06    |         5e-06    |           5e-06   |         5e-06    |           5e-07    |

> For further details, please refer to the accompanying paper.

## 📜 License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact
For any inquiries or feedback, please contact [Alkis Koudounas](mailto:alkis.koudounas@polito.it) and [Claudio Savelli](mailto:claudio.savelli@polito.it).

## 📄 Citation
If you find this repository useful or you use GUM or CGUM, please consider citing our papers:

```bibtex
@inproceedings{koudounas25c_interspeech,
  title     = {{``Alexa, can you forget me?'' Machine Unlearning Benchmark in Spoken Language Understanding}},
  author    = {Alkis Koudounas and Claudio Savelli and Flavio Giobergia and Elena Baralis},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {1768--1772},
  doi       = {10.21437/Interspeech.2025-2607},
  issn      = {2958-1796},
}

@article{savelli2026unslu,
  title={UnSLU-BENCH+: Extended Machine Unlearning Benchmark for Spoken Language Understanding},
  author={Savelli, Claudio and Koudounas, Alkis and Giobergia, Flavio and Baralis, Elena},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  year={2026},
  publisher={IEEE}
}
```

