# Machine Unlearning on Spoken Language Understanding

This repository contains the detailed experimental results for the INTERSPEECH 2025 submission paper **"Alexa, can you forget me?” Machine Unlearning Benchmark on Spoken Language Understanding**. 

Full code, experimental setup, and results will be released upon paper acceptance.

---

## ⚙️ Experimental Setup

**Transformer models.** For English datasets (FSC and SLURP), we use [wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base) and [HuBERT](https://huggingface.co/facebook/facebook-base-ls960) models, while for ITALIC and SpeechMASSIVE we employ the multilingual [XLS-R-128](https://huggingface.co/facebook/wav2vec2-xls-r-300m) and XLS-R-53 models, with the latter ASR fine-tuned for the target language (i.e., [Italian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-italian), [German](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german),
[French](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french)).
On FSC, the models are trained for 2800 steps, with 8 batch size, 10% warmup steps, 4 gradient accumulation steps, 5e-4 initial learning rate, AdamW with weight decay, plateau scheduler, and early stopping criterion. SLURP follows the same configuration, but training lasts for 30000 steps.
On ITALIC and SpeechMASSIVE, both XLS-R-128 and specialized XLS-R-53 models are trained with the same configuration, but for 15000 total steps, 1e-4 initial learning rate, and 1 gradient accumulation step.

**Unlearning methods.** All unlearning methods were tested for a single epoch and with the hyperparameters recommended in the original papers. For example, in cf-k, the entire network was frozen except the last layer (cf-1). With SCRUB, a temperature T = 4.0 was used, and with Bad Teaching, the KL temperature is 1. In UNSIR, the learning rate of the noise was set to 0.01. AdamW was used as the optimizer for all methods to remain compliant with the training. 

All experiments are conducted on a single NVIDIA A6000 48GB GPU. 

---

## 📊 Detailed Results

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

---


> [!Note]  
> For further details, please refer to the accompanying paper.
