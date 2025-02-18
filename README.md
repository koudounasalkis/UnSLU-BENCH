# Machine Unlearning on Spoken Language Understanding

This repository contains the detailed experimental results for the INTERSPEECH 2025 submission paper **"Alexa, can you forget me?” Machine Unlearning Benchmark on Spoken Language Understanding**. 

Full code, experimental setup, and results will be released upon paper acceptance.

---

## ⚙️ Experimental Setup

**2D-CNN.** The architecture includes four convolutional layers with increasing output channels: 16, 32, 64, and 128. Each layer uses a kernel size of 3, a stride of 1, and a padding of 2. After each convolution, we apply two-dimensional batch normalization, GeLU activation, and dropout. Max pooling is applied after the second and fourth convolutional layers to downsample the feature maps. A fully connected classification block follows the convolutional layers and consists of three linear layers, each followed by layer normalization, GeLU activation, and dropout.  
Mel spectrograms are computed using an FFT size of 400, a window length of 400, a hop length of 160, and 64 Mel filters. The model is trained for up to 20 epochs with early stopping after 5 epochs if no improvement is observed. We use a batch size of 256 and an initial learning rate of $1e^{-4}$. Training is optimized using AdamW with weight decay. A plateau scheduler adjusts the learning rate based on validation accuracy.  

**Transformer models.** For English datasets (FSC and SLURP), we use [wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base) and [HuBERT](https://huggingface.co/facebook/facebook-base-ls960) models, while for ITALIC and SpeechMASSIVE we employ the multilingual [XLS-R-128](https://huggingface.co/facebook/wav2vec2-xls-r-300m) and XLS-R-53 models, with the latter ASR fine-tuned for the target language (i.e., [Italian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-italian), [German](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german),
[French](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french)).   
On FSC, the models are trained for 2800 steps, with 8 batch size, 10% warmup steps, 4 gradient accumulation steps, $5e^{-4}$ initial learning rate, AdamW with weight decay, and plateau scheduler. SLURP follows the same configuration but training lasts for 30000 steps.

Training is conducted on a single NVIDIA A6000 48GB GPU.

---

## 📊 Detailed Results

---

---


---


> [!Note]  
> For further details, please refer to the accompanying paper.
