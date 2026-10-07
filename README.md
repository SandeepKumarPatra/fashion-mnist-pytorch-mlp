# Fashion-MNIST Image Classification with PyTorch

A PyTorch-based Deep Learning project implementing a Multi-Layer Perceptron (MLP) architecture to classify $28 \times 28$ grayscale images of clothing items into 10 categories using the Fashion-MNIST dataset.

---

## Features

- **Custom Dataset Handler:** Native PyTorch `Dataset` class (`CustomDataset`) to convert tabular Pandas DataFrames into tensors.
- **Hardware Acceleration:** Configured for both CPU and CUDA (GPU) acceleration.
- **Data Normalization:** Pixel values scaled from $[0, 255]$ to $[0, 1]$.
- **Modular Pipeline:** Evaluation loop using `torch.no_grad()` to compute accuracy metrics on test data.

---

## Model Architecture

The multi-layer neural network architecture consists of:

| Layer | Input Size | Output Size | Activation / Function |
| :--- | :--- | :--- | :--- |
| **Linear 1** | 784 features | 128 units | ReLU |
| **Linear 2** | 128 units | 64 units | ReLU |
| **Linear 3 (Output)** | 64 units | 10 classes | Raw Logits |

---

## Hyperparameters & Settings

- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Loss Function:** `CrossEntropyLoss`
- **Learning Rate:** `0.1`
- **Batch Size:** `32`
- **Epochs:** `100`

---

## Performance

- **Subset Dataset (6,000 samples):** ~85.5% Test Accuracy
- **Full Dataset (60,000 train / 10,000 test samples):** ~88.11% Test Accuracy

---

## Prerequisites & Installation

Ensure you have Python installed along with the required libraries:

```bash
pip install torch pandas numpy scikit-learn
