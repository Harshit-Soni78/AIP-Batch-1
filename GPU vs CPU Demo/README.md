# Hands-On Session: GPU vs CPU in Machine Learning

## Introduction

This hands-on session demonstrates the advantages of using **GPU acceleration** in machine learning compared to **CPU-based training**. We will train an **AlexNet model** on the **Fashion MNIST dataset** and measure the time differences in training and inference between CPU and GPU using **Google Colab**.

## Learning Objectives

By the end of this session, participants will:

- Understand the benefits of GPU acceleration in deep learning.
- Learn how to enable and use a GPU in Google Colab.
- Train a deep learning model (AlexNet) on the Fashion MNIST dataset.
- Compare training and inference times between CPU and GPU.
- Visualize performance differences using graphs.

---

## Session Outline

### 1. **Introduction to GPU Acceleration**

- Why GPUs are preferred over CPUs in deep learning.
- Parallel processing and its role in model training.
- Introduction to CUDA and TensorFlow/PyTorch GPU optimization.

### 2. **Setting Up Google Colab**

- Open **Google Colab** and go to `Runtime` → `Change runtime type` → Select `GPU`.
- Install required libraries:

  ```bash
  !pip install torch torchvision tensorflow
  ```

### 3. **Loading the Fashion MNIST Dataset**

- Load the dataset using PyTorch:

  ```python
  import torchvision.transforms as transforms
  from torchvision.datasets import FashionMNIST
  from torch.utils.data import DataLoader

  transform = transforms.Compose([transforms.ToTensor()])
  train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
  test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

  train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
  ```

### 4. **Building an AlexNet Model**

- Define a **simplified AlexNet** model for Fashion MNIST:

  ```python
  import torch.nn as nn
  import torch.optim as optim
  import torch

  class AlexNet(nn.Module):
      def __init__(self):
          super(AlexNet, self).__init__()
          self.conv_layers = nn.Sequential(
              nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2)
          )
          self.fc_layers = nn.Sequential(
              nn.Linear(128 * 7 * 7, 512),
              nn.ReLU(),
              nn.Linear(512, 10)
          )

      def forward(self, x):
          x = self.conv_layers(x)
          x = x.view(x.size(0), -1)
          x = self.fc_layers(x)
          return x

  model = AlexNet()
  ```

### 5. **Training on CPU vs GPU**

#### **Training on CPU:**

```python
import time

device_cpu = torch.device("cpu")
model.to(device_cpu)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
for epoch in range(2):
    for images, labels in train_loader:
        images, labels = images.to(device_cpu), labels.to(device_cpu)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

end_time = time.time()
print("Training time on CPU:", end_time - start_time, "seconds")
```

#### **Training on GPU:**

```python
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device_gpu)

start_time = time.time()
for epoch in range(2):
    for images, labels in train_loader:
        images, labels = images.to(device_gpu), labels.to(device_gpu)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

end_time = time.time()
print("Training time on GPU:", end_time - start_time, "seconds")
```

### 6. **Comparing Inference Time**

- Measure inference time on CPU and GPU:

  ```python
  model.eval()

  # CPU inference
  model.to(device_cpu)
  cpu_start = time.time()
  with torch.no_grad():
      for images, labels in test_loader:
          images = images.to(device_cpu)
          outputs = model(images)
  cpu_end = time.time()
  print("Inference time on CPU:", cpu_end - cpu_start, "seconds")

  # GPU inference
  model.to(device_gpu)
  gpu_start = time.time()
  with torch.no_grad():
      for images, labels in test_loader:
          images = images.to(device_gpu)
          outputs = model(images)
  gpu_end = time.time()
  print("Inference time on GPU:", gpu_end - gpu_start, "seconds")
  ```

### 7. **Visualizing Performance Gains**

- Compare CPU vs GPU inference time using a bar chart:

  ```python
  import matplotlib.pyplot as plt

  times = [cpu_end - cpu_start, gpu_end - gpu_start]
  labels = ['CPU', 'GPU']

  plt.bar(labels, times, color=['red', 'blue'])
  plt.xlabel('Device')
  plt.ylabel('Time (seconds)')
  plt.title('Inference Time Comparison')
  plt.show()
  ```

### 8. **Key Takeaways**

- **GPU training is significantly faster** than CPU due to parallel processing.
- **Inference on GPU** is also much faster, making it ideal for real-time applications.
- Google Colab provides **free access to GPUs**, making it an excellent platform for experimentation.
- **Optimizing deep learning models** for GPU usage can lead to massive efficiency gains in real-world applications.

---

## **Conclusion**

This hands-on session provided a practical demonstration of how GPUs accelerate deep learning training and inference. By comparing CPU vs GPU performance, participants now understand why GPUs are the preferred choice for large-scale machine learning tasks.

### **Future Enhancements**

- Experiment with different batch sizes to observe GPU utilization changes.
- Train on larger datasets like CIFAR-10 or ImageNet.
- Optimize training by using techniques like mixed-precision training with **AMP (Automatic Mixed Precision)**.

---

## **Acknowledgments**

This session was conducted using **Google Colab** and **PyTorch** for deep learning model implementation. Special thanks to all participants for their active engagement!

🚀 Happy Coding!

## Colab Notebook

Colab Notebook: [Notebook Link](https://colab.research.google.com/drive/1sCPjRxHXVM3G61QyKONh29asZHFwjrSP?usp=sharing)

## Author

**Harshit Soni**  
GitHub: [Harshit-Soni78](https://github.com/Harshit-Soni78)

---
Made with ❤️ by Harshit Soni
