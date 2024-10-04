A simple reproduction of score-based model NCSN (Noise Conditional Score Networks)

Original paper: https://proceedings.neurips.cc/paper_files/paper/2019/file/3001ef257407d5a371a96dcd947c7d93-Paper.pdf

Official repo: https://github.com/ermongroup/ncsn

Usage: python train.py

Implementation details: Use naive U-Net composed of convolutions and IN instead of the refineNet archetecture from original paper, which turns out that the convergence will be slower than official release

Some samples on MNIST after 50 epochs:

![imageData (2)](https://github.com/user-attachments/assets/c6230598-dc22-4f6e-9a95-4015433f0d63)

Loss trend:

![image](https://github.com/user-attachments/assets/b02dd5ae-47af-4cf8-b444-9c46bae03933)

