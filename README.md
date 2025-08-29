# Zhang’s Six-Frame in CUDA (TARA CUDA)
**Proponents**:
- Arceta, Althea Zyrie
- Mendoza, Antonio Gabriel
- Tan, Jose Tristan

**Adviser**: Uy, Roger Luis

---

This repository contains the source code, datasets, and documentation for the thesis project **TARA CUDA: An SIMT Implementation of Zhang’s Six-Frame Algorithm using CUDA**. The project demonstrates how SIMT-based parallelization using CUDA can significantly reduce the runtime of sequence alignment compared to sequential and SIMD (AVX2) implementations.  

Benchmark results show that the CUDA implementation achieved an average speedup of **3.51×** (maximum **6.03×**) on the fruit fly dataset, and an average of **3.44×** (maximum **6.44×**) on the mouse-ear cress dataset, relative to the sequential version. When compared against the SIMD implementation, the CUDA approach achieved an average speedup of **3.14×**, confirming CUDA’s viability as an alternative paradigm for accelerating sequence alignment.  

Additionally, a [YouTube playlist](https://www.youtube.com/playlist?list=PLd3yBvnKNYJz0m9w5AXMP8xE9ouf9FWKY) has been prepared to further explain the project.

---

## Features
- Implements Zhang’s Six-Frame Alignment Algorithm in both **Sequential (CPU)** and **Parallel (CUDA SIMT)** versions.  
- Supports both **3-frame** and **6-frame** alignments.  
- Uses the **BLOSUM62 scoring matrix** with configurable gap and shift penalties.  
- Outputs the best matching proteins with their scores, indices, and execution time.  

---

## Requirements
- **CUDA-capable NVIDIA GPU** (tested on Compute Capability 7.5, 8.6, and 8.9).  
- **Visual Studio 2022 (Community Edition)** with C++ Desktop Development.  
- **CUDA Toolkit** (corresponding to your GPU and OS version).  

---

## Installation

1. **Install Visual Studio 2022**  
   - Download from: [Visual Studio Community](https://visualstudio.microsoft.com/free-developer-offers/)  
   - During installation, select **Desktop Development with C++**.

2. **Install CUDA Toolkit**  
   - Download from: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)  
   - Run the installer and follow prompts.

3. **Clone this repository**  
   ```bash
   git clone https://github.com/Shiray427/Zhangs-Six-Frame-in-CUDA.git
   cd Zhangs-Six-Frame-in-CUDA
   ```

4. **Open the project in Visual Studio**  
   - Launch `Zhang's Six Frame in CUDA.sln`.

---

## Running the Program

1. Press **Run** in Visual Studio 2022.  
2. Input required files when prompted:  
   - Protein FASTA file (first).  
   - DNA FASTA file (second).  
3. Choose options:  
   - Implementation: `0` = Sequential, `1` = CUDA.  
   - Frame count: `3` = Forward only, `6` = Forward + Reverse complement.  
   - Output: `0` = Top 1 score, `1` = Top 5 scores.  
