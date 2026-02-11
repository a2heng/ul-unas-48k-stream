# UL-UNAS (48kHz Fork)

This repository is forked from [Xiaobin-Rong/ul-unas](https://github.com/Xiaobin-Rong/ul-unas), the official implementation of the *IEEE TASLP* paper:

> [UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search](https://arxiv.org/abs/2503.00340)  
> by Xiaobin Rong, Dahan Wang, Yuxiang Hu, Changbao Zhu, Kai Chen, and Jing Lu

[![arxiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.00340)
[![demo](https://img.shields.io/badge/GitHub-Demo-orange.svg)](https://xiaobin-rong.github.io/ul-unas_demo/)

## Adaptations in This Fork

This fork adapts the original 16kHz UL-UNAS for 48kHz high-quality audio processing with some minor modifications.

### 1. 48kHz Audio Support
Adapted the model to support **48 kHz sampling rate**:
- Adjusted STFT parameters: FFT size 960, hop length 480
- Modified ERB filter bank configuration for 48kHz (80 low subbands + 41 high subbands)

### 2. DNS5 Dataset
Retrained the model on the **DNS5 (Deep Noise Suppression Challenge 5)** dataset for broader noise scenarios.


## Minor Improvements

### 3. Simplified GRNN Architecture
Made a small modification to the **GRNN (Grouped RNN)** module:
- Replaced the original dual-GRU design with a single standard GRU
- This slightly reduces computational overhead and CPU usage

### 4. Streaming Inference Support
Added basic streaming inference capability with state management:
- **StreamingConvBuffer**: Handles convolution buffers for causal padding
- **StreamingGRUState**: Manages hidden states frame by frame
- **ONNX Export**: Supports exporting to ONNX format with explicit state I/O for edge deployment

## Notes on Model Architecture

The model architecture parameters (channel sizes, kernel sizes, block types, etc.) are kept consistent with the original paper. However, please note that:

- The original paper performed Neural Architecture Search (NAS) specifically for **16 kHz** audio
- This fork uses the same architecture for **48 kHz** without re-running the search
- The 48 kHz configuration may benefit from further architecture optimization, as the optimal parameters could differ from the 16 kHz version

## Citation

If you use this work, please cite the original paper:

```bibtex
@misc{rong2025ulunas,
      title={UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search}, 
      author={Xiaobin Rong and Dahan Wang and Yuxiang Hu and Changbao Zhu and Kai Chen and Jing Lu},
      year={2025},
      eprint={2503.00340},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2503.00340}, 
}
```

## Contact

Fork Maintainer: [a2heng](https://github.com/a2heng)

Original Author: Xiaobin Rong [xiaobin.rong@smail.nju.edu.cn](mailto:xiaobin.rong@smail.nju.edu.cn)
