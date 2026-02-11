# UL-UNAS (48kHz Fork)

This repository is forked from [Xiaobin-Rong/ul-unas](https://github.com/Xiaobin-Rong/ul-unas), the official implementation of the *IEEE TASLP* paper:

> [UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search](https://arxiv.org/abs/2503.00340)  
> by Xiaobin Rong, Dahan Wang, Yuxiang Hu, Changbao Zhu, Kai Chen, and Jing Lu

[![arxiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.00340)
[![demo](https://img.shields.io/badge/GitHub-Demo-orange.svg)](https://xiaobin-rong.github.io/ul-unas_demo/)

## Modifications in This Fork

This fork extends the original 16kHz UL-UNAS with the following improvements:

### 1. High-Quality Audio Support (48kHz)
- Extended the model to support **48 kHz sampling rate** for high-fidelity speech enhancement
- Adjusted STFT parameters: FFT size 960, hop length 480
- Modified ERB filter bank configuration for 48kHz (80 low subbands + 41 high subbands)

### 2. DNS5 Dataset Training
- Trained on the **DNS5 (Deep Noise Suppression Challenge 5)** dataset
- Improved noise robustness across diverse real-world scenarios

### 3. Enhanced Loss Function
The training loss has been enhanced with two additional components:
- **Energy Ceiling Loss**: Suppresses artifacts and over-amplification for quiet sounds
- **Loudness Loss**: Improves perceptual quality by matching the loudness characteristics of clean speech

Combined with the original loss terms (RI loss, magnitude loss, short-time loss, Si-SNR), the model achieves better subjective and objective quality.

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
