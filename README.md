# RingCo-DETR: Train-time Stability for Oriented DETR 

[![python](https://img.shields.io/badge/Python-3.10+-informational)]()
[![pytorch](https://img.shields.io/badge/PyTorch-2.1+-informational)]()




**English**: RingCo-DETR improves oriented DETR **at train time only** via:
- **Adaptive matching** (EMA scaling + rank-preserving quantile norm + IoU ring auxiliaries)
- **F-RIPE** spectral equal-energy ring consistency
- **OGQC** query cooperation/competition on the unit circle (self-tuned)


