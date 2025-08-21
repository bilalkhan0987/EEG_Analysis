# Decoding Executive Functions using CNN+Transformer using EEG signals

## Project Description

This repository provides a PyTorch implementation of a hybrid **CNN+Transformer model** for decoding
executive functions (Updating, Shifting, Inhibition) from EEG signals.  
The model combines convolutional layers to capture spatial features with transformer encoders
to model temporal dependencies in EEG time series.

The framework is modular and can be extended to:
- Other EEG decoding tasks
- Cognitive state classification
- General time-series classification problems

It includes:
- Custom multi-head self-attention
- Transformer encoder blocks
- Training scripts with checkpointing and evaluation
