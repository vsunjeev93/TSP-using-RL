# Transformer-Based TSP Solver

This is an implementation of the paper ["Attention, Learn to Solve Routing Problems!"](https://arxiv.org/abs/1803.08475) (Kool et al., 2018). The code uses PyTorch to implement a Transformer model to solve the Traveling Salesman Problem (TSP) using an actor-critic architecture.

## Files
- `transformer.py`: Main Transformer model implementation
- `attention.py`: Multi-Head Attention mechanism
- `encoder.py`: Transformer encoder block
- `decoder_MHA.py`: Decoder with Multi-Head Attention
- `critic.py`: Critic network implementation
- `train.py`: Training script
- `test.py`: Evaluation script

## Quick Start

### Training
```python
python train.py
```

### Testing
```python
python test.py
```

## Configuration

Default parameters:
- Encoder layers: 3
- Embedding dimension: 128
- Attention heads: 8
- Batch size: 512
- Learning rate: 0.0001
- Cities: 20

## Requirements
- PyTorch
- CUDA (optional)
