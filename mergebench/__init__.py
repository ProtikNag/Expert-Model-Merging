"""MergeBench-scale validation of WHC (Tier 0 divergence, Tier 1 merging).

Standalone, framework-light tooling that operates directly on HuggingFace
checkpoints (safetensors) so it scales to 2B-9B models on a single node
without depending on the MergeBench Python framework. The merged checkpoints
produced here are plain HF model directories that MergeBench's own
``scripts/evaluate.sh`` can consume unchanged.
"""
