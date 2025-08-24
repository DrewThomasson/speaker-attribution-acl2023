#!/bin/bash
# Example training command for speaker attribution using ModernBERT
# 
# This script demonstrates how to train the speaker attribution model with ModernBERT
# on the PDNC dataset.

set -e  # Exit on any error

# Configuration
TRAIN_DATA="training/data/pdnc/train_fold_0.txt"
DEV_DATA="training/data/pdnc/dev_fold_0.txt"  
TEST_DATA="training/data/pdnc/test_fold_0.txt"
MODEL_NAME="answerdotai/ModernBERT-base"
SAVE_PATH="training/trained_models/modernbert_speaker_model"

echo "Training speaker attribution model with ModernBERT..."
echo "Model: $MODEL_NAME"
echo "Training data: $TRAIN_DATA"
echo "Dev data: $DEV_DATA"
echo "Test data: $TEST_DATA"
echo "Save path: $SAVE_PATH"
echo ""

# Create output directory
mkdir -p "$SAVE_PATH"

# Run training
python training/train_speaker.py \
    --trainData "$TRAIN_DATA" \
    --devData "$DEV_DATA" \
    --testData "$TEST_DATA" \
    --base_model "$MODEL_NAME" \
    --savePath "$SAVE_PATH"

echo ""
echo "Training completed! Model saved to: $SAVE_PATH"
echo ""
echo "Output files:"
echo "  - best_model.model: Trained model weights"
echo "  - val_preds.csv: Validation predictions"
echo "  - test_preds.csv: Test predictions"
echo "  - val_preds_max.csv: Validation predictions (max pooling)"
echo "  - test_preds_max.csv: Test predictions (max pooling)"

# Example of using different ModernBERT variants:
echo ""
echo "Other ModernBERT variants you can try:"
echo "  --base_model answerdotai/ModernBERT-base    # Default, 768 hidden size"
echo "  --base_model answerdotai/ModernBERT-large   # Larger model, 1024 hidden size"