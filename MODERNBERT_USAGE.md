# ModernBERT Integration

This repository has been updated to use ModernBERT instead of BERT for improved performance and efficiency in speaker attribution tasks.

## Key Changes

### Model Architecture
- **Previous**: Used BERT (Bidirectional Encoder Representations from Transformers)
- **Current**: Uses ModernBERT, a more efficient and faster alternative
- **Default Model**: `answerdotai/ModernBERT-base` (768 hidden dimensions)
- **Large Model**: `answerdotai/ModernBERT-large` (1024 hidden dimensions)

### Updated Components
- `ModernBERTSpeakerID` class for speaker identification
- All training data generation scripts
- Entity tagger and coreference resolution modules
- Backward compatibility maintained with `BERTSpeakerID` alias

## Training with ModernBERT

### Basic Usage
```bash
python training/train_speaker.py \
    --trainData training/data/pdnc/train_fold_0.txt \
    --devData training/data/pdnc/dev_fold_0.txt \
    --testData training/data/pdnc/test_fold_0.txt \
    --base_model answerdotai/ModernBERT-base \
    --savePath training/trained_models/my_model
```

### Using the Example Script
```bash
# Make the script executable (if not already)
chmod +x examples/train_with_modernbert.sh

# Run the training
./examples/train_with_modernbert.sh
```

### Model Variants
- `answerdotai/ModernBERT-base`: Standard model (768 dimensions)
- `answerdotai/ModernBERT-large`: Larger model (1024 dimensions) for better performance

### Output Files
After training, the following files will be saved in your specified `--savePath`:
- `best_model.model`: Trained model weights (best performing on dev set)
- `val_preds.csv`: Development set predictions 
- `test_preds.csv`: Test set predictions
- `val_preds_max.csv`: Development set predictions (using max pooling)
- `test_preds_max.csv`: Test set predictions (using max pooling)

## Requirements

Ensure you have the required dependencies installed:
```bash
pip install torch transformers datasets
```

## Benefits of ModernBERT

1. **Faster Training**: More efficient architecture reduces training time
2. **Better Performance**: Improved accuracy on downstream tasks
3. **Memory Efficient**: Lower memory requirements compared to traditional BERT
4. **Drop-in Replacement**: Minimal code changes required from BERT

## Backward Compatibility

The old `BERTSpeakerID` class name is still supported as an alias to `ModernBERTSpeakerID`, ensuring existing code continues to work without modification.