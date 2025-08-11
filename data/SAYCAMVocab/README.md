# SAYCAMVocab

Vocabulary mapping for the CVCL (Child Visual Concept Learning) model's tokenizer.

## File Contents

### vocab.json
- **Format**: JSON dictionary mapping tokens to indices
- **Size**: 1,418 tokens (indices 0-1417)
- **Purpose**: Used by CVCL model for text encoding

## Token Structure

### Special Tokens
- `<pad>`: 0 (padding)
- `<unk>`: 1 (unknown token)
- `<sos>`: 2 (start of sequence)
- `<eos>`: 3 (end of sequence)

### Common Tokens
The vocabulary contains everyday words a child might encounter:
- Common objects: ball, apple, car, book, etc.
- Colors: red, blue, green, yellow, etc.
- Sizes: big, small, little
- Actions: go, eat, play, walk
- People: baby, mommy, papa

## Usage in CVCL Model

### Text Encoding Process
```python
# CVCL uses this vocabulary for tokenization
tokens, token_len = model.tokenize(texts)  # Uses vocab.json
text_features = model.encode_text(tokens, token_len)
```

### Why This Vocabulary?
- **Child-centered**: Based on SAYCam dataset (infant perspective)
- **Limited vocabulary**: Reflects early language learning
- **Real-world relevance**: Words children actually hear/use

## Relationship to Project

### Model Differences
- **CVCL models**: Use this vocabulary for text encoding
- **CLIP models**: Use their own tokenizer (not this file)

### Impact on Classification
- CVCL limited to concepts within this vocabulary
- May affect zero-shot performance on novel concepts
- Important for understanding CVCL vs CLIP differences

## Key Characteristics
- **Total tokens**: 1,418
- **Language**: English
- **Source**: SAYCam child language corpus
- **Case**: Lowercase tokens

## Notes
- Essential for CVCL model operation
- Cannot be modified without retraining model
- Explains why CVCL may struggle with concepts outside child vocabulary
- Used by all CVCL-based notebooks in PatrickProject/