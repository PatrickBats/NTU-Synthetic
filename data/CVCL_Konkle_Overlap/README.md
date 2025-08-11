# CVCL_Konkle_Overlap

Analysis of vocabulary overlap between CVCL model and Konkle dataset object classes.

## File Contents

### CVCLKonkMatches 
- **Purpose**: Maps KonkLab object classes to CVCL vocabulary availability
- **Key insight**: Identifies which Konkle objects have corresponding words in CVCL's limited vocabulary

## Why This Matters

### The Challenge
- **CVCL vocabulary**: Limited to ~1,400 child-relevant words
- **KonkLab dataset**: Contains 200+ object classes
- **Issue**: Not all KonkLab objects may be in CVCL vocabulary

### Impact on Experiments
This overlap analysis helps explain:
- Why CVCL might perform poorly on certain object classes
- Which classification tests are fair comparisons with CLIP
- Where CVCL's child-based training shows limitations

## Usage in Project

### For Researchers
- Check this file before designing new experiments
- Understand CVCL performance limitations
- Select appropriate test objects for fair model comparison

### Example Insights
Objects likely in CVCL vocabulary:
- apple, ball, book (common child words)

Objects possibly missing:
- babushkadolls, beanbagchair (less common in child speech)

## Relationship to Classification Tests

### Fair Comparisons
- Tests using objects in both vocabularies are most fair
- Missing vocabulary explains some CVCL failures
- Important context for interpreting results

### Zero-shot Limitations
- CVCL cannot classify objects not in its vocabulary
- CLIP has broader vocabulary from internet training
- This file documents these systematic differences

## Notes
- Critical for understanding model comparison results
- Explains performance gaps between CVCL and CLIP
- Should be consulted when interpreting classification accuracies
- Highlights the importance of training data in model capabilities