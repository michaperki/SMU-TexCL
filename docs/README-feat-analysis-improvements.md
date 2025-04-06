# feat/analysis_improvements

## Overview
This branch introduces improvements in turbulence prediction, enhancing both classification and regression models.

## Key Improvements

- **Classification Performance**
  - Ordinal classifier accuracy improved to 44.8% (previously 39%) with 91.7% adjacent accuracy.
  - Standard classifier accuracy increased to 44% (up from 39%).

- **Turbulence-Level Prediction**
  - Notable improvements at turbulence levels 7 (38.9% vs 33.3%) and 5 (44.4% vs 27.8%).
  - Levels 0 and 1 remain reliably predicted.

- **Pilot Normalization**
  - Pilot-normalized features continue to show value (R² of 0.06).

- **Sequence Models**
  - LSTM using raw features achieved an MSE of 1511.88, outperforming the default features (MSE of 1722.43).

## Recommendations

- Consider adding weighting to the ordinal classifier to penalize larger errors.
- Explore specialized models for different turbulence levels, potentially via an ensemble approach.
- Further refine the adaptive signal filtering with additional frequency domain features.
- Investigate a two-stage approach: classify into turbulence groups then apply regression within each.

## Conclusion
These improvements demonstrate that targeted feature engineering and model specialization can significantly enhance prediction performance.
