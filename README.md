# LSTM Glucose Prediction Starter Template

A PyTorch implementation for continuous glucose monitoring (CGM) prediction using LSTM networks, designed for diabetes management applications.

## What This Starter Includes

* **Data handling:** Loads a CSV, enforces 5-min cadence, fills small gaps, and builds sliding windows.
* **Feature engineering:** Simple IOB/COB proxies, time-of-day and day-of-week encodings, macro fractions, CGM lags.
* **Model:** Stacked LSTM → MLP head for either single-step (e.g., +30 min) or **multi-step** (full 30–60 min trajectory).
* **Training loop:** Adam, gradient clipping, time-based train/val/test split, best-model selection.
* **Metrics:** RMSE, MAE, and a basic **Clarke Error Grid** summary (A/B/C/D/E zone fractions).
* **Artifacts:** Saves `lstm_glucose.pt`, `predictions.npz`, and `feature_cols.txt`.

## CSV Format (5-min rows)

```csv
timestamp, cgm_mgdl, bolus_units, basal_u_per_hr, carbs_g, protein_g, fat_g
```

Optional extra columns (if you have them): `activity_min`, `heart_rate`, illness flags, etc.

## How to Run It

1. **Install dependencies:**
```bash
pip install torch pandas numpy scikit-learn
```

2. **Single-step prediction** (e.g., **+30 min** horizon = 6×5-min):
```bash
python lstm_glucose_prediction.py --data your_data.csv --horizon 6
```

3. **Multi-step trajectory** (e.g., **+60 min** = 12 steps):
```bash
python lstm_glucose_prediction.py --data your_data.csv --horizon 12 --multi_step
```

4. **Tune important knobs:**
   * `--lookback 36` → 3-hour history window (try 24–72)
   * `--horizon 6–12` → 30–60 min ahead
   * `--layers 1–3`, `--hidden 64–256`, `--dropout 0.1–0.3`
   * `--epochs 30–100`, with early stopping (kept simple here)

## Good Practices (Short 'Recipe')

* **Split by time**, not randomly (script already does ~70/15/15).
* **Normalize features** using train-only stats (already in script).
* **Personalize per user** (train one model per person) or add a user-ID embedding if pooling multiple users.
* **Calibrate targets:** If CGM has bias vs. fingerstick, consider periodic calibration or target smoothing.
* **Choose horizon wisely:** +30–45 min is a sweet spot for CGM control; longer horizons are noisier.
* **Handle missing data:** The script interpolates short gaps; for longer gaps, drop those segments.

## Extending the Starter (When You're Ready)

* Add **attention** or a **Temporal Convolutional Network** front-end.
* Learn better **IOB/COB** with absorption kernels (e.g., piecewise linear or gamma).
* Predict **uncertainty** (use MC-dropout or a probabilistic head predicting mean+variance).
* Add **exogenous inputs:** activity/HR/sleep, stress, menstrual phase, illness.
* Evaluate with **Parkes/Consensus error grids** and **time in range** impacts via simulated alarms.

## Key Features Explained

### IOB (Insulin on Board)
The script uses a simplified exponential decay model with a 5-hour time constant (τ=300 min) to estimate active insulin in the body. This combines both basal and bolus insulin contributions.

### COB (Carbs on Board)
Similar exponential decay for carbohydrate absorption, with a faster 2-hour time constant (τ=120 min), representing typical meal absorption dynamics.

### Time Encodings
Cyclical sine/cosine encodings capture daily and weekly patterns in glucose dynamics, important for circadian rhythm effects.

### Macronutrient Ratios
Protein and fat affect glucose absorption rates differently than carbs. The script calculates fraction of each macronutrient per meal to capture these effects.

### Lag Features
Previous CGM values at 5, 10, 15, 30, and 60 minutes help the model learn glucose trends and rates of change.

## Output Files

After training, the script saves:
- `lstm_glucose.pt` - Trained model weights
- `predictions.npz` - Test set predictions for visualization
- `feature_cols.txt` - List of features used (for inference)

## Performance Expectations

With default settings on typical CGM data:
- **RMSE:** 15-25 mg/dL for 30-min horizon
- **Clarke A-zone:** 75-85% of predictions
- **MAE:** 10-20 mg/dL

Performance varies significantly based on:
- Individual glucose variability
- Meal/insulin logging accuracy
- Data completeness
- Horizon length

## Limitations & Disclaimers

⚠️ **This is a research/educational tool, not medical software.**
- Not validated for clinical use
- Predictions should not replace CGM alarms or medical advice
- Always consult healthcare providers for diabetes management
- Model performance degrades rapidly beyond 45-60 min horizons
- Requires accurate meal and insulin logging for best results

## Citation

If you use this starter template in research, please cite:
```
@software{lstm_glucose_prediction,
  title = {LSTM Glucose Prediction Starter Template},
  year = {2024},
  url = {https://github.com/yourusername/glucose-lstm-predictor}
}
```

## License

Proprietary License - See LICENSE file for details

## Contributing

Contributions welcome! Areas of interest:
- Additional feature engineering
- Alternative model architectures
- Uncertainty quantification methods
- Multi-user personalization approaches
- Integration with CGM APIs

Please open an issue first to discuss major changes.
