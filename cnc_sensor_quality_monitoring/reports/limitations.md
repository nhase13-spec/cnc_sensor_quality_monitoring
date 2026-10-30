# Known limitations and skipped work

1. **Sample size:** 18 independent experiments only.
2. **Label semantics:** Tool condition is categorical wear state, not measured tool wear depth.
3. **Visual inspection:** Missing for 4 incomplete runs; not equivalent to surface roughness or subsurface damage.
4. **Preprocessing:** Program-number invalid rule documented in README was deemed incompatible with observed data (program 1 during cuts).
5. **Deep learning:** PyTorch installation failed (network/SSL restrictions); E5 temporal comparison skipped.
6. **PyWavelets:** Not installed; wavelet energy uses a deterministic fallback implementation.
7. **Anomaly detection:** Unsupervised scores show poor recall for worn tools at experiment level.
8. **Raw data redistribution:** Verify Kaggle licence before publishing raw files in Git; use download instructions in README.
