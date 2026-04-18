# Baseline CNN vs MobileNetV2 — Comparison Report

*Generated automatically by `scripts/compare_models.py`*

**Machine:** Intel64 Family 6 Model 141 Stepping 1, GenuineIntel · 6 logical cores · torch 2.11.0+cu128

## Summary

| Metric | baseline | mobilenet_v2 |
|--------|--------|--------|
| Total parameters | 398,662 | 2,272,550 |
| Test accuracy | 0.9970 (99.70%) | 0.9968 (99.68%) |
| Macro F1 | 0.9970 | 0.9968 |
| CPU latency — median (ms) | 40.72 | 19.05 |
| CPU latency — p95 (ms) | 44.26 | 21.53 |
| CPU latency — mean (ms) | 40.71 | 19.23 |
| Best epoch (baseline) | 24 (stage 1) | — |
| Best val acc (baseline) | 0.9972 | — |
| Best epoch (mobilenet_v2) | — | 28 (stage 2) |
| Best val acc (mobilenet_v2) | — | 0.9970 |

## Per-class F1 (sorted by baseline F1 ascending)

Δ = MobileNetV2 F1 − Baseline F1

| Class | F1 (baseline) | F1 (mobilenet_v2) | Δ |
|--------|--------|--------|--------|
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.9698 | 0.9804 | +0.0105 |
| Corn_(maize)___Northern_Leaf_Blight | 0.9775 | 0.9833 | +0.0058 |
| Tomato___Target_Spot | 0.9891 | 0.9823 | -0.0068 |
| Tomato___Late_blight | 0.9913 | 0.9871 | -0.0043 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.9931 | 0.9842 | -0.0089 |
| Potato___Late_blight | 0.9938 | 1.0000 | +0.0062 |
| Potato___healthy | 0.9956 | 0.9978 | +0.0022 |
| Tomato___Early_blight | 0.9959 | 0.9854 | -0.0105 |
| Pepper,_bell___healthy | 0.9960 | 0.9980 | +0.0020 |
| Tomato___Bacterial_spot | 0.9977 | 0.9976 | -0.0000 |
| Peach___healthy | 0.9977 | 0.9977 | +0.0000 |
| Raspberry___healthy | 0.9978 | 1.0000 | +0.0022 |
| Tomato___Tomato_mosaic_virus | 0.9978 | 1.0000 | +0.0022 |
| Blueberry___healthy | 0.9978 | 1.0000 | +0.0022 |
| Cherry_(including_sour)___healthy | 0.9978 | 1.0000 | +0.0022 |
| Grape___Esca_(Black_Measles) | 0.9979 | 1.0000 | +0.0021 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.9980 | 0.9980 | +0.0000 |
| Apple___Apple_scab | 1.0000 | 0.9980 | -0.0020 |
| Apple___Black_rot | 1.0000 | 1.0000 | +0.0000 |
| Apple___Cedar_apple_rust | 1.0000 | 1.0000 | +0.0000 |
| Apple___healthy | 1.0000 | 1.0000 | +0.0000 |
| Cherry_(including_sour)___Powdery_mildew | 1.0000 | 0.9976 | -0.0024 |
| Corn_(maize)___Common_rust_ | 1.0000 | 1.0000 | +0.0000 |
| Corn_(maize)___healthy | 1.0000 | 0.9979 | -0.0021 |
| Grape___Black_rot | 1.0000 | 1.0000 | +0.0000 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1.0000 | 1.0000 | +0.0000 |
| Grape___healthy | 1.0000 | 1.0000 | +0.0000 |
| Orange___Haunglongbing_(Citrus_greening) | 1.0000 | 1.0000 | +0.0000 |
| Peach___Bacterial_spot | 1.0000 | 1.0000 | +0.0000 |
| Pepper,_bell___Bacterial_spot | 1.0000 | 1.0000 | +0.0000 |
| Potato___Early_blight | 1.0000 | 1.0000 | +0.0000 |
| Soybean___healthy | 1.0000 | 0.9980 | -0.0020 |
| Squash___Powdery_mildew | 1.0000 | 1.0000 | +0.0000 |
| Strawberry___Leaf_scorch | 1.0000 | 1.0000 | +0.0000 |
| Strawberry___healthy | 1.0000 | 1.0000 | +0.0000 |
| Tomato___Leaf_Mold | 1.0000 | 0.9979 | -0.0021 |
| Tomato___Septoria_leaf_spot | 1.0000 | 0.9954 | -0.0046 |
| Tomato___healthy | 1.0000 | 1.0000 | +0.0000 |

## Top 10 Confused Class Pairs

### baseline

| Rank | Count | True class | Predicted as |
|------|-------|-----------|-------------|
| 1 | 11 | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Northern_Leaf_Blight |
| 2 | 2 | Tomato___Late_blight | Potato___Late_blight |
| 3 | 1 | Raspberry___healthy | Potato___healthy |
| 4 | 1 | Potato___Late_blight | Tomato___Late_blight |
| 5 | 1 | Tomato___Late_blight | Tomato___Early_blight |
| 6 | 1 | Tomato___Target_Spot | Tomato___Spider_mites Two-spotted_spider_mite |
| 7 | 1 | Potato___healthy | Blueberry___healthy |
| 8 | 1 | Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| 9 | 1 | Tomato___Spider_mites Two-spotted_spider_mite | Tomato___Target_Spot |
| 10 | 1 | Tomato___Target_Spot | Tomato___Early_blight |

### mobilenet_v2

| Rank | Count | True class | Predicted as |
|------|-------|-----------|-------------|
| 1 | 6 | Tomato___Target_Spot | Tomato___Spider_mites Two-spotted_spider_mite |
| 2 | 5 | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Northern_Leaf_Blight |
| 3 | 3 | Tomato___Early_blight | Tomato___Late_blight |
| 4 | 3 | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot |
| 5 | 2 | Tomato___Late_blight | Tomato___Early_blight |
| 6 | 1 | Tomato___Leaf_Mold | Tomato___Early_blight |
| 7 | 1 | Potato___healthy | Soybean___healthy |
| 8 | 1 | Apple___Apple_scab | Cherry_(including_sour)___Powdery_mildew |
| 9 | 1 | Tomato___Early_blight | Tomato___Spider_mites Two-spotted_spider_mite |
| 10 | 1 | Tomato___Bacterial_spot | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
