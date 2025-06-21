# Aircraft Engine Remaining Useful Life (RUL) Prediction

**Bridging Academic Research with Practical Aircraft Engine Prognostics**

A deep learning framework for predicting aircraft engine RUL using attention-enhanced LSTM networks, achieving state-of-the-art performance on the NASA C-MAPSS benchmark.

---

## **The Problem**

Aircraft engine RUL prediction is critical for aviation safety and maintenance optimization. **A single engine costs $10+ million**, and unplanned failures can ground aircraft and expose operators to massive losses.

While deep learning models achieve impressive results on clean simulation data, **they fail catastrophically in real-world conditions** with irregular sampling, sensor drift, and operational heterogeneity.

---

## **Advanced RUL Prediction on NASA C-MAPSS**

Our research demonstrates **state-of-the-art performance** on the widely-used NASA C-MAPSS benchmark, with LSTM-Attention achieving the best results across all fault scenarios.

---

## **NASA C-MAPSS Benchmark Results**

**Note: The NASA C-MAPSS dataset was taken down from public access as of April 2025, making this one of the final comprehensive evaluations on the complete benchmark.**

**RMSE Performance (± standard deviation) across 40 independent training runs**

| Model | FD001 | FD002 | FD003 | FD004 | **Average** |
|-------|-------|-------|-------|-------|-------------|
| **LSTM-Attention** | **14.47 ± 0.70** | **30.52 ± 5.79** | **15.12 ± 1.00** | **27.18 ± 1.50** | **21.82** |
| CNN-LSTM | 19.33 ± 1.99 | 32.86 ± 2.75 | 18.41 ± 1.54 | 33.24 ± 1.44 | 25.96 |
| LSTM | 16.40 ± 1.23 | 35.65 ± 3.67 | 17.26 ± 1.38 | 33.32 ± 3.46 | 25.66 |
| Transformer | 24.84 ± 7.13 | 35.12 ± 4.55 | 18.69 ± 4.36 | 37.55 ± 1.11 | 29.05 |
| MLP | 30.46 ± 6.04 | 30.95 ± 0.72 | 24.99 ± 3.27 | 46.54 ± 0.93 | 33.24 |

**Key Achievements:**
- **LSTM-Attention achieves lowest RMSE across all four datasets**
- **15% improvement** over second-best model on average
- **Consistent performance** with low variance across fault scenarios
- **State-of-the-art results** on FD001 with RMSE of 14.47

---

## **LSTM with Additive Attention Architecture**

<img width="455" alt="LSTM-Attention Architecture" src="https://github.com/user-attachments/assets/bb054280-fdfa-40d1-bd5f-dfc8ebad008f" />

**Architecture Flow:**
1. **LSTM Layer**: Processes input sequence to produce hidden states H = [h₁, h₂, ..., h₁₀₀]
2. **Attention Scoring**: Each hidden state scored via trainable projection
3. **Weight Normalization**: Softmax creates attention distribution
4. **Context Aggregation**: Weighted sum produces fixed-length context vector
5. **RUL Prediction**: Dropout + dense layer outputs scalar prediction

**Mathematical Foundation:**
```
u_t = tanh(W_h * h_t + b_h)                    # Hidden state projection
α_t = exp(v^T * u_t) / Σ_j exp(v^T * u_j)      # Normalized attention weights  
c = Σ_t α_t * h_t                              # Weighted context vector
```

**Innovation**: The model learns which temporal moments are most predictive, dynamically focusing on critical failure indicators.

---

## **Key Research Findings**

**1. Massive Reality Gap**: Models suffer **catastrophic performance loss** when deployed on real data, highlighting the need for realistic benchmarks.

**2. Attention Effectiveness**: Depends on data quality - provides 15% improvement on clean data but offers no benefit with sparse, noisy industrial data.

**3. Feature Engineering Essential**: Domain-specific features (cumulative damage, temporal context) proved indispensable for convergence on real data.

**4. Complexity vs. Quality Trade-off**: Sophisticated architectures require high-quality data; simpler models often outperform complex ones in noisy conditions.

---

## **Real-World Impact**

**Aviation Applications:**
- **Predictive Maintenance**: Optimize engine replacement schedules
- **Cost Reduction**: Minimize $10M+ unplanned engine removals
- **Safety Enhancement**: Prevent in-flight failures
- **Fleet Optimization**: Improve aircraft availability

**Industrial Applications:**
- Power generation turbine monitoring
- Maritime engine prognostics  
- Manufacturing machinery RUL prediction
- Wind turbine gearbox monitoring

---

## **Setup and Usage**

### **Project Structure**
```
├── data/                    # NASA C-MAPSS and industrial datasets
├── src/                     # Core implementation
│   ├── models/             # Model architectures (LSTM-A, CNN-LSTM, etc.)
│   ├── preprocessing/      # Data pipeline and feature engineering
│   └── utils/              # Evaluation metrics and helpers
├── results/                # Model checkpoints and performance plots
└── evaluate.py            # Main training and evaluation pipeline
```

### **Quick Start**

```bash
# Clone and setup
git clone [repository-url]
cd rul-prediction
pip install tensorflow pandas numpy scikit-learn matplotlib

# Prepare data and run
# Place NASA C-MAPSS data in data/ directory
# Update data paths in evaluate.py
python evaluate.py
```

### **Key Features**
- **Robust Evaluation**: 10-fold cross-validation with proper engine-level grouping
- **Reproducible Results**: Fixed random seeds and consistent preprocessing
- **Production Ready**: GPU-accelerated training with early stopping
- **Real-World Validation**: Industrial dataset evaluation

---

## **Research Contributions**

1. **State-of-the-Art C-MAPSS Performance**: Achieved best-in-class results across all NASA C-MAPSS datasets
2. **Novel Attention Architecture**: Demonstrated superior performance of attention mechanisms for RUL prediction
3. **Comprehensive Model Comparison**: Systematic evaluation of five different architectures under identical conditions
4. **Feature Engineering Insights**: Identified critical domain-specific features for optimal performance

---

## **Citation**

```bibtex
@article{barnett2025realistic,
  title={Towards Realistic Benchmarking for Aircraft Engine Remaining Useful Life Prediction},
  author={Barnett, Brendan and Nadella, Suhaas and Nguyen, Truc Ho and Nagabandi, Ankith and Goel, Jai and Lee, Jeremiah and Seto, Danbing and Ji, Shihao},
  journal={arXiv preprint},
  year={2025},
  month={May}
}
```

---

**Built for advancing predictive maintenance from laboratory to flight line**
