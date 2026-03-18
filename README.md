Gobbling-Data 

> A comprehensive corporate analytics platform combining data visualization, anomaly detection, predictive analytics, and portfolio optimization for financial intelligence and risk assessment. 
🎯 Overview 

Gobbling-Data is a full-stack data analytics platform designed for financial analysis, corporate intelligence, and portfolio management. The platform provides advanced tools for filtering company data, detecting anomalies, generating predictions, analyzing causal impacts, and optimizing investment portfolios.

🏗️ Architecture

Technology Stack

- **Frontend**: React with Streamlit for rapid dashboard development
  - Visualization: Plotly, Recharts, Matplotlib, Seaborn
  - Data Tables: Ag-Grid/Perspective.js for pivot table functionality
  - Async request handling
  - Native integration with ML/AI models
- **Data Layer**: Excel/CSV/Parquet files with optional SQLite/PostgreSQL support
- **Machine Learning**: scikit-learn, XGBoost, Isolation Forest
- **Causal Inference**: DoWhy
- **Network Analysis**: NetworkX, Graphviz
- **NLP**: HuggingFace Transformers, Sentence-Transformers

✨ Key Features

1. 🔍 Filtering & Search (Kevin)
- Filter companies by specific categories with flexible conditions
- Search for comparable companies based on similarity metrics
- Find subsidiaries by parent company and vice versa
- Support for industry, revenue, market cap, and custom filters

2. 📊 Tableau-esque Visualization (Sophie)
- Interactive pivot table construction
- Dynamic chart generation (line, bar, scatter, etc.)
- Save and retrieve custom visualizations
- Multi-dimensional data exploration

3. 🚨 Anomaly Detection (Joseph)
- Ensemble-based anomaly detection using Isolation Forest
- Detect unusual patterns in revenue, profit margins, employee counts
- Network-based contagion risk analysis
- Corporate hierarchy risk mapping
- **Forensic Score System (SFS)** for entity-level risk assessment

4. 📈 Predictive Analytics (Danny)
- Revenue forecasting using ensemble models
- Multi-horizon predictions with confidence intervals
- External factor integration
- Model performance tracking

5. 🔬 Causal Impact Analysis (Joel)
- Analyze the impact of interventions (mergers, acquisitions, policy changes)
- Pre/post-period comparison with control groups
- Counterfactual scenario modeling

6. 💼 Portfolio Optimization (Joel)
- **Gower Distance-Based Diversification**: Propose investments that maximize portfolio diversity
- **LLM-Powered Risk Analysis**: AI-generated insights on portfolio risks
- **Multi-metric Diversification Analysis**: Sector coverage, asset intensity, and correlation metrics
- Portfolio management (save, update, delete portfolios)

📁 Project Structure

```
Gobbling-Data/
├── backend/                   
│   ├── main.py                 # Main application entry point
│   ├── Filter.py               # Data filtering and normalization utilities
│   ├── company_relations.py    # Corporate hierarchy analysis
│   ├── data/                   # Data storage
│   ├── services/               # Backend services
│   │   └── sic_embedding.py    # Industry classification embeddings
│   ├── api/                    # API route definitions
│   │   └── routes/
│   │       ├── causal_analytics.py
│   │       └── portfolio_diversification.py
│   ├── models/                 # Data models and schemas
│   │   └── portfolio_diversification.py
│   └── requirements.txt        # Python dependencies
│
├── frontend/                   # React/Streamlit frontend
│   ├── frontend.py             # Main dashboard application
│   └── artifacts/              # Model artifacts
│       └── revenue_model.pkl
│
├── anomaly/                    # Anomaly detection module
│   ├── dashboard.py            # Forensic analysis dashboard
│   ├── forensic_pipeline.py    # Data preprocessing and metric generation
│   ├── detective_models.py     # Isolation Forest implementation
│   ├── nexus_network.py        # Corporate network graph construction
│   ├── quant_engine.py         # PCA-based risk optimization
│   ├── evidence_visualizer.py  # Network visualization tools
│   ├── sensitivity.py          # Parameter sensitivity testing
│   ├── decay_test.py           # Risk propagation modeling
│   └── param_test.py           # Hyperparameter optimization
│
├── submissions/                # Project deliverables
├── API_CONTRACTS.md            # Complete API documentation
└── README.md                   # This file
``` 

🚀 Getting Started

Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- pip or conda for package management

Installation

1. **Install backend dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Install anomaly detection dependencies**
   ```bash
   cd ../anomaly
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place your data file in `backend/data/`
   - Supported formats: Excel (.xlsx), CSV, Parquet

4. **Activate virtual environment**
run `venv/Scripts/activate`

Running the Application

Frontend Dashboard
```bash
cd frontend
streamlit run frontend.py
```   
The dashboard will open automatically in your browser

Anomaly Detection Dashboard
```bash
cd anomaly
streamlit run dashboard.py
```

Base URL
```
http://localhost:8501
```


🔬 Anomaly Detection System

Forensic Score System (SFS)

The platform uses a sophisticated multi-layer anomaly detection system:

1. **Feature Engineering**
   - Compute Density: IT assets per employee
   - Revenue per Employee: Financial efficiency
   - Asset Intensity: Revenue per IT asset
   - Total Compute Units: Aggregated hardware metrics

2. **Isolation Forest**
   - Unsupervised outlier detection
   - Optimized contamination rate via grid search
   - Silhouette score validation

3. **Network Contagion Analysis**
   - Corporate hierarchy graph construction
   - Risk propagation through parent-child relationships
   - Decay-factor modeling for risk transmission

4. **PCA-Based Risk Weighting**
   - Dimensionality reduction on risk factors
   - Optimal feature combination via explained variance
   - Dynamic threshold adjustment

Key Metrics

- **Forensic Score**: Anomaly intensity (higher = more anomalous)
- **Network Exposure**: Risk from connected entities
- **Infection Rate**: Percentage of anomalous subsidiaries in a conglomerate
- **Risk Multiplier**: Amplification factor for systemic risk

👥 Team & Contributors

- **Kevin**: Filtering, search, and corporate relations
- **Sophie**: Visualization and pivot table systems
- **Joseph**: Anomaly detection and forensic analysis
- **Danny**: Predictive analytics and revenue modeling
- **Joel**: Causal analytics and portfolio optimization

📄 License

MIT


**Built with ❤️ by Team Gobbling Data**
