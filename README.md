Architecture
* Frontend: React (with a library like Ag-Grid or Perspective.js for the pivot tables—do *not* build this from scratch) + a Charting Lib (Recharts or Plotly).
* Backend: FastAPI (Python). It integrates natively with your AI/ML models and handles async requests for the dashboard well.
* Data Layer: A clean, processed CSV/Parquet file or a lightweight SQL database (SQLite/PostgreSQL) that everyone pulls from.

Key Features
(1)
1. (kevin) filters by specific categories
2. (kevin) search for comparables for each company
2.1 (kevin) find subsidiaries by parent company / vice versa

(2 & 3)
3. (sophie) tableau-esque feature to construct pivot tables and visuals for each company

(3 & 4)
4. (joseph) anomaly detection
5. (danny) predictive analytics on revenue

(4)
6. (joel) causal impact analytics

(5)
7. (joel) portfolio diversification assessor + LLM to suggest key risks for each portfolio
