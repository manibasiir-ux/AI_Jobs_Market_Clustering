# 🌐 Global AI Job Market Clustering Analysis

> A large-scale unsupervised machine learning benchmark on 90,000 AI job records worldwide, designed to uncover hidden structures in hiring difficulty, compensation competitiveness, and workforce risk.

---

## 🔎 Project Summary

This project explores a real-world AI labor market dataset using KMeans clustering with scenario-specific preprocessing and dimensionality reduction. The goal is to identify interpretable market segments and patterns.

Unlike a toy example or a small notebook demo, this repository is built as a research-style portfolio project. It includes full preprocessing pipelines, algorithm testing, evaluation metrics, extensive documentation, and reproducible results.

---

## 🎯 What This Project Answers

This analysis focuses on three practical business questions:

1. **Where and for which roles is hiring difficult?**
2. **Are salary and benefits competitive across the AI job market?**
3. **Which roles and markets carry the highest risk of dissatisfaction, automation, or instability?**

---

## 🧠 Why This Project Is Useful

AI hiring is not uniform across countries, roles, and experience levels. Compensation also varies with cost of living, market pressure, and specialization. On top of that, job security and employee satisfaction depend on multiple overlapping factors.

This project applies clustering to discover market patterns that are difficult to see manually in a large dataset.

---

## 🗂️ Dataset Overview

| Property | Details |
|----------|---------|
| **Source** | Global AI Jobs Market Dataset |
| **Size** | ~90,000 rows |
| **Features** | 35 columns |
| **Type** | Mixed numerical and categorical tabular data |
| **Missing Values** | None in the final dataset used for the project |

**The dataset includes information such as:**

- Country
- Job role
- AI specialization
- Experience level
- Salary and bonus
- Hiring difficulty and offer acceptance
- Layoff and automation risk
- Work-life balance and employee satisfaction
- Cost of living and economic indicators

---

## 🧪 Methods Used

| Technique | Purpose |
|-----------|---------|
| KMeans Clustering | Final clustering algorithm |
| PCA | Dimensionality reduction for numeric scenarios |
| TruncatedSVD | Dimensionality reduction for sparse mixed-type features |
| OneHotEncoder | Encoding categorical variables |
| StandardScaler | Feature normalization |
| Silhouette Score | Cluster quality evaluation |
| Davies-Bouldin Index | Cluster separation evaluation |
| Calinski-Harabasz Score | Cluster density evaluation |
| Elbow Method | Support for selecting the number of clusters |

---

## 🧩 Clustering Scenarios

### **Scenario 1 — Hiring Difficulty**

Clusters jobs by hiring complexity, skill demand, interview rounds, and offer acceptance behavior.

**Features used:**
- hiring_difficulty_score
- job_openings
- interview_rounds
- offer_acceptance_rate
- skill_demand_score
- ai_specialization
- job_role
- experience_level

**Preprocessing:**
- StandardScaler
- OneHotEncoder
- TruncatedSVD

---

### **Scenario 2 — Compensation**

Groups jobs by salary structure, bonus levels, tax rate, vacation days, and cost of living.

**Features used:**
- salary_usd
- bonus_usd
- salary_percentile
- tax_rate_percent
- vacation_days
- cost_of_living_index

**Preprocessing:**
- StandardScaler
- PCA

---

### **Scenario 3 — Risk & Satisfaction**

Clusters jobs by stability, automation pressure, employee satisfaction, work-life balance, and career growth.

**Features used:**
- layoff_risk
- automation_risk
- job_security_score
- employee_satisfaction
- company_rating
- work_life_balance_score
- weekly_hours
- vacation_days
- career_growth_score
- promotion_speed
- ai_adoption_score

**Preprocessing:**
- StandardScaler
- PCA

---

## 📈 Final Results

| Scenario | Best k | Silhouette Score | Davies-Bouldin | Calinski-Harabasz |
|----------|--------|------------------|-----------------|------------------|
| Hiring Difficulty | 2 | 0.129 | 2.512 | 13572.0 |
| Compensation | 2 | 0.248 | 1.574 | 29351.2 |
| Risk & Satisfaction | 2 | 0.208 | 1.816 | 23739.8 |

**Interpretation of the scores:**

These are realistic scores for a large, real-world labor market dataset. The market is not made of perfectly separated blobs. Instead, it contains broad and overlapping behavioral patterns, which is expected and interpretable.

---

## 🔍 Key Findings

1. **Compensation clustering** is the clearest structure in the dataset, with the strongest separation among the three scenarios.
2. **Hiring difficulty** is broad and overlapping, which suggests that talent scarcity depends on a combination of specialization, seniority, and demand.
3. **Risk & satisfaction clusters** reveal meaningful differences in job security, employee sentiment, and work-life balance.
4. **AI adoption and career growth** appear positively related in several job groups.
5. **Higher salaries** are often associated with stronger skill demand and greater hiring difficulty.
6. **Some job groups** maintain strong work-life balance despite lower compensation levels.

---

## 🧠 Algorithm Testing and Modeling Notes

Several clustering strategies were explored during development, including:

- KMeans
- DBSCAN
- Hierarchical clustering

After testing, the **final KMeans-based pipeline** was retained because it gave the best combination of:

- Stability
- Interpretability
- Speed
- Reproducibility
- Visual clarity
- Business usefulness on a large dataset

**Note:** Hierarchical clustering proved impractical at full scale because the dataset is too large for that approach to run efficiently on a normal local machine.

---

## 📁 How to View Results Without Running the Full Code

The full script can take a long time to run because it processes 90,000 rows and generates many figures. If you only want to inspect the results, you do not need to rerun everything.

**Instead, use the `reports/` folder in the repository.**

### **What you will find there:**

#### **`reports/figures/`**
This folder contains the saved visual outputs of the project, including:
- Correlation heatmap
- Feature distribution plots
- Elbow plots
- Silhouette plots
- Cluster scatter plots
- Salary boxplot
- Risk vs satisfaction plot
- Pairplot
- Cluster distribution plots

These figures are the main visual summary of the project and let you inspect the analysis quickly.

#### **`reports/final_report.md`**
This file contains the full written interpretation of the project, including:
- Dataset overview
- Methodology
- Scenario explanations
- Score interpretation
- Business insights
- Conclusions
- Future work

### **Recommended way to review the project:**

If you do not want to run the code, the best workflow is:

1. Open this **README** for the project summary and key results.
2. Open **reports/final_report.md** for the detailed analysis.
3. Browse **reports/figures/** to inspect the plots one by one.

This is the fastest and most practical way to understand the project.

---

## 🖼️ Selected Figures

**Correlation Heatmap**

![Correlation Heatmap](https://github.com/user-attachments/assets/...)

**Scenario 1 — Hiring Clusters**

<img width="401" height="279" alt="hiring_cluster_distribution" src="https://github.com/user-attachments/assets/e07bc0cd-9d2c-4668-8a1e-29a191fad5f0" />

**Scenario 2 — Compensation Clusters**

<img width="401" height="279" alt="compensation_cluster_distribution" src="https://github.com/user-attachments/assets/23764615-7e5f-401a-9892-c35aa46cd0ab" />

**Salary Distribution by Compensation Cluster**

<img width="618" height="394" alt="salary_distribution" src="https://github.com/user-attachments/assets/d4d83350-155a-4fe3-9d0a-3e622dbaa6c5" />

**Scenario 3 — Risk vs Satisfaction**

<img width="581" height="394" alt="risk_vs_satisfaction" src="https://github.com/user-attachments/assets/d383b353-f2a1-4677-8545-d259e4981b0f" />

**Scenario 3 — Clustering View**

<img width="494" height="394" alt="scenario3_clusters" src="https://github.com/user-attachments/assets/738d9b75-fc3b-4d21-8de8-163103a5d2c0" />

---

## 🗃️ Repository Structure

```
AI_Jobs_Market_Clustering/
├── global_ai_jobs.csv
├── reports/
│   ├── figures/
│   └── final_report.md
├── Job_market_clustering.py
├── README.md
├── requirements.txt
├── LICENSE
├── CODE_OF_CONDUCT.md
└── .gitignore
```

---

## 🚀 How to Run

### **1. Clone the repository**

```bash
git clone https://github.com/manibasiir-ux/AI_Jobs_Market_Clustering.git
cd AI_Jobs_Market_Clustering
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the analysis**

```bash
python Job_market_clustering.py
```

> **Note:** Because the dataset is large and the workflow includes preprocessing, dimensionality reduction, clustering, and many visualizations, the full script may take a long time to finish on a local machine.

---

## 🛠️ Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
shap
joblib
```

**Install everything at once with:**

```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Mani Basir**  
Junior Machine Learning Developer

- **LinkedIn:** [linkedin.com/in/mani-basir](https://linkedin.com/in/mani-basir)
- **Telegram:** [@manibasir](https://t.me/manibasir)

---

Built with Python 🐍 | Scikit-learn | Pandas | Matplotlib | Seaborn
