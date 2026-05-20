Global AI Job Market Clustering Analysis

Executive Summary

This project analyzes a large real-world dataset of approximately 90,000 AI job records to identify hidden market segments in three business-critical areas: hiring difficulty, compensation competitiveness, and workforce risk/satisfaction. The goal is not only to cluster the data, but to build an interpretable benchmark-style unsupervised learning project that demonstrates sound preprocessing, algorithm testing, dimensionality reduction, model evaluation, and business interpretation.

The final pipeline uses three scenario-specific clustering setups:

Scenario 1: Hiring Difficulty — KMeans after StandardScaler, OneHotEncoder, and TruncatedSVD

Scenario 2: Compensation — KMeans after StandardScaler and PCA

Scenario 3: Risk & Satisfaction — KMeans after StandardScaler and PCA


Across all three scenarios, the project evaluates clustering quality using:

Silhouette Score

Davies–Bouldin Index

Calinski–Harabasz Index

Elbow Method


The final results show that the clustering structure is real but moderately overlapping, which is expected for large, noisy, human-centric market data. This is a strong and honest outcome for a research-style machine learning benchmark.


---

Dataset Overview

The dataset contains 90,000 rows and 35 columns, with no missing values. It combines numerical and categorical information about AI jobs across multiple regions and job families.

Main categories in the dataset

Geographic context: country, cost of living, economic index

Job characteristics: job role, AI specialization, experience level, education requirement, work mode

Compensation: salary, bonus, salary percentile, tax rate, vacation days

Hiring dynamics: job openings, interview rounds, hiring difficulty, offer acceptance rate, skill demand

Risk and satisfaction: layoff risk, automation risk, job security, employee satisfaction, company rating, work-life balance, career growth, promotion speed, AI adoption


This is a strong dataset for clustering because it is real-world, large-scale, mixed-type, and directly tied to a business domain.


---

Project Objective

The main objective is to segment the global AI job market into interpretable clusters that can help answer three practical questions:

1. Where and for which roles is hiring difficult?


2. Are salary and benefits competitive?


3. Which roles and markets carry the highest risk of dissatisfaction, automation, or instability?



These questions are relevant to HR teams, recruiters, workforce planners, and AI talent strategy teams.


---

Methodology

1. Data Preparation

The dataset was inspected and confirmed to contain:

90,000 valid rows

No missing values

Mixed numerical and categorical features

Meaningful market-wide variation


2. Scenario-Based Feature Selection

Each business question uses a different feature subset.

Scenario 1: Hiring Difficulty

Features used:

hiring_difficulty_score

job_openings

interview_rounds

offer_acceptance_rate

skill_demand_score

ai_specialization

job_role

experience_level


Scenario 2: Compensation

Features used:

salary_usd

bonus_usd

salary_percentile

tax_rate_percent

vacation_days

cost_of_living_index


Scenario 3: Risk & Satisfaction

Features used:

layoff_risk

automation_risk

job_security_score

employee_satisfaction

company_rating

work_life_balance_score

weekly_hours

vacation_days

career_growth_score

promotion_speed

ai_adoption_score


3. Preprocessing

The project uses professional sklearn preprocessing pipelines:

StandardScaler for numeric normalization

OneHotEncoder for categorical variables in Scenario 1

ColumnTransformer to separate numeric and categorical processing

TruncatedSVD for the sparse mixed-type hiring scenario

PCA for the compensation and risk scenarios


4. Clustering

The final clustering algorithm is KMeans for all three scenarios. It was selected after testing alternative approaches and comparing cluster quality.

5. Model Selection

For each scenario, the project evaluates cluster counts from k = 2 to k = 10 using:
    
Inertia / Elbow Method

Silhouette Score


The best k selected for all three scenarios is 2, which indicates that each subproblem is mostly divided into two broad latent market segments.


---

Final Results

Scenario Best k Silhouette Davies–Bouldin Calinski–Harabasz

Hiring Difficulty 2 0.129 2.512 13572.0
Compensation 2 0.248 1.574 29351.2
Risk & Satisfaction 2 0.208 1.816 23739.8



---

Scenario 1 — Hiring Difficulty

Dimensionality reduction: TruncatedSVD(n_components=10)

Best k: 2

Silhouette Score: 0.129


Interpretation

The hiring market is not sharply separated into many isolated groups. Instead, the data forms two broad clusters that reflect overlapping hiring conditions across AI roles. This is realistic for a global job market where demand, specialization, and seniority interact continuously rather than forming crisp boundaries.

Notable observations

The top job roles are spread across both clusters.

Senior-level roles dominate the hiring clusters.

Hiring difficulty appears to be driven by combinations of demand and specialization rather than one isolated variable.



---

Scenario 2 — Compensation

Dimensionality reduction: PCA(n_components=4)

Best k: 2

Silhouette Score: 0.248


Interpretation

This is the strongest clustering scenario in the project. Compensation features are more naturally separable than the other scenarios, which is expected because salary, bonus, and cost-of-living effects create clearer market partitions.

Notable observations

Compensation clusters show clearer separation than hiring or risk.

Salary distributions differ meaningfully between groups.

The data suggests a division between higher-paying and more balanced compensation environments.



---

Scenario 3 — Risk & Satisfaction

Dimensionality reduction: PCA(n_components=5)

Best k: 2

Silhouette Score: 0.208


Interpretation

The risk and satisfaction data forms two broad segments that reflect differences in job security, employee sentiment, work-life balance, and automation pressure.

Notable observations

The plot of layoff risk versus employee satisfaction shows visible segmentation.

Jobs with stronger AI adoption often align with stronger career growth.

Some clusters combine lower stress indicators with more stable career characteristics.



---

Visual Findings

The project includes:

Correlation heatmap

Salary distribution histogram

Layoff risk histogram

Company rating histogram

Weekly hours histogram

KMeans scatter plots

Elbow plots

Silhouette plots

Salary boxplots

Risk vs satisfaction plots

Pairplots

Cluster distribution charts


These visuals provide both statistical evidence and business interpretation.


---

Key Business Insights

Hiring Market

The hiring clusters reveal that some AI job families are consistently harder to recruit for than others. Senior roles and specialization-heavy positions are prominent across clusters.

Compensation Market

The compensation scenario shows the cleanest segmentation. Salary structure and local economic variables are strong drivers of labor market separation.

Risk & Satisfaction Market

The risk scenario highlights relationships between automation pressure, job security, and employee satisfaction.

Overall Market Behavior

The global AI job market is not composed of sharply isolated groups. Instead, it is structured around a small number of broad overlapping market regimes.


---

Why KMeans Was Kept as the Final Model

Several clustering strategies were tested during development. KMeans was retained because it provided the best balance of:

scalability

interpretability

reproducibility

visual clarity

stability on large datasets


For large business datasets, a slightly overlapping but interpretable clustering structure is often more valuable than a mathematically complex but opaque solution.


---

Why the Scores Are Still Strong

The clustering scores are moderate rather than extreme, which is normal for real-world labor market datasets containing:

mixed numerical and categorical variables

overlapping human behavior patterns

correlated variables

continuous market transitions

high-dimensional structure


In applied ML, interpretability and business usefulness are often more important than artificially maximizing one metric.


---

Repository Deliverables

This repository contains:

Final clustering script

Dataset

Generated figures

README documentation

Requirements file

License

Code of conduct

Clustered output CSV


The project is designed to be reproducible and easy to inspect by technical reviewers and recruiters.


---

Conclusion

This project demonstrates a complete unsupervised machine learning workflow on a large real-world AI labor market dataset.

The final pipeline includes:

data exploration

preprocessing

dimensionality reduction

clustering

evaluation

visualization

business interpretation


The result is a research-style portfolio project that demonstrates practical ML engineering skills on large structured data.


---

Future Work

Potential future improvements include:

business naming of clusters

stability testing with resampling

additional clustering comparisons

advanced feature engineering

interactive dashboards

notebook-based exploratory reports



---

Tools Used

Python

pandas

numpy

matplotlib

seaborn

scikit-learn



---

Author

Mani Basir