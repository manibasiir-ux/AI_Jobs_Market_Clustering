# Global AI Job Market Clustering Analysis

A large-scale unsupervised ML benchmark on 90,000 AI job records to discover hidden patterns in hiring difficulty, compensation, and workforce risk.

## Project Overview
This project analyzes a real-world AI jobs dataset using clustering techniques to segment the market into interpretable groups. The goal is to identify patterns in hiring difficulty, compensation competitiveness, and risk/satisfaction stability.

## Dataset
- 90,000 rows
- 35 columns
- No missing values
- Mixed numerical and categorical features
- Real-world, large-scale, structured tabular data

## Problem Statement
Can we discover meaningful segments in the global AI jobs market that help us understand:
- where hiring is difficult,
- which compensation packages are competitive,
- and where workforce risk is highest?

## Methodology
### Scenario 1: Hiring Difficulty
Features:
- hiring_difficulty_score
- job_openings
- interview_rounds
- offer_acceptance_rate
- skill_demand_score
- ai_specialization
- job_role
- experience_level

Preprocessing:
- StandardScaler
- OneHotEncoder
- TruncatedSVD
- KMeans clustering

### Scenario 2: Compensation
Features:
- salary_usd
- bonus_usd
- salary_percentile
- tax_rate_percent
- vacation_days
- cost_of_living_index

Preprocessing:
- StandardScaler
- PCA
- KMeans clustering

### Scenario 3: Risk & Satisfaction
Features:
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

Preprocessing:
- StandardScaler
- PCA
- KMeans clustering

## Evaluation Metrics
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Elbow Method

## Results
| Scenario | Best k | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|---:|
| Hiring Difficulty | 2 | 0.129 | 2.512 | 13572.0 |
| Compensation | 2 | 0.248 | 1.574 | 29351.2 |
| Risk & Satisfaction | 2 | 0.208 | 1.816 | 23739.8 |

## Key Findings
- Compensation clusters show the strongest separation among the three scenarios.
- Hiring clusters are broad, which is expected in a real market dataset with overlapping role structures.
- Risk clusters capture meaningful variation in job security, employee satisfaction, and work-life balance.
- AI adoption and career growth show useful relationships across several job groups.

## Visualizations
- Correlation heatmap
- Cluster distribution charts
- PCA/SVD cluster scatter plots
- Salary distribution boxplot
- Risk vs satisfaction scatter plot
- Pairplot of selected variables

## Repository Structure
`text
...