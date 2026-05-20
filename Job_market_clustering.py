"""
==========================================================
GLOBAL AI JOB MARKET CLUSTERING ANALYSIS
==========================================================

Author: Mani Basir

Project Goals:
- Analyze the global AI job market
- Discover hidden structures in AI jobs
- Compare hiring difficulty, compensation,
  and workforce risk using clustering

Techniques:
- KMeans Clustering
- PCA
- TruncatedSVD
- OneHotEncoding
- Feature Scaling
- Cluster Evaluation Metrics

Evaluation Metrics:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

Dataset Size:
~90,000 AI job records

==========================================================
"""

# =========================
# IMPORTS
# =========================

from math import pi
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (silhouette_score,davies_bouldin_score,calinski_harabasz_score)
from sklearn.decomposition import PCA, TruncatedSVD

# =========================
# GLOBAL STYLE
# =========================

sns.set_style("whitegrid")

plt.rcParams['figure.figsize'] = (10,6)

plt.rcParams['font.size'] = 11

# =========================
# HELPER FUNCTION
# =========================

def section(title):

    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)

# =========================
# LOAD DATA
# =========================

section("LOADING DATASET")

DF = pd.read_csv('global_ai_jobs')

print("Dataset Shape:", DF.shape)

print("\nColumns:\n")

print(DF.columns)

# ============================================================
# DATASET OVERVIEW
# ============================================================

section("DATASET OVERVIEW")

print("\nFirst 5 Rows:\n")

print(DF.head())

print("\nDataset Information:\n")

print(DF.info())

print("\nMissing Values:\n")

print(DF.isnull().sum())

# ============================================================
# HISTOGRAMS
# ============================================================

section("FEATURE DISTRIBUTIONS")

DF['salary_usd'].hist(bins=40)

plt.title("Salary Distribution")

plt.xlabel("Salary USD")

plt.ylabel("Frequency")

plt.savefig("salary_distribution.png",bbox_inches='tight')

plt.show()

DF['layoff_risk'].hist(bins=40)

plt.title("Layoff Risk Distribution")

plt.xlabel("Layoff Risk")

plt.ylabel("Frequency")

plt.savefig("layoff_risk_distribution.png",bbox_inches='tight')

plt.show()

DF['company_rating'].hist(bins=40)

plt.title("Company Rating Distribution")

plt.xlabel("Company Rating")

plt.ylabel("Frequency")

plt.savefig("company_rating_distribution.png",bbox_inches='tight')

plt.show()

DF['weekly_hours'].hist(bins=40)

plt.title("Weekly Hours Distribution")

plt.xlabel("Weekly Hours")

plt.ylabel("Frequency")

plt.savefig("weekly_hours_distribution.png",bbox_inches='tight')

plt.show()

# ============================================================
# CORRELATION HEATMAP
# ============================================================

section("CORRELATION HEATMAP")

plt.figure(figsize=(14,10))

sns.heatmap(
    DF.select_dtypes(include=np.number).corr(),
    cmap='coolwarm',
    center=0)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png",bbox_inches='tight')

plt.show()

# ============================================================
# ============================================================
# SCENARIO 1
# HIRING DIFFICULTY
# ============================================================
# ============================================================

section("SCENARIO 1 - HIRING DIFFICULTY")

# =========================
# FEATURE SELECTION
# =========================

MD1 = DF[['hiring_difficulty_score','job_openings',
          'interview_rounds','offer_acceptance_rate'
          ,'skill_demand_score','ai_specialization','job_role','experience_level']]

NumcolsMD1 = ['hiring_difficulty_score','job_openings',
              'interview_rounds','offer_acceptance_rate','skill_demand_score']

CatcolsMD1 = ['ai_specialization','job_role','experience_level']

# =========================
# PREPROCESSING
# =========================

NumMD_pipeline1 = Pipeline([('scaler', StandardScaler())])

CatMD_pipeline1 = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessorMd1 = ColumnTransformer([('num', NumMD_pipeline1, NumcolsMD1),
                                     ('cat', CatMD_pipeline1, CatcolsMD1)])

MDdata1 = preprocessorMd1.fit_transform(MD1)

# =========================
# DIMENSION REDUCTION
# =========================

svd1 = TruncatedSVD(n_components=10,random_state=42)

X1 = svd1.fit_transform(MDdata1)

print("Scenario 1 Shape:", X1.shape)

print("\nExplained Variance Ratio:\n")

print(svd1.explained_variance_ratio_)

print("\nTotal Explained Variance:")

print(np.sum(svd1.explained_variance_ratio_))

# =========================
# FIND BEST K
# =========================

K_range = range(2, 11)

inertias1 = []

sil_scores1 = []

for k in K_range:

    km = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=20,
        random_state=42)

    labels = km.fit_predict(X1)
    inertias1.append(km.inertia_)
    sil_scores1.append(silhouette_score(X1, labels))

# =========================
# ELBOW PLOT
# =========================

plt.figure(figsize=(7,4))

plt.plot(
    list(K_range),
    inertias1,
    marker='o')

plt.title("Scenario 1 - Elbow Method")

plt.xlabel("k")

plt.ylabel("Inertia")

plt.savefig("scenario1_elbow.png",bbox_inches='tight')

plt.show()

# =========================
# SILHOUETTE PLOT
# =========================

plt.figure(figsize=(7,4))

plt.plot(
    list(K_range),
    sil_scores1,
    marker='o')

plt.title("Scenario 1 - Silhouette Scores")

plt.xlabel("k")

plt.ylabel("Silhouette Score")

plt.savefig("scenario1_silhouette.png",bbox_inches='tight')

plt.show()

# =========================
# BEST K
# =========================

best_k1 = list(K_range)[int(np.argmax(sil_scores1))]

print("\nBest k:", best_k1)

# =========================
# FINAL MODEL
# =========================

K_means1 = KMeans(
    init='k-means++',
    n_clusters=best_k1,
    n_init=20,
    random_state=42)

K_means1.fit(X1)

labels1 = K_means1.labels_

# =========================
# SAVE LABELS
# =========================

DF['cluster_hiring'] = labels1

# =========================
# VISUALIZATION
# =========================

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X1[:,0],
    X1[:,1],
    c=labels1,
    cmap='viridis',
    s=20,
    alpha=0.7)

plt.title('Scenario 1 - KMeans Clustering')

plt.xlabel('Component 1')

plt.ylabel('Component 2')

plt.legend(
    *scatter.legend_elements(),
    title="Clusters")

centers1 = K_means1.cluster_centers_

plt.scatter(
    centers1[:,0],
    centers1[:,1],
    c='red',
    s=200,
    marker='X')

plt.savefig("scenario1_clusters.png",bbox_inches='tight')

plt.show()

# =========================
# METRICS
# =========================

silhouette1 = silhouette_score(X1, labels1)

davies_bouldin1 = davies_bouldin_score(X1, labels1)

calinski_harabasz1 = calinski_harabasz_score(X1, labels1)

print(f"\nSilhouette Score: {silhouette1:.3f}")

print(f"Davies-Bouldin Score: {davies_bouldin1:.3f}")

print(f"Calinski-Harabasz Score: {calinski_harabasz1:.1f}")

# ============================================================
# SCENARIO 1 CLUSTER SUMMARY
# ============================================================

section("SCENARIO 1 CLUSTER SUMMARY")


scenario1_summary = DF.groupby(
    'cluster_hiring'
)[
    [
        'hiring_difficulty_score',
        'job_openings',
        'offer_acceptance_rate',
        'skill_demand_score'
    ]
].mean()

print(scenario1_summary)

scenario1_summary.plot(
    kind='bar')


plt.title('Hiring Clusters Comparison')

plt.ylabel('Average Value')

plt.xticks(rotation=0)

plt.savefig("scenario1_cluster_summary.png",bbox_inches='tight')

plt.show()

print("\nMost Common Job Roles:\n")

print(
    DF.groupby('cluster_hiring')['job_role']
    .agg(lambda x: x.value_counts().index[0]))

print("\nMost Common Experience Levels:\n")

print(
    DF.groupby('cluster_hiring')['experience_level']
    .agg(lambda x: x.value_counts().index[0]))

# ============================================================
# SCENARIO 2
# ============================================================

section("SCENARIO 2 - COMPENSATION")

MD2 = DF[['salary_usd','bonus_usd','salary_percentile','tax_rate_percent',
          'vacation_days','cost_of_living_index']]

NumcolsMD2 = MD2.columns.tolist()

NumMD2_pipeline = Pipeline([
    ('scaler', StandardScaler())])

preprocessorMd2 = ColumnTransformer([
    ('num', NumMD2_pipeline, NumcolsMD2)])

MD2data = preprocessorMd2.fit_transform(MD2)

pca2 = PCA(
    n_components=4,
    random_state=42)

X2 = pca2.fit_transform(MD2data)

print("Scenario 2 Shape:", X2.shape)

print("\nTotal Explained Variance:")

print(np.sum(pca2.explained_variance_ratio_))

K_range = range(2,11)

inertias2 = []

sil_scores2 = []

for k in K_range:

    km = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=20,
        random_state=42)

    labels2 = km.fit_predict(X2)

    inertias2.append(km.inertia_)

    sil_scores2.append(
        silhouette_score(X2, labels2))

plt.figure(figsize=(7,4))

plt.plot(
    list(K_range),
    inertias2,
    marker='o')

plt.title("Scenario 2 - Elbow Method")

plt.xlabel("k")

plt.ylabel("Inertia")

plt.savefig("scenario2_elbow.png",bbox_inches='tight')

plt.show()

plt.figure(figsize=(7,4))

plt.plot(
    list(K_range),
    sil_scores2,
    marker='o')

plt.title("Scenario 2 - Silhouette Scores")

plt.xlabel("k")

plt.ylabel("Silhouette Score")

plt.savefig("scenario2_silhouette.png",bbox_inches='tight')

plt.show()

best_k2 = list(K_range)[
    int(np.argmax(sil_scores2))]

print("\nBest k:", best_k2)

K_means2 = KMeans(
    init='k-means++',
    n_clusters=best_k2,
    n_init=20,
    random_state=42)

K_means2.fit(X2)

labels2 = K_means2.labels_

DF['cluster_compensation'] = labels2

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X2[:,0],
    X2[:,1],
    c=labels2,
    cmap='viridis',
    s=20,
    alpha=0.7)

plt.title('Scenario 2 - KMeans Clustering')

plt.xlabel('PCA Component 1')

plt.ylabel('PCA Component 2')

plt.legend(
    *scatter.legend_elements(),
    title="Clusters")

centers2 = K_means2.cluster_centers_

plt.scatter(
    centers2[:,0],
    centers2[:,1],
    c='red',
    s=200,
    marker='X')

plt.savefig("scenario2_clusters.png",bbox_inches='tight')

plt.show()

silhouette2 = silhouette_score(X2, labels2)

davies_bouldin2 = davies_bouldin_score(X2, labels2)

calinski_harabasz2 = calinski_harabasz_score(X2, labels2)

print(f"\nSilhouette Score: {silhouette2:.3f}")

print(f"Davies-Bouldin Score: {davies_bouldin2:.3f}")

print(f"Calinski-Harabasz Score: {calinski_harabasz2:.1f}")

# ============================================================
# SALARY BOXPLOT
# ============================================================

plt.figure(figsize=(10,6))

sns.boxplot(
    x='cluster_compensation',
    y='salary_usd',
    data=DF)

plt.title('Salary Distribution by Compensation Cluster')

plt.savefig("salary_cluster_boxplot.png",bbox_inches='tight')

plt.show()

# ============================================================
# SCENARIO 3
# ============================================================

section("SCENARIO 3 - RISK & SATISFACTION")

MD3 = DF[['layoff_risk','automation_risk','job_security_score',
          'employee_satisfaction','company_rating',
          'work_life_balance_score','weekly_hours','vacation_days',
          'career_growth_score','promotion_speed','ai_adoption_score']]

NumcolsMD3 = MD3.columns.tolist()

NumMD3_pipeline = Pipeline([
    ('scaler', StandardScaler())])

preprocessorMd3 = ColumnTransformer([
    ('num', NumMD3_pipeline, NumcolsMD3)])

MD3data = preprocessorMd3.fit_transform(MD3)

pca3 = PCA(
    n_components=5,
    random_state=42)

X3 = pca3.fit_transform(MD3data)

print("Scenario 3 Shape:", X3.shape)

print("\nTotal Explained Variance:")

print(np.sum(pca3.explained_variance_ratio_))

K_range = range(2,11)

inertias3 = []

sil_scores3 = []

for k in K_range:

    km = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=20,
        random_state=42)

    labels3 = km.fit_predict(X3)

    inertias3.append(km.inertia_)

    sil_scores3.append(
        silhouette_score(X3, labels3))

plt.figure(figsize=(7,4))

plt.plot(
    list(K_range),
    inertias3,
    marker='o')

plt.title("Scenario 3 - Elbow Method")

plt.xlabel("k")

plt.ylabel("Inertia")

plt.savefig("scenario3_elbow.png",bbox_inches='tight')

plt.show()

plt.figure(figsize=(7,4))

plt.plot(
    list(K_range),
    sil_scores3,
    marker='o')

plt.title("Scenario 3 - Silhouette Scores")

plt.xlabel("k")

plt.ylabel("Silhouette Score")

plt.savefig("scenario3_silhouette.png",bbox_inches='tight')

plt.show()

best_k3 = list(K_range)[
    int(np.argmax(sil_scores3))]

print("\nBest k:", best_k3)

K_means3 = KMeans(
    init='k-means++',
    n_clusters=best_k3,
    n_init=20,
    random_state=42)

K_means3.fit(X3)

labels3 = K_means3.labels_

DF['cluster_risk'] = labels3

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    X3[:,0],
    X3[:,1],
    c=labels3,
    cmap='viridis',
    s=20,
    alpha=0.7)

plt.title('Scenario 3 - KMeans Clustering')

plt.xlabel('PCA Component 1')

plt.ylabel('PCA Component 2')

plt.legend(
    *scatter.legend_elements(),
    title="Clusters")

centers3 = K_means3.cluster_centers_

plt.scatter(
    centers3[:,0],
    centers3[:,1],
    c='red',
    s=200,
    marker='X')

plt.savefig("scenario3_clusters.png",bbox_inches='tight')

plt.show()

silhouette3 = silhouette_score(X3, labels3)

davies_bouldin3 = davies_bouldin_score(X3, labels3)

calinski_harabasz3 = calinski_harabasz_score(X3, labels3)

print(f"\nSilhouette Score: {silhouette3:.3f}")

print(f"Davies-Bouldin Score: {davies_bouldin3:.3f}")

print(f"Calinski-Harabasz Score: {calinski_harabasz3:.1f}")

# ============================================================
# RISK VISUALIZATION
# ============================================================

plt.figure(figsize=(10,6))

plt.scatter(
    DF['layoff_risk'],
    DF['employee_satisfaction'],
    c=DF['cluster_risk'],
    cmap='viridis',
    alpha=0.5)

plt.xlabel('Layoff Risk')

plt.ylabel('Employee Satisfaction')

plt.title('Risk vs Satisfaction Clusters')

plt.colorbar(label='Cluster')

plt.savefig("risk_vs_satisfaction.png",bbox_inches='tight')

plt.show()

# ============================================================
# PAIRPLOT
# ============================================================

section("PAIRPLOT VISUALIZATION")

sample_df = DF.sample(
    2000,
    random_state=42)

sns.pairplot(
    sample_df[
        [
            'salary_usd',
            'bonus_usd',
            'company_rating',
            'employee_satisfaction',
            'cluster_compensation']],hue='cluster_compensation')

plt.savefig("pairplot_clusters.png",bbox_inches='tight')

plt.show()

# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

section("CLUSTER DISTRIBUTIONS")

plt.figure(figsize=(6,4))

DF['cluster_hiring'].value_counts().sort_index().plot(
    kind='bar')

plt.title('Hiring Clusters Distribution')



plt.xlabel('Cluster')

plt.ylabel('Count')

plt.savefig("hiring_cluster_distribution.png",bbox_inches='tight')

plt.show()

plt.figure(figsize=(6,4))

DF['cluster_compensation'].value_counts().sort_index().plot(kind='bar')

plt.title('Compensation Clusters Distribution')

plt.xlabel('Cluster')

plt.ylabel('Count')

plt.savefig("compensation_cluster_distribution.png",bbox_inches='tight')

plt.show()

plt.figure(figsize=(6,4))

DF['cluster_risk'].value_counts().sort_index().plot(
    kind='bar')

plt.title('Risk Clusters Distribution')

plt.xlabel('Cluster')

plt.ylabel('Count')

plt.savefig("risk_cluster_distribution.png",bbox_inches='tight')

plt.show()

# ============================================================
# TOP JOB ROLES
# ============================================================

section("TOP JOB ROLES PER HIRING CLUSTER")

for cluster in sorted(DF['cluster_hiring'].unique()):

    print(f"\nCluster {cluster}")

    print(
        DF[
            DF['cluster_hiring'] == cluster
        ]['job_role']
        .value_counts()
        .head(5))

# ============================================================
# SAVE FINAL DATASET
# ============================================================

DF.to_csv(
    'clustered_global_ai_jobs.csv',
    index=False)

print("\nClustered dataset saved successfully.")

# ============================================================
# FINAL BUSINESS INSIGHTS
# ============================================================

section("FINAL BUSINESS INSIGHTS")

print("""

1. Compensation clusters reveal clear separation between
high-income and balanced-income AI jobs.

2. Hiring clusters indicate differences in hiring difficulty,
skill demand, and offer acceptance rates.

3. Risk clusters demonstrate relationships between
automation risk, employee satisfaction, and job security.

4. AI adoption appears positively correlated with
career growth in several job categories.

5. Higher salaries are often associated with
higher hiring difficulty and stronger skill demand.

6. Some job groups maintain strong work-life balance
despite lower compensation levels.

""")