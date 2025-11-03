"""Advanced PCA Analysis with Statistical Insights and LLM-style Interpretation.

This script performs:
1. Univariate analysis of each feature
2. Bivariate relationships and correlation analysis
3. PCA decomposition and interpretation
4. Statistical tests and distribution analysis
5. LLM-style insights and recommendations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataAnalyst:
    """Agent-like analyst that provides insights about the data and PCA results."""
    
    def analyze_distributions(self, df, features):
        insights = []
        for feature in features:
            # Basic stats
            stats_dict = df[feature].describe()
            skew = df[feature].skew()
            kurtosis = df[feature].kurtosis()
            
            # Normality test
            _, p_value = stats.normaltest(df[feature].dropna())
            
            insight = f"\nAnalysis of {feature}:\n"
            insight += f"- Mean: {stats_dict['mean']:.2f}, Median: {df[feature].median():.2f}\n"
            insight += f"- Std Dev: {stats_dict['std']:.2f}\n"
            insight += f"- Skewness: {skew:.2f} ({'right-skewed' if skew > 0 else 'left-skewed'})\n"
            insight += f"- Kurtosis: {kurtosis:.2f} ({'heavy-tailed' if kurtosis > 0 else 'light-tailed'})\n"
            insight += f"- Distribution: {'likely normal' if p_value > 0.05 else 'non-normal'} (p={p_value:.4f})\n"
            
            insights.append(insight)
        return "\n".join(insights)
    
    def analyze_correlations(self, corr_matrix):
        insights = ["\nCorrelation Analysis:"]
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                strength = 'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.3 else 'weak'
                direction = 'positive' if corr > 0 else 'negative'
                if abs(corr) > 0.1:  # Only report meaningful correlations
                    insights.append(f"- {col1} and {col2}: {strength} {direction} correlation ({corr:.3f})")
        return "\n".join(insights)
    
    def analyze_pca_results(self, pca, explained_var_ratio, feature_names):
        insights = ["\nPCA Insights:"]
        
        # Overall variance explained
        total_var = sum(explained_var_ratio)
        insights.append(f"- Total variance explained by all components: {total_var:.2%}")
        
        # Component-wise analysis
        for i, (ratio, components) in enumerate(zip(explained_var_ratio, pca.components_)):
            insights.append(f"\nPrincipal Component {i+1} ({ratio:.2%} variance):")
            # Sort features by absolute importance
            feature_importance = list(zip(feature_names, components))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Describe main drivers
            main_drivers = [f"{f} ({c:.3f})" for f, c in feature_importance if abs(c) > 0.3]
            insights.append(f"- Main drivers: {', '.join(main_drivers)}")
            
            # Interpret the component
            if i == 0:
                insights.append("- This component represents the primary pattern of variation in the data")
            elif i == 1:
                insights.append("- This component captures the second most important pattern, orthogonal to PC1")
                
        return "\n".join(insights)
    
    def suggest_preprocessing(self, df, features):
        suggestions = ["\nPreprocessing Suggestions:"]
        
        # Check for skewness
        for feature in features:
            skew = df[feature].skew()
            if abs(skew) > 1:
                suggestions.append(f"- {feature} is significantly skewed ({skew:.2f}). Consider log transformation.")
        
        # Check for scaling needs
        scales = df[features].std()
        if scales.max() / scales.min() > 10:
            suggestions.append("- Features have very different scales. Standardization is recommended.")
            
        # Missing values
        missing = df[features].isnull().sum()
        if missing.any():
            suggestions.append("\nMissing Value Strategy:")
            for feature in features:
                if missing[feature] > 0:
                    pct = (missing[feature] / len(df)) * 100
                    suggestions.append(f"- {feature}: {missing[feature]} missing values ({pct:.1f}%)")
                    if pct < 5:
                        suggestions.append("  → Consider median imputation")
                    elif pct < 15:
                        suggestions.append("  → Consider KNN or iterative imputation")
                    else:
                        suggestions.append("  → High missing rate, consider feature engineering or dropping")
        
        return "\n".join(suggestions)

def main():
    # Load data
    df = pd.read_csv('enhanced_data.csv')
    sensors = ['sensor1', 'sensor2', 'sensor3']
    
    # Create analysis agent
    analyst = DataAnalyst()
    
    # Setup plotting style
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 25))
    
    # 1. Univariate Analysis
    print("Performing univariate analysis...")
    for i, sensor in enumerate(sensors, 1):
        plt.subplot(6, 2, i)
        plt.hist(df[sensor].dropna(), bins=30, density=True, alpha=0.7)
        plt.title(f'Distribution of {sensor}')
        plt.subplot(6, 2, i+3)
        stats.probplot(df[sensor].dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {sensor}')
    
    # 2. Bivariate Analysis
    print("Analyzing bivariate relationships...")
    plt.subplot(6, 2, 7)
    plt.scatter(df['sensor1'], df['sensor2'], c=df['cluster'], alpha=0.6, cmap='tab10')
    plt.title('sensor1 vs sensor2 (colored by cluster)')
    
    plt.subplot(6, 2, 8)
    plt.scatter(df['sensor2'], df['sensor3'], c=df['cluster'], alpha=0.6, cmap='tab10')
    plt.title('sensor2 vs sensor3 (colored by cluster)')
    
    # 3. Correlation Analysis
    print("Computing correlations...")
    plt.subplot(6, 2, 9)
    correlation_matrix = df[sensors + ['pca1', 'pca2']].corr()
    plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title('Feature Correlation Matrix')
    
    # 4. PCA Analysis
    print("Performing PCA analysis...")
    # Prepare data
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(df[sensors])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot explained variance
    plt.subplot(6, 2, 10)
    explained_var = pca.explained_variance_ratio_ * 100
    plt.bar(range(1, len(explained_var) + 1), explained_var)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.title('Scree Plot: Explained Variance by Component')
    
    # 5. Cluster Analysis in PCA Space
    plt.subplot(6, 2, 11)
    scatter = plt.scatter(df['pca1'], df['pca2'], c=df['cluster'], 
                         cmap='tab10', alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Clusters in PCA Space')
    plt.colorbar(scatter, label='Cluster')
    
    # Save the full analysis plot
    plt.tight_layout()
    plt.savefig('advanced_analysis.png')
    plt.close()
    
    # 6. Generate Insights
    print("\n=== Statistical Analysis and Insights ===")
    print(analyst.analyze_distributions(df, sensors))
    print(analyst.analyze_correlations(correlation_matrix))
    print(analyst.analyze_pca_results(pca, pca.explained_variance_ratio_, sensors))
    print(analyst.suggest_preprocessing(df, sensors))
    
    # 7. Summary Statistics
    print("\n=== Summary Statistics ===")
    print(df[sensors].describe())
    
    # 8. Cluster Characteristics
    print("\n=== Cluster Characteristics ===")
    cluster_stats = df.groupby('cluster')[sensors].agg(['mean', 'std']).round(2)
    print(cluster_stats)

if __name__ == '__main__':
    main()

# 1. Scree Plot (Explained Variance)
plt.figure(figsize=(15, 5))
plt.subplot(131)
explained_var = pca.explained_variance_ratio_ * 100
plt.bar(range(1, len(explained_var) + 1), explained_var)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance by Component')

# 2. Feature Contributions (Component Loadings)
plt.subplot(132)
loadings = pca.components_
plt.imshow(loadings, cmap='RdBu', aspect='auto')
plt.xticks(range(len(sensors)), sensors)
plt.yticks(range(len(sensors)), [f'PC{i+1}' for i in range(len(sensors))])
plt.colorbar(label='Loading Strength')
plt.title('PCA Component Loadings')

# 3. Data in PCA Space colored by clusters
plt.subplot(133)
scatter = plt.scatter(df['pca1'], df['pca2'], c=df['cluster'], 
                     cmap='tab10', alpha=0.6)
plt.xlabel('First Principal Component (48.26%)')
plt.ylabel('Second Principal Component (27.84%)')
plt.title('Data in PCA Space\nColored by Cluster')
plt.colorbar(scatter, label='Cluster')

plt.tight_layout()
plt.savefig('pca_analysis.png')
plt.close()

# Print component interpretations
print("\nPCA Component Interpretations:")
for i, (comp, var) in enumerate(zip(loadings, explained_var), 1):
    print(f"\nPrincipal Component {i} ({var:.2f}% variance):")
    # Sort features by absolute loading value
    feature_loadings = list(zip(sensors, comp))
    feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
    for feature, loading in feature_loadings:
        # Show direction and strength
        direction = "positively" if loading > 0 else "negatively"
        strength = abs(loading)
        print(f"{feature}: {loading:6.3f} ({direction} correlated, strength={strength:.3f})")

# Print feature correlations
print("\nFeature Correlations:")
correlations = pd.DataFrame(X_imputed, columns=sensors).corr()
print(correlations)