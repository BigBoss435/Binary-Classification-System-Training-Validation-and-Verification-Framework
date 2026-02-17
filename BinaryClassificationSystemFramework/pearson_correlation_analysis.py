import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def analyze_and_plot_isic_data(file_path='data/ISIC_2020_Training_GroundTruth_v2.csv'):
    """
    Performs comprehensive analysis of ISIC 2020 melanoma dataset including:
    - Data loading and preprocessing
    - Statistical correlation analysis
    - Multiple visualization types
    - Age-based analysis
    - Distribution analysis

    Parameters:
        file_path (str): Path to the ISIC 2020 dataset CSV file
    """
    # Set global plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })

    try:
        # 1. Load and examine the dataset
        df = pd.read_csv(file_path)
        print("Dataset Summary:")
        print("-" * 50)
        print(f"Total samples: {len(df)}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nDataset Information:")
        df.info()

        # 2. Handle missing values
        initial_nans = df['age_approx'].isnull().sum()
        if initial_nans > 0:
            df['age_approx'].fillna(df['age_approx'].mean(), inplace=True)
            print(f"\nHandled {initial_nans} missing values in age_approx")

        # 3. Statistical Analysis
        if 'age_approx' in df.columns and 'target' in df.columns:
            # Pearson Correlation
            pearson_corr, pearson_p = pearsonr(df['age_approx'], df['target'])
            # Spearman Correlation
            spearman_corr, spearman_p = spearmanr(df['age_approx'], df['target'])

            print("\nCorrelation Analysis:")
            print("-" * 50)
            print(f"Pearson Correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
            print(f"Spearman Correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")

        # 4. Visualization Section
        print("\nGenerating visualizations...")

        # 4.1 Age vs Malignancy Relationship
        plt.figure(figsize=(12, 7))
        sns.regplot(
            x='age_approx', y='target', data=df, logistic=True, ci=None,
            scatter_kws={'alpha': 0.1, 's': 30, 'edgecolors': 'none'},
            line_kws={'color': 'red', 'linewidth': 2}
        )
        plt.title('Age vs Melanoma Risk', pad=20)
        plt.xlabel('Patient Age (years)')
        plt.ylabel('Diagnosis (0=Benign, 1=Malignant)')
        plt.yticks([0, 1], ['Benign', 'Malignant'])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.figtext(0.02, -0.1, 'Red line shows the predicted probability of malignancy based on age',
                    fontsize=12, ha='left')
        plt.tight_layout()
        plt.show()

        # 4.2 Age Distribution
        plt.figure(figsize=(12, 7))
        # Create a new column with string labels for better control
        df['diagnosis'] = df['target'].map({0: 'Benign', 1: 'Malignant'})
        sns.histplot(
            data=df,
            x='age_approx',
            hue='diagnosis',  # Use the new string labels
            bins=30,
            kde=True,
            element='step',
            palette={'Benign': 'green', 'Malignant': 'red'}
        )
        plt.title('Age Distribution by Diagnosis', pad=20)
        plt.xlabel('Age (years)')
        plt.ylabel('Number of Cases')
        plt.legend(title='Diagnosis', title_fontsize=14, fontsize=12)
        plt.tight_layout()
        plt.show()

        # 4.3 Class Distribution
        plt.figure(figsize=(10, 6))
        class_counts = df['target'].value_counts()
        sns.barplot(x=['Benign', 'Malignant'], y=class_counts.values,
                    palette=['green', 'red'])
        plt.title('Distribution of Diagnoses', pad=20)
        plt.ylabel('Number of Cases')
        for i, v in enumerate(class_counts.values):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

        # 4.4 Age Statistics by Diagnosis
        plt.figure(figsize=(10, 7))
        sns.boxplot(x='target', y='age_approx', data=df,
                    palette=['green', 'red'])
        plt.title('Age Distribution by Diagnosis Type', pad=20)
        plt.xlabel('Diagnosis')
        plt.ylabel('Age (years)')
        plt.xticks([0, 1], ['Benign', 'Malignant'])
        plt.tight_layout()
        plt.show()

        # 4.5 Correlation Heatmap
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f",
                        cbar_kws={'label': 'Correlation Coefficient'})
            plt.title('Feature Correlation Heatmap', pad=20)
            plt.tight_layout()
            plt.show()

        # 4.6 Age Group Analysis
        df['age_group'] = pd.cut(
            df['age_approx'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['0-20', '21-40', '41-60', '61-80', '81-100']
        )
        age_group_stats = df.groupby('age_group')['target'].mean()

        # 4.7 Sex Distribution by Malignancy
        plt.figure(figsize=(10, 6))
        sex_mal = pd.crosstab(df['sex'], df['target'], normalize='index') * 100
        sex_mal.plot(kind='bar', stacked=True, color=['green', 'red'])
        plt.title('Sex Distribution by Diagnosis', pad=20)
        plt.xlabel('Sex')
        plt.ylabel('Percentage')
        plt.legend(title='Diagnosis', labels=['Benign', 'Malignant'])
        for i in range(len(sex_mal)):
            for j in range(len(sex_mal.columns)):
                plt.text(i, sex_mal.iloc[i, j] / 2 + (sex_mal.iloc[i, :j].sum()),
                         f'{sex_mal.iloc[i, j]:.1f}%', ha='center')
        plt.tight_layout()
        plt.show()

        # 4.8 Violin Plot of Age Distribution
        plt.figure(figsize=(10, 7))
        sns.violinplot(x='target', y='age_approx', data=df, palette=['green', 'red'])
        plt.title('Age Distribution by Diagnosis (Violin Plot)', pad=20)
        plt.xlabel('Diagnosis')
        plt.ylabel('Age (years)')
        plt.xticks([0, 1], ['Benign', 'Malignant'])
        plt.tight_layout()
        plt.show()

        # 4.9 Anatomical Site vs Malignancy
        plt.figure(figsize=(12, 6))
        site_mal = pd.crosstab(df['anatom_site_general_challenge'], df['target'], normalize='index') * 100
        site_mal.plot(kind='barh', stacked=True, color=['green', 'red'])
        plt.title('Anatomical Site vs Diagnosis', pad=20)
        plt.xlabel('Percentage')
        plt.ylabel('Anatomical Site')
        plt.legend(title='Diagnosis', labels=['Benign', 'Malignant'])
        for i in range(len(site_mal)):
            for j in range(len(site_mal.columns)):
                plt.text(site_mal.iloc[i, j] / 2 + (site_mal.iloc[i, :j].sum()),
                         i,
                         f'{site_mal.iloc[i, j]:.1f}%',
                         ha='center',
                         va='center')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        age_group_stats.plot(kind='bar', color='skyblue')
        plt.title('Melanoma Risk by Age Group', pad=20)
        plt.ylabel('Risk Rate')
        plt.xlabel('Age Group')
        plt.xticks(rotation=45)
        for i, v in enumerate(age_group_stats):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.show()

        print("\nAnalysis completed successfully!")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    analyze_and_plot_isic_data('data/ISIC_2020_Training_GroundTruth_v2.csv')