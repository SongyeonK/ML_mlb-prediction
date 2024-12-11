import seaborn as sns
import matplotlib.pyplot as plt

def plot_missing_values(df):
    """결측치 시각화."""
    import missingno as msno
    msno.matrix(df)
    plt.title('Missing Values Visualization')
    plt.show()

def plot_correlation_heatmap(df, cols):
    """상관관계 히트맵 생성."""
    corr = df[cols].corr(method='pearson')
    sns.heatmap(corr, annot=True, cmap='RdYlBu', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.show()

def plot_scatter(df, x, y):
    """산점도 생성."""
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f'{x} vs {y}')
    plt.show()
