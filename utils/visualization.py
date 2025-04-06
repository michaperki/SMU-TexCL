"""
Visualization utilities for the SMU-Textron Cognitive Load dataset analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def set_plot_style(style: str = 'whitegrid', context: str = 'notebook', palette: str = 'viridis') -> None:
    """Set plotting style.
    
    Args:
        style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
        palette: Color palette
    """
    sns.set(style=style, context=context, palette=palette)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def save_or_show_plot(save_path: Optional[str] = None, message: str = "Figure saved", dpi: int = 300) -> None:
    """Save or show a plot.
    
    Args:
        save_path: Path to save the figure
        message: Message to print when saving
        dpi: Resolution for saving
    """
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"{message} to {save_path}")
    else:
        plt.show()


def create_custom_colormap(colors: List[str], name: str = 'custom_cmap') -> LinearSegmentedColormap:
    """Create a custom colormap.
    
    Args:
        colors: List of colors
        name: Name of the colormap
        
    Returns:
        Custom colormap
    """
    return LinearSegmentedColormap.from_list(name, colors)


def plot_feature_distributions(
    df: pd.DataFrame, 
    features: List[str], 
    hue: Optional[str] = None,
    palette: Optional[str] = None,
    kde: bool = True,
    bins: int = 20,
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot feature distributions.
    
    Args:
        df: DataFrame with features
        features: List of features to plot
        hue: Column to use for coloring
        palette: Color palette
        kde: Whether to include KDE curve
        bins: Number of histogram bins
        figsize: Figure size (width, height)
        save_path: Path to save the figure
    """
    # Calculate number of rows and columns for subplots
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # Plot each feature
    for i, feature in enumerate(features):
        if i < len(axes):
            if hue is None:
                sns.histplot(df[feature], kde=kde, bins=bins, ax=axes[i], palette=palette)
            else:
                sns.histplot(df, x=feature, hue=hue, kde=kde, bins=bins, ax=axes[i], palette=palette)
            axes[i].set_title(f'Distribution of {feature}')
            
            # Add statistics to the plot
            if hue is None:
                mean = df[feature].mean()
                std = df[feature].std()
                median = df[feature].median()
                axes[i].axvline(mean, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean:.2f}')
                axes[i].axvline(median, color='green', linestyle='-', alpha=0.8, label=f'Median: {median:.2f}')
                axes[i].legend()
                axes[i].text(0.05, 0.95, f'Std: {std:.2f}', transform=axes[i].transAxes, 
                           verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.5})
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_or_show_plot(save_path, "Feature distributions plot saved")


def plot_correlation_matrix(
    df: pd.DataFrame, 
    features: Optional[List[str]] = None,
    method: str = 'pearson',
    cmap: str = 'coolwarm',
    annot: bool = True,
    mask_upper: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> None:
    """Plot correlation matrix.
    
    Args:
        df: DataFrame with features
        features: List of features to include
        method: Correlation method ('pearson', 'kendall', 'spearman')
        cmap: Colormap
        annot: Whether to annotate cells
        mask_upper: Whether to mask the upper triangle
        figsize: Figure size (width, height)
        save_path: Path to save the figure
    """
    if features is not None:
        df = df[features]
    
    # Calculate correlation matrix
    corr = df.corr(method=method)
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool)) if mask_upper else None
    
    # Set up the matplotlib figure
    plt.figure(figsize=figsize)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        annot=annot, 
        fmt=".2f"
    )
    plt.title(f'Feature Correlation Matrix ({method.capitalize()})')
    
    save_or_show_plot(save_path, "Correlation matrix plot saved")
    
    return corr


def plot_feature_clusters(
    df: pd.DataFrame,
    features: List[str],
    method: str = 'pearson',
    cmap: str = 'coolwarm',
    save_path: Optional[str] = None
) -> None:
    """Plot feature correlation matrix with hierarchical clustering.
    
    Args:
        df: DataFrame with features
        features: List of features to include
        method: Correlation method ('pearson', 'kendall', 'spearman')
        cmap: Colormap
        save_path: Path to save the figure
    """
    # Select features
    data = df[features]
    
    # Compute correlation matrix
    corr = data.corr(method=method)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(14, 12))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True) if cmap == 'coolwarm' else cmap
    
    # Draw the heatmap with clustered rows and columns
    clustered_heatmap = sns.clustermap(
        corr,
        cmap=cmap,
        center=0,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        annot=True,
        fmt=".2f",
        figsize=(14, 12)
    )
    
    # Add title
    plt.suptitle(f'Clustered Feature Correlation Matrix ({method.capitalize()})', 
                 fontsize=16, y=1.02)
    
    save_or_show_plot(save_path, "Feature clusters plot saved")


def plot_pca_explained_variance(
    X: Union[pd.DataFrame, np.ndarray],
    n_components: int = 10,
    title: str = "PCA Explained Variance",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> PCA:
    """Plot PCA explained variance.
    
    Args:
        X: Feature matrix
        n_components: Number of components to plot
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure
        
    Returns:
        Fitted PCA object
    """
    # Fit PCA
    pca = PCA(n_components=min(n_components, X.shape[1]))
    pca.fit(X)
    
    # Calculate cumulative explained variance
    cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot explained variance
    plt.figure(figsize=figsize)
    
    # Bar plot for individual variance
    plt.bar(
        range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_, 
        alpha=0.7, 
        color='skyblue',
        label='Individual explained variance'
    )
    
    # Line plot for cumulative variance
    plt.step(
        range(1, len(cum_var_exp) + 1), 
        cum_var_exp, 
        where='mid', 
        color='red',
        label='Cumulative explained variance'
    )
    
    # Add thresholds
    for threshold in [0.7, 0.8, 0.9, 0.95]:
        if max(cum_var_exp) >= threshold:
            # Find the number of components needed to reach this threshold
            n_comp = np.argmax(cum_var_exp >= threshold) + 1
            plt.axhline(y=threshold, linestyle='--', color='gray', alpha=0.5)
            plt.text(n_comp + 0.1, threshold + 0.01, f'{threshold:.0%}: {n_comp} components',
                    verticalalignment='bottom')
    
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.ylim([0, 1.05])
    
    save_or_show_plot(save_path, "PCA explained variance plot saved")
    
    return pca


def plot_pca_components(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    feature_names: Optional[List[str]] = None,
    n_components: int = 2,
    plot_type: str = '2d',
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> Tuple[PCA, plt.Figure]:
    """Plot PCA components.
    
    Args:
        X: Feature matrix
        y: Target values for coloring
        feature_names: Names of features
        n_components: Number of components to use
        plot_type: Type of plot ('2d', '3d', 'biplot')
        figsize: Figure size (width, height)
        save_path: Path to save the figure
        
    Returns:
        Tuple of (PCA object, Figure)
    """
    # Get feature names if not provided
    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    elif feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    if plot_type == '2d' and n_components >= 2:
        # 2D scatter plot
        ax = fig.add_subplot(111)
        
        # Plot points
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, s=50)
        
        # Add legend if y has categorical values
        if y is not None and len(np.unique(y)) <= 10:
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend1)
        elif y is not None:
            cbar = plt.colorbar(scatter)
            cbar.set_label('Target Value')
        
        # Add labels and title
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title('PCA: 2D Projection')
        ax.grid(True)
        
    elif plot_type == '3d' and n_components >= 3:
        # 3D scatter plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
            c=y, cmap='viridis', alpha=0.8, s=50
        )
        
        # Add colorbar or legend
        if y is not None and len(np.unique(y)) <= 10:
            legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend1)
        elif y is not None:
            cbar = plt.colorbar(scatter)
            cbar.set_label('Target Value')
        
        # Add labels and title
        ax.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC 3 ({pca.explained_variance_ratio_[2]:.2%})')
        ax.set_title('PCA: 3D Projection')
        
    elif plot_type == 'biplot' and n_components >= 2:
        # Biplot (PCA with feature vectors)
        ax = fig.add_subplot(111)
        
        # Plot points
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5, s=30)
        
        # Add feature vectors
        for i, (name, vec) in enumerate(zip(feature_names, pca.components_.T)):
            # Scale the vectors for visibility
            multiplier = 5
            x = vec[0] * multiplier
            y = vec[1] * multiplier
            
            # Plot vector
            ax.arrow(0, 0, x, y, head_width=0.1, head_length=0.1, fc='red', ec='red')
            
            # Add feature name
            text_pos = (x * 1.1, y * 1.1)
            ax.text(text_pos[0], text_pos[1], name, color='red', fontsize=9)
        
        # Add circle for scale reference
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        
        # Add labels and title
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title('PCA: Biplot with Feature Vectors')
        ax.grid(True)
        
        # Ensure equal scaling
        ax.set_aspect('equal')
        
        # Set limits for better visualization
        lim = max(
            abs(X_pca[:, 0].max()), abs(X_pca[:, 0].min()),
            abs(X_pca[:, 1].max()), abs(X_pca[:, 1].min())
        ) * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    
    plt.tight_layout()
    save_or_show_plot(save_path, "PCA components plot saved")
    
    return pca, fig


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importances",
    top_n: Optional[int] = None,
    std: Optional[np.ndarray] = None,
    color: str = 'skyblue',
    horizontal: bool = True,
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot feature importances.
    
    Args:
        feature_names: Names of features
        importances: Importance values
        title: Plot title
        top_n: Number of top features to show
        std: Standard deviations for error bars
        color: Bar color
        horizontal: Whether to use horizontal bars
        figsize: Figure size (width, height)
        save_path: Path to save the figure
    """
    # Create DataFrame for easier handling
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Add standard deviations if provided
    if std is not None:
        importance_df['std'] = std
    
    # Select top N features if specified
    if top_n is not None:
        importance_df = importance_df.head(top_n)
    
    # Determine figure size based on number of features
    if figsize is None:
        if horizontal:
            figsize = (12, max(8, len(importance_df) * 0.4))
        else:
            figsize = (max(8, len(importance_df) * 0.4), 12)
    
    # Plot
    plt.figure(figsize=figsize)
    
    if horizontal:
        if std is not None and 'std' in importance_df.columns:
            bars = plt.barh(
                importance_df['feature'], 
                importance_df['importance'],
                xerr=importance_df['std'],
                color=color,
                alpha=0.8,
                error_kw={'ecolor': 'black', 'capsize': 5}
            )
        else:
            bars = plt.barh(
                importance_df['feature'], 
                importance_df['importance'],
                color=color,
                alpha=0.8
            )
        
        # Add values next to bars
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + bar.get_width() * 0.01,
                i,
                f'{bar.get_width():.3f}',
                ha='left',
                va='center'
            )
            
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
    else:
        if std is not None and 'std' in importance_df.columns:
            bars = plt.bar(
                importance_df['feature'], 
                importance_df['importance'],
                yerr=importance_df['std'],
                color=color,
                alpha=0.8,
                error_kw={'ecolor': 'black', 'capsize': 5}
            )
        else:
            bars = plt.bar(
                importance_df['feature'], 
                importance_df['importance'],
                color=color,
                alpha=0.8
            )
        
        # Add values on top of bars
        for bar in bars:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + bar.get_height() * 0.01,
                f'{bar.get_height():.3f}',
                ha='center',
                va='bottom'
            )
            
        plt.ylabel('Importance')
        plt.xlabel('Feature')
        plt.xticks(rotation=45, ha='right')
    
    plt.title(title)
    plt.grid(True, axis='x' if horizontal else 'y')
    plt.tight_layout()
    
    save_or_show_plot(save_path, "Feature importance plot saved")


def plot_learning_curves(
    history: Dict[str, List[float]],
    title: str = "Learning Curves",
    figsize: Tuple[int, int] = (10, 6),
    show_min_max: bool = True,
    legend_loc: str = 'best',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot learning curves.
    
    Args:
        history: Dictionary with training history
        title: Plot title
        figsize: Figure size (width, height)
        show_min_max: Whether to show min/max annotations
        legend_loc: Legend location
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for key, values in history.items():
        epochs = range(1, len(values) + 1)
        line, = ax.plot(epochs, values, 'o-', label=key)
        
        if show_min_max:
            min_val = min(values)
            min_idx = values.index(min_val)
            max_val = max(values)
            max_idx = values.index(max_val)
            
            # Annotate minimum and maximum points
            if 'loss' in key.lower():
                # For loss, only annotate minimum
                ax.annotate(
                    f'Min: {min_val:.4f}',
                    xy=(epochs[min_idx], min_val),
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=line.get_color()),
                    color=line.get_color()
                )
            else:
                # For metrics, only annotate maximum
                ax.annotate(
                    f'Max: {max_val:.4f}',
                    xy=(epochs[max_idx], max_val),
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=line.get_color()),
                    color=line.get_color()
                )
    
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend(loc=legend_loc)
    ax.grid(True)
    
    # Add epoch ticks
    ax.set_xticks(epochs)
    
    plt.tight_layout()
    save_or_show_plot(save_path, "Learning curves plot saved")
    
    return fig


def plot_time_series(
    time_series: List[np.ndarray],
    labels: List[str],
    sampling_rates: List[float],
    title: str = "Time Series Data",
    events: Optional[List[Dict[str, Any]]] = None,
    highlight_regions: Optional[List[Dict[str, Any]]] = None,
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot multiple time series.
    
    Args:
        time_series: List of time series arrays
        labels: List of labels for each time series
        sampling_rates: List of sampling rates for each time series
        title: Plot title
        events: List of event dictionaries with keys 'time', 'label', and optional 'color'
        highlight_regions: List of region dictionaries with keys 'start', 'end', 'label', and optional 'color', 'alpha'
        figsize: Figure size (width, height)
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    n_series = len(time_series)
    
    if figsize is None:
        figsize = (12, 3 * n_series)
    
    fig, axes = plt.subplots(n_series, 1, figsize=figsize, sharex=True)
    axes = [axes] if n_series == 1 else axes
    
    for i, (ts, label, fs) in enumerate(zip(time_series, labels, sampling_rates)):
        # Create time axis
        time = np.arange(len(ts)) / fs
        
        # Plot time series
        axes[i].plot(time, ts, label=f'{label} ({fs} Hz)')
        
        # Add events if provided
        if events is not None:
            for event in events:
                if event['time'] <= time[-1]:
                    color = event.get('color', 'red')
                    axes[i].axvline(event['time'], color=color, linestyle='--', alpha=0.7)
                    axes[i].text(
                        event['time'], 
                        axes[i].get_ylim()[1] * 0.95, 
                        event['label'], 
                        rotation=90, 
                        verticalalignment='top',
                        color=color
                    )
        
        # Add highlight regions if provided
        if highlight_regions is not None:
            for region in highlight_regions:
                if region['start'] <= time[-1]:
                    end = min(region['end'], time[-1])
                    color = region.get('color', 'yellow')
                    alpha = region.get('alpha', 0.3)
                    axes[i].axvspan(region['start'], end, color=color, alpha=alpha)
                    axes[i].text(
                        (region['start'] + end) / 2,
                        axes[i].get_ylim()[1] * 0.9,
                        region['label'],
                        horizontalalignment='center',
                        verticalalignment='top',
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2}
                    )
        
        # Add legend, labels, and grid
        axes[i].legend(loc='upper right')
        axes[i].set_ylabel(label)
        axes[i].grid(True)
        
        # Add statistics
        mean = np.mean(ts)
        std = np.std(ts)
        axes[i].text(
            0.01, 0.05, 
            f'Mean: {mean:.2f}\nStd: {std:.2f}', 
            transform=axes[i].transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2}
        )
    
    # Add common x-axis label
    axes[-1].set_xlabel('Time (s)')
    
    # Add title
    plt.suptitle(title)
    plt.tight_layout()
    
    save_or_show_plot(save_path, "Time series plot saved")
    
    return fig


def plot_time_series_animation(
    time_series: np.ndarray,
    sampling_rate: float,
    window_size: float = 10.0,
    step_size: float = 0.5,
    title: str = "Time Series Animation",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """Create an animation of a time series with a moving window.
    
    Args:
        time_series: Time series array
        sampling_rate: Sampling rate in Hz
        window_size: Window size in seconds
        step_size: Step size in seconds
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the animation (as .mp4 or .gif)
    """
    # Convert window and step sizes to samples
    window_samples = int(window_size * sampling_rate)
    step_samples = int(step_size * sampling_rate)
    
    # Create time axis
    time = np.arange(len(time_series)) / sampling_rate
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initial plot (full time series as background)
    ax.plot(time, time_series, 'k-', alpha=0.2)
    
    # Line that will show current window
    line, = ax.plot([], [], 'b-', linewidth=2)
    
    # Vertical lines to indicate window
    left_line = ax.axvline(-1, color='r', linestyle='--')
    right_line = ax.axvline(-1, color='r', linestyle='--')
    
    # Text to display the current time
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Set axis limits
    ax.set_xlim(0, time[-1])
    ax.set_ylim(np.min(time_series) * 1.1, np.max(time_series) * 1.1)
    
    # Add labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True)
    
    def init():
        """Initialize animation."""
        line.set_data([], [])
        left_line.set_xdata(-1)
        right_line.set_xdata(-1)
        time_text.set_text('')
        return line, left_line, right_line, time_text
    
    def animate(i):
        """Update animation for each frame."""
        # Calculate current window start and end
        start_idx = i * step_samples
        end_idx = min(start_idx + window_samples, len(time_series))
        
        # Update the data
        line.set_data(time[start_idx:end_idx], time_series[start_idx:end_idx])
        
        # Update window indicators
        left_line.set_xdata(time[start_idx])
        right_line.set_xdata(time[end_idx-1])
        
        # Update time text
        time_text.set_text(f'Time: {time[start_idx]:.2f}s - {time[end_idx-1]:.2f}s')
        
        return line, left_line, right_line, time_text
    
    # Create animation
    frames = (len(time_series) - window_samples) // step_samples + 1
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=frames, 
        interval=200, blit=True
    )
    
    # Save animation if path provided
    if save_path:
        if save_path.endswith('.gif'):
            writer = animation.PillowWriter(fps=10)
            anim.save(save_path, writer=writer)
        else:
            # Default to mp4
            writer = animation.FFMpegWriter(fps=10)
            if not save_path.endswith('.mp4'):
                save_path += '.mp4'
            anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    
    return anim


def plot_attention_weights(
    text_or_sequence: List[str],
    attention_weights: np.ndarray,
    title: str = "Attention Weights",
    annotate: bool = True,
    figsize: Tuple[int, int] = None,
    cmap: str = "YlGnBu",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot attention weights.
    
    Args:
        text_or_sequence: List of items (words, time points, etc.)
        attention_weights: Attention weight matrix
        title: Plot title
        annotate: Whether to annotate cells with values
        figsize: Figure size (width, height)
        cmap: Colormap
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    if figsize is None:
        # Adjust figsize based on number of elements
        n_elements = len(text_or_sequence)
        figsize = (max(12, n_elements * 0.5), 10)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = sns.heatmap(
        attention_weights,
        cmap=cmap,
        annot=annotate,
        fmt=".2f",
        xticklabels=text_or_sequence,
        yticklabels=[title],
        cbar=True,
        ax=ax
    )
    
    # Rotate x-tick labels if there are many elements
    if len(text_or_sequence) > 10:
        plt.xticks(rotation=45, ha='right')
    
    # Highlight the highest attention weight
    if attention_weights.shape[1] > 0:  # Make sure there's at least one column
        max_idx = np.argmax(attention_weights[0])
        ax.add_patch(plt.Rectangle(
            (max_idx, 0), 1, 1, fill=False, edgecolor='red', lw=2
        ))
        
    plt.title(title)
    plt.tight_layout()
    
    save_or_show_plot(save_path, "Attention weights plot saved")
    
    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    cmap: str = "Blues",
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: Names of classes
        title: Plot title
        normalize: Whether to normalize by row (true labels)
        cmap: Colormap
        figsize: Figure size (width, height)
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    if figsize is None:
        # Adjust figsize based on number of classes
        n_classes = len(class_names)
        figsize = (max(10, n_classes * 1.5), max(8, n_classes * 1.5))
    
    # Normalize if requested
    if normalize:
        cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-6)
        fmt = '.2f'
        vmax = 1.0
    else:
        cm = confusion_matrix
        fmt = 'd'
        vmax = None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        vmax=vmax,
        ax=ax
    )
    
    # Set labels
    plt.title(title + (" (Normalized)" if normalize else ""))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add totals
    if not normalize:
        # Add row totals (true counts)
        row_totals = cm.sum(axis=1)
        for i, total in enumerate(row_totals):
            ax.text(len(class_names) + 0.5, i + 0.5, int(total),
                   va='center', ha='center', fontweight='bold')
        
        # Add column totals (predicted counts)
        col_totals = cm.sum(axis=0)
        for i, total in enumerate(col_totals):
            ax.text(i + 0.5, len(class_names) + 0.5, int(total),
                   va='center', ha='center', fontweight='bold')
        
        # Add grid lines
        ax.axhline(y=cm.shape[0], color='k', linewidth=2)
        ax.axvline(x=cm.shape[1], color='k', linewidth=2)
        
    # Calculate and display metrics
    if cm.shape[0] == cm.shape[1]:  # Square matrix
        diag = np.diag(cm)
        accuracy = diag.sum() / cm.sum()
        
        # Add accuracy text
        plt.figtext(
            0.5, 0.01, 
            f'Overall Accuracy: {accuracy:.2%}',
            ha='center', 
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}
        )
    
    plt.tight_layout()
    save_or_show_plot(save_path, "Confusion matrix plot saved")
    
    return fig


def plot_tsne(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    perplexity: int = 30,
    n_components: int = 2,
    random_state: int = 42,
    title: str = "t-SNE Visualization",
    point_size: int = 50,
    figsize: Tuple[int, int] = (10, 8),
    hover_text: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot t-SNE visualization.
    
    Args:
        X: Feature matrix
        y: Labels for coloring
        perplexity: Perplexity parameter for t-SNE
        n_components: Number of components
        random_state: Random state for reproducibility
        title: Plot title
        point_size: Size of scatter points
        figsize: Figure size (width, height)
        hover_text: Text to display when hovering over points (for interactive plots)
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    
    # Create figure
    if n_components == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Color points by label if provided
    if y is not None:
        # Get unique labels
        unique_labels = np.unique(y)
        
        # If few unique values, treat as categorical
        if len(unique_labels) <= 10:
            for i, label in enumerate(unique_labels):
                mask = y == label
                
                if n_components == 3:
                    ax.scatter(
                        X_tsne[mask, 0], X_tsne[mask, 1], X_tsne[mask, 2],
                        label=str(label),
                        s=point_size,
                        alpha=0.7
                    )
                else:
                    ax.scatter(
                        X_tsne[mask, 0], X_tsne[mask, 1],
                        label=str(label),
                        s=point_size,
                        alpha=0.7
                    )
            
            ax.legend()
        
        # Otherwise, use continuous colormap
        else:
            if n_components == 3:
                scatter = ax.scatter(
                    X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                    c=y, cmap='viridis',
                    s=point_size,
                    alpha=0.7
                )
            else:
                scatter = ax.scatter(
                    X_tsne[:, 0], X_tsne[:, 1],
                    c=y, cmap='viridis',
                    s=point_size,
                    alpha=0.7
                )
            
            plt.colorbar(scatter, label='Target Value')
    
    # No labels provided
    else:
        if n_components == 3:
            ax.scatter(
                X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],
                s=point_size,
                alpha=0.7
            )
        else:
            ax.scatter(
                X_tsne[:, 0], X_tsne[:, 1],
                s=point_size,
                alpha=0.7
            )
    
    # Add interactive hover text if provided and using MPL backend that supports it
    if hover_text is not None and len(hover_text) == X_tsne.shape[0]:
        try:
            from mplcursors import cursor
            
            # Add cursor
            cursor_obj = cursor(hover=True)
            
            @cursor_obj.connect("add")
            def on_add(sel):
                idx = sel.target.index
                if idx < len(hover_text):
                    sel.annotation.set_text(hover_text[idx])
        except ImportError:
            print("mplcursors not available for interactive hover. Install with 'pip install mplcursors'")
    
    # Set labels and title
    if n_components == 3:
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')
    else:
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
    
    ax.set_title(title)
    ax.grid(True)
    
    plt.tight_layout()
    save_or_show_plot(save_path, "t-SNE plot saved")
    
    return fig


def plot_physiological_signals(
    data_dict: Dict[str, np.ndarray],
    sampling_rates: Dict[str, float],
    events: Optional[List[Dict[str, Any]]] = None,
    annotations: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot physiological signals with events and annotations.
    
    Args:
        data_dict: Dictionary of signal arrays with signal names as keys
        sampling_rates: Dictionary of sampling rates for each signal
        events: List of event dictionaries with keys 'time', 'label', and optional 'color'
        annotations: Dictionary of signal-specific annotations with signal names as keys
                    and lists of annotation dictionaries with keys 'start', 'end', 'label', and optional 'color'
        figsize: Figure size (width, height)
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    # Get signal names
    signal_names = list(data_dict.keys())
    n_signals = len(signal_names)
    
    # Determine figure size
    if figsize is None:
        figsize = (14, 3 * n_signals)
    
    # Create figure with grid layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_signals, 1, height_ratios=[1] * n_signals)
    
    # Find the total duration
    max_duration = 0
    for signal_name, signal_data in data_dict.items():
        fs = sampling_rates[signal_name]
        duration = len(signal_data) / fs
        max_duration = max(max_duration, duration)
    
    # Plot each signal
    axes = []
    for i, signal_name in enumerate(signal_names):
        signal_data = data_dict[signal_name]
        fs = sampling_rates[signal_name]
        
        # Create time axis
        time = np.arange(len(signal_data)) / fs
        
        # Create axis for this signal
        ax = fig.add_subplot(gs[i])
        axes.append(ax)
        
        # Plot signal
        ax.plot(time, signal_data, label=signal_name)
        
        # Add events if provided
        if events is not None:
            for event in events:
                if event['time'] <= time[-1]:
                    color = event.get('color', 'red')
                    ax.axvline(event['time'], color=color, linestyle='--', alpha=0.7)
                    ax.text(
                        event['time'], 
                        ax.get_ylim()[1] * 0.95, 
                        event['label'], 
                        rotation=90, 
                        verticalalignment='top',
                        color=color
                    )
        
        # Add signal-specific annotations if provided
        if annotations is not None and signal_name in annotations:
            for annotation in annotations[signal_name]:
                if annotation['start'] <= time[-1]:
                    end = min(annotation['end'], time[-1])
                    color = annotation.get('color', 'yellow')
                    alpha = annotation.get('alpha', 0.3)
                    ax.axvspan(annotation['start'], end, color=color, alpha=alpha)
                    ax.text(
                        (annotation['start'] + end) / 2,
                        ax.get_ylim()[1] * 0.9,
                        annotation['label'],
                        horizontalalignment='center',
                        verticalalignment='top',
                        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2}
                    )
        
        # Add signal statistics
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        
        stats_text = f"Mean: {mean:.2f}, Std: {std:.2f}"
        if np.isfinite(signal_data).all() and len(signal_data) > 0:
            min_val = np.min(signal_data)
            max_val = np.max(signal_data)
            stats_text += f", Min: {min_val:.2f}, Max: {max_val:.2f}"
        
        ax.text(
            0.98, 0.05, 
            stats_text, 
            transform=ax.transAxes,
            horizontalalignment='right',
            bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 2}
        )
        
        # Set labels
        ax.set_ylabel(signal_name)
        ax.legend(loc='upper left')
        ax.grid(True)
        
        # If not the last subplot, hide x-label
        if i < n_signals - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time (s)')
        
        # Set x-limits to match across subplots
        ax.set_xlim(0, max_duration)
    
    # Link x-axes
    for ax in axes[1:]:
        axes[0].get_shared_x_axes().join(axes[0], ax)
    
    plt.tight_layout()
    save_or_show_plot(save_path, "Physiological signals plot saved")
    
    return fig


def plot_signal_spectrogram(
    signal: np.ndarray,
    fs: float,
    window_size: int = 256,
    overlap: int = 128,
    title: str = "Signal Spectrogram",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot spectrogram of a signal.
    
    Args:
        signal: Signal array
        fs: Sampling rate in Hz
        window_size: FFT window size
        overlap: Overlap between windows
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(
        signal, 
        fs=fs, 
        window=('tukey', 0.25), 
        nperseg=window_size, 
        noverlap=overlap, 
        scaling='spectrum'
    )
    
    # Plot spectrogram
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power/Frequency (dB/Hz)')
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    
    # Set y-axis to log scale for better visualization of lower frequencies
    ax.set_yscale('log')
    
    # Set y-axis limit to Nyquist frequency
    ax.set_ylim(0.5, fs/2)
    
    plt.tight_layout()
    save_or_show_plot(save_path, "Spectrogram plot saved")
    
    return fig


"""
Implemented improvements:
1. Added interactive visualizations with hover capabilities
2. Implemented 3D visualization options for PCA and t-SNE
3. Added support for creating custom colormaps
4. Enhanced attention weight visualization with highlighting
5. Improved confusion matrix visualization with totals and metrics
6. Implemented specialized physiological signal visualization
7. Added spectrogram visualization for frequency analysis
8. Enhanced time series plotting with events and annotations
9. Implemented animation capabilities for time series data
10. Added advanced statistical visualizations on plots
11. Improved customization options for all plot types
12. Enhanced learning curve visualization with min/max annotations
13. Implemented clustered feature correlation visualization
14. Added feature vector visualization in PCA biplot
15. Enhanced all visualizations with better documentation and return values
"""
