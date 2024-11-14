import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def analyze_benchmark_results(csv_file: str = 'benchmark_results.csv') -> Dict:
    """Analyze benchmark results and generate comprehensive report."""
    # Load data
    df = pd.read_csv(csv_file)

    # Set up plotting style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")

    # Calculate key statistics with explicit type conversion
    stats = {
        'proof_generation': {
            'mean_throughput': float(df[df['operation'] == 'proof_generation']['throughput'].mean()),
            'max_throughput': float(df[df['operation'] == 'proof_generation']['throughput'].max()),
            'optimal_batch_size': int(df[df['operation'] == 'proof_generation'].loc[
                                          df[df['operation'] == 'proof_generation']['throughput'].idxmax()
                                      ]['batch_size']),
            'mean_memory_usage': float(df[df['operation'] == 'proof_generation']['memory_delta'].mean())
        },
        'verification': {
            'mean_throughput': float(df[df['operation'] == 'verification']['throughput'].mean()),
            'max_throughput': float(df[df['operation'] == 'verification']['throughput'].max()),
            'optimal_batch_size': int(df[df['operation'] == 'verification'].loc[
                                          df[df['operation'] == 'verification']['throughput'].idxmax()
                                      ]['batch_size']),
            'mean_memory_usage': float(df[df['operation'] == 'verification']['memory_delta'].mean()),
            'success_rate': float(df[df['operation'] == 'verification']['success_rate'].mean())
        }
    }

    # Generate detailed plots
    # Throughput vs Batch Size plot
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=df, x='batch_size', y='throughput', hue='operation',
                 style='vector_size', markers=True, dashes=False)

    plt.xlabel('Batch Size')
    plt.ylabel('Operations per Second')
    plt.title('QZKP Performance Analysis - Throughput vs Batch Size')
    plt.legend(title='Operation Type')
    plt.tight_layout()
    plt.savefig('qzkp_detailed_performance.png')
    plt.close()

    # Memory usage heatmap
    plt.figure(figsize=(12, 8))
    pivot_memory = df.pivot_table(
        values='memory_delta',
        index='vector_size',
        columns='batch_size',
        aggfunc='mean'
    )
    sns.heatmap(pivot_memory, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Memory Usage (MB) by Vector Size and Batch Size')
    plt.tight_layout()
    plt.savefig('qzkp_memory_heatmap.png')
    plt.close()

    # Memory usage over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='batch_size', y='memory_delta', hue='operation')
    plt.xlabel('Batch Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Batch Size')
    plt.tight_layout()
    plt.savefig('qzkp_memory_usage.png')
    plt.close()

    # Success rate analysis (for verification)
    if 'success_rate' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[df['operation'] == 'verification'],
                    x='batch_size', y='success_rate')
        plt.xlabel('Batch Size')
        plt.ylabel('Success Rate')
        plt.title('Verification Success Rate by Batch Size')
        plt.tight_layout()
        plt.savefig('qzkp_success_rate.png')
        plt.close()

    # Generate recommendations with explicit type conversion
    recommendations = {
        'optimal_configuration': {
            'batch_size': int(stats['proof_generation']['optimal_batch_size']),
            'thread_count': int(df['thread_count'].iloc[0]),
            'gpu_recommendation': 'Use GPU' if df['gpu_used'].any() else 'CPU-only mode sufficient'
        },
        'performance_tips': [
            f"Optimal batch size for proof generation: {int(stats['proof_generation']['optimal_batch_size'])}",
            f"Optimal batch size for verification: {int(stats['verification']['optimal_batch_size'])}",
            f"Expected throughput: {float(stats['proof_generation']['max_throughput']):.2f} proofs/sec",
            f"Expected memory usage: {float(stats['proof_generation']['mean_memory_usage']):.2f} MB",
            "GPU acceleration recommended" if df['gpu_used'].any() else "CPU-only mode sufficient"
        ]
    }

    # Save analysis results using the custom encoder
    with open('benchmark_analysis.json', 'w') as f:
        json.dump({**stats, **recommendations}, f, indent=4, cls=NumpyEncoder)

    return {**stats, **recommendations}


if __name__ == "__main__":
    try:
        # Analyze results
        results = analyze_benchmark_results()

        # Print summary
        print("\nBenchmark Analysis Summary:")
        print("===========================")
        print(f"\nProof Generation:")
        print(f"  Max Throughput: {results['proof_generation']['max_throughput']:.2f} proofs/sec")
        print(f"  Optimal Batch Size: {results['proof_generation']['optimal_batch_size']}")
        print(f"  Mean Memory Usage: {results['proof_generation']['mean_memory_usage']:.2f} MB")

        print(f"\nVerification:")
        print(f"  Max Throughput: {results['verification']['max_throughput']:.2f} verifications/sec")
        print(f"  Optimal Batch Size: {results['verification']['optimal_batch_size']}")
        print(f"  Mean Memory Usage: {results['verification']['mean_memory_usage']:.2f} MB")
        print(f"  Success Rate: {results['verification']['success_rate']:.2%}")

        print("\nRecommended Configuration:")
        print("=========================")
        for tip in results['performance_tips']:
            print(f"- {tip}")

        print("\nGenerated Visualizations:")
        print("- qzkp_detailed_performance.png")
        print("- qzkp_memory_heatmap.png")
        print("- qzkp_memory_usage.png")
        print("- qzkp_success_rate.png")
        print("- benchmark_analysis.json")

    except FileNotFoundError:
        print("Error: benchmark_results.csv not found. Please run the benchmark first.")
    except Exception as e:
        print(f"Error analyzing benchmark results: {e}")
        raise  # Re-raise the exception for debugging