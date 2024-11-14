import asyncio
import time
import numpy as np
import psutil
import os
import logging
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from memory_profiler import profile
from tqdm import tqdm

# Import the optimized QZKP implementation
from src.qzkp.optimized_qzkp import QuantumZKP, THREAD_COUNT, USE_GPU

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QZKPBenchmark:
    def __init__(self):
        self.zk_proof_system = QuantumZKP()
        self.results = {}

    def _generate_test_vectors(self, num_vectors: int, vector_size: int) -> Tuple[List[np.ndarray], List[str]]:
        """Generate test vectors and identifiers."""
        rng = np.random.default_rng(time.time_ns())
        vectors = [rng.random(vector_size) for _ in range(num_vectors)]
        identifiers = [f"identifier_{i}" for i in range(num_vectors)]
        return vectors, identifiers

    async def _warmup_run(self):
        """Perform warmup run to initialize JIT compilation."""
        logger.info("Performing warmup run...")
        vectors, identifiers = self._generate_test_vectors(10, 8)
        await self.zk_proof_system.prove_vector_knowledge_batch(vectors, identifiers, batch_size=10)

    @profile
    async def benchmark_proof_generation(self, num_vectors: int, vector_size: int, batch_size: int) -> Dict:
        """Benchmark proof generation with memory profiling."""
        vectors, identifiers = self._generate_test_vectors(num_vectors, vector_size)

        # Memory usage before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        # Run proof generation with progress bar
        with tqdm(total=num_vectors, desc="Generating proofs") as pbar:
            commitments_proofs = []
            for i in range(0, num_vectors, batch_size):
                batch_vectors = vectors[i:min(i + batch_size, num_vectors)]
                batch_ids = identifiers[i:min(i + batch_size, num_vectors)]
                batch_results = await self.zk_proof_system.prove_vector_knowledge_batch(
                    batch_vectors,
                    batch_ids,
                    batch_size=batch_size
                )
                commitments_proofs.extend(batch_results)
                pbar.update(len(batch_vectors))

        proving_time = time.time() - start_time

        # Memory usage after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        return {
            "operation": "proof_generation",
            "num_vectors": num_vectors,
            "vector_size": vector_size,
            "batch_size": batch_size,
            "total_time": proving_time,
            "throughput": num_vectors / proving_time,
            "memory_before": mem_before,
            "memory_after": mem_after,
            "memory_delta": mem_after - mem_before,
            "gpu_used": USE_GPU,
            "thread_count": THREAD_COUNT
        }

    @profile
    async def benchmark_verification(self, commitments_proofs: List[Tuple], identifiers: List[str],
                                     batch_size: int) -> Dict:
        """Benchmark proof verification with memory profiling."""
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        start_time = time.time()

        # Prepare verification data
        commitments_proofs_ids = [
            (commitment, proof, identifier)
            for (commitment, proof), identifier in zip(commitments_proofs, identifiers)
        ]

        # Run verification with progress bar
        num_verifications = len(commitments_proofs_ids)
        with tqdm(total=num_verifications, desc="Verifying proofs") as pbar:
            verification_results = []
            for i in range(0, num_verifications, batch_size):
                batch = commitments_proofs_ids[i:min(i + batch_size, num_verifications)]
                batch_results = self.zk_proof_system.verify_proof_batch(batch, batch_size=batch_size)
                verification_results.extend(batch_results)
                pbar.update(len(batch))

        verification_time = time.time() - start_time

        mem_after = process.memory_info().rss / 1024 / 1024

        return {
            "operation": "verification",
            "num_vectors": len(commitments_proofs_ids),
            "batch_size": batch_size,
            "total_time": verification_time,
            "throughput": len(commitments_proofs_ids) / verification_time,
            "memory_before": mem_before,
            "memory_after": mem_after,
            "memory_delta": mem_after - mem_before,
            "success_rate": sum(verification_results) / len(verification_results),
            "gpu_used": USE_GPU,
            "thread_count": THREAD_COUNT
        }

    def plot_results(self, results_list: List[Dict]):
        """Generate performance visualization plots."""
        df = pd.DataFrame(results_list)

        # Create throughput comparison plot
        plt.figure(figsize=(12, 6))
        operations = df['operation'].unique()

        for op in operations:
            op_data = df[df['operation'] == op]
            plt.plot(op_data['batch_size'], op_data['throughput'],
                     marker='o', label=f"{op} throughput")

        plt.xlabel('Batch Size')
        plt.ylabel('Operations per Second')
        plt.title('QZKP Performance Analysis')
        plt.legend()
        plt.grid(True)
        plt.savefig('qzkp_performance.png')
        plt.close()

        # Create memory usage plot
        plt.figure(figsize=(12, 6))
        for op in operations:
            op_data = df[df['operation'] == op]
            plt.plot(op_data['batch_size'], op_data['memory_delta'],
                     marker='o', label=f"{op} memory usage")

        plt.xlabel('Batch Size')
        plt.ylabel('Memory Usage (MB)')
        plt.title('QZKP Memory Analysis')
        plt.legend()
        plt.grid(True)
        plt.savefig('qzkp_memory.png')
        plt.close()


async def run_full_benchmark():
    """Run comprehensive benchmark suite."""
    benchmark = QZKPBenchmark()

    # Perform warmup
    await benchmark._warmup_run()

    # Test parameters
    vector_sizes = [8,16]
    batch_sizes = [500, 1500, 2000]
    num_vectors = 100000

    results = []

    logger.info("Starting comprehensive benchmark...")

    for vector_size in vector_sizes:
        for batch_size in batch_sizes:
            # Benchmark proof generation
            proof_results = await benchmark.benchmark_proof_generation(
                num_vectors,
                vector_size,
                batch_size
            )
            results.append(proof_results)

            # Generate test vectors for verification
            vectors, identifiers = benchmark._generate_test_vectors(num_vectors, vector_size)
            commitments_proofs = await benchmark.zk_proof_system.prove_vector_knowledge_batch(
                vectors,
                identifiers,
                batch_size=batch_size
            )

            # Benchmark verification
            verify_results = await benchmark.benchmark_verification(
                commitments_proofs,
                identifiers,
                batch_size
            )
            results.append(verify_results)

            # Log progress
            logger.info(f"Completed benchmark for vector_size={vector_size}, batch_size={batch_size}")

            # Clear memory between runs
            import gc
            gc.collect()

    # Generate visualization
    benchmark.plot_results(results)

    # Save results to CSV
    pd.DataFrame(results).to_csv('benchmark_results.csv', index=False)

    logger.info("Benchmark completed. Results saved to 'benchmark_results.csv'")
    logger.info("Performance plots saved as 'qzkp_performance.png' and 'qzkp_memory.png'")


if __name__ == "__main__":
    # Set process priority
    try:
        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -20)
    except Exception as e:
        logger.warning(f"Could not set process priority: {e}")

    # Run benchmark
    asyncio.run(run_full_benchmark())