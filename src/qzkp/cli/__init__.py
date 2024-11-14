"""Command-line interface for QZKP."""

import argparse
import asyncio
import numpy as np
from ..optimized_qzkp import QuantumZKP

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="QZKP Command Line Interface")
    parser.add_argument("--dimensions", type=int, default=8,
                      help="Number of dimensions for the quantum system")
    parser.add_argument("--security-level", type=int, default=128,
                      help="Security level for the proof system")

    args = parser.parse_args()

    # Create QZKP instance
    qzkp = QuantumZKP(dimensions=args.dimensions, security_level=args.security_level)

    # Example usage
    vector = np.random.random(args.dimensions)
    vector = vector / np.linalg.norm(vector)

    async def run_example():
        commitment, proof = await qzkp.prove_vector_knowledge(vector, "test")
        is_valid = qzkp.verify_proof(commitment, proof, "test")
        print(f"Proof verification: {'Success' if is_valid else 'Failed'}")

    asyncio.run(run_example())

if __name__ == "__main__":
    main()
