# Quantum Zero-Knowledge Proof Implementation (QZKP)

A high-performance implementation of quantum-inspired zero-knowledge proofs using Numba acceleration and parallel processing.

## Features

- Numba-optimized quantum computations
- Parallel processing for vector operations
- Thread-safe implementation
- Batch processing capabilities
- Memory-efficient operations
- Post-quantum cryptography with Falcon
- Comprehensive test suite

## Installation

```bash
# Clone the repository
git clone https://github.com/theaxiomverse/qzkp
cd qzkp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

```python
import numpy as np
from qzkp import QuantumZKP

# Initialize QZKP system
qzkp = QuantumZKP(dimensions=8, security_level=128)

# Create a test vector
vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
vector = vector / np.linalg.norm(vector)  # Normalize

# Generate proof
identifier = "test_vector_1"
commitment, proof = await qzkp.prove_vector_knowledge(vector, identifier)

# Verify proof
is_valid = qzkp.verify_proof(commitment, proof, identifier)
print(f"Proof verification: {'Success' if is_valid else 'Failed'}")
```

## Batch Processing

```python
# Generate multiple proofs in batch
vectors = [np.random.random(8) for _ in range(100)]
vectors = [v / np.linalg.norm(v) for v in vectors]
identifiers = [f"vector_{i}" for i in range(100)]

# Batch proof generation
commitments_proofs = await qzkp.prove_vector_knowledge_batch(
    vectors,
    identifiers,
    batch_size=10
)

# Batch verification
verification_data = [
    (commitment, proof, identifier)
    for (commitment, proof), identifier in zip(commitments_proofs, identifiers)
]

results = qzkp.verify_proof_batch(verification_data, batch_size=10)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_qzkp.py

# Run with coverage report
pytest --cov=qzkp tests/

# Run benchmarks
pytest tests/test_qzkp.py -v --benchmark-only
```

## Performance Optimization

The implementation includes several optimizations:

1. **Numba Acceleration**:
   - JIT compilation for numerical operations
   - Parallel processing for vector operations
   - SIMD vectorization where possible

2. **Memory Optimization**:
   - Efficient memory layout
   - Cache management
   - Minimal allocations

3. **Thread Safety**:
   - Thread-safe caching
   - Proper resource management
   - Batch processing capabilities

## Documentation

Detailed documentation is available in the `docs/` directory. To build the documentation:

```bash
cd docs
make html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

