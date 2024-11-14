import pytest
import numpy as np
from ..src.qzkp import QuantumZKP, QuantumStateVector
import asyncio
from typing import List, Tuple
import logging
import concurrent.futures
import queue

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def qzkp_instance():
    """Create a QZKP instance for testing."""
    return QuantumZKP(dimensions=8, security_level=128)


@pytest.fixture
def test_vector():
    """Generate a test vector."""
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    vector = rng.random(8)
    return vector / np.linalg.norm(vector)  # Normalize vector


@pytest.fixture
def test_vectors():
    """Generate multiple test vectors."""
    rng = np.random.default_rng(42)
    vectors = [rng.random(8) for _ in range(10)]
    return [v / np.linalg.norm(v) for v in vectors]


@pytest.mark.asyncio
async def test_proof_generation(qzkp_instance, test_vector):
    """Test single proof generation."""
    identifier = "test_vector_1"
    commitment, proof = await qzkp_instance.prove_vector_knowledge(test_vector, identifier)

    assert commitment is not None
    assert isinstance(commitment, bytes)
    assert proof is not None
    assert isinstance(proof, dict)
    assert 'quantum_dimensions' in proof
    assert 'basis_coefficients' in proof
    assert 'measurements' in proof
    assert 'signature' in proof

    # Validate measurements structure
    assert isinstance(proof['measurements'], list)
    for measurement in proof['measurements']:
        assert 'basis_index' in measurement
        assert 'probability' in measurement
        assert 'phase' in measurement
        assert 0 <= measurement['probability'] <= 1


@pytest.mark.asyncio
async def test_proof_verification(qzkp_instance, test_vector):
    """Test proof verification."""
    identifier = "test_vector_1"
    commitment, proof = await qzkp_instance.prove_vector_knowledge(test_vector, identifier)

    # Verify the proof
    is_valid = qzkp_instance.verify_proof(commitment, proof, identifier)
    assert is_valid is True


@pytest.mark.asyncio
async def test_batch_proof_generation(qzkp_instance, test_vectors):
    """Test batch proof generation."""
    identifiers = [f"test_vector_{i}" for i in range(len(test_vectors))]
    commitments_proofs = await qzkp_instance.prove_vector_knowledge_batch(
        test_vectors,
        identifiers,
        batch_size=5
    )

    assert len(commitments_proofs) == len(test_vectors)
    for commitment, proof in commitments_proofs:
        assert isinstance(commitment, bytes)
        assert isinstance(proof, dict)
        assert all(key in proof for key in ['quantum_dimensions', 'basis_coefficients', 'measurements'])


@pytest.mark.asyncio
async def test_batch_proof_verification(qzkp_instance, test_vectors):
    """Test batch proof verification."""
    identifiers = [f"test_vector_{i}" for i in range(len(test_vectors))]
    commitments_proofs = await qzkp_instance.prove_vector_knowledge_batch(
        test_vectors,
        identifiers,
        batch_size=5
    )

    # Prepare verification data
    verification_data = [
        (commitment, proof, identifier)
        for (commitment, proof), identifier in zip(commitments_proofs, identifiers)
    ]

    # Verify proofs
    verification_results = qzkp_instance.verify_proof_batch(
        verification_data,
        batch_size=5
    )

    assert len(verification_results) == len(test_vectors)
    assert all(verification_results)


@pytest.mark.asyncio
async def test_invalid_proof_rejection(qzkp_instance, test_vector):
    """Test that invalid proofs are rejected."""
    identifier = "test_vector_1"
    commitment, proof = await qzkp_instance.prove_vector_knowledge(test_vector, identifier)

    # Tamper with the proof
    proof['basis_coefficients'][0] += 1.0

    # Verify tampered proof
    is_valid = qzkp_instance.verify_proof(commitment, proof, identifier)
    assert is_valid is False


@pytest.mark.asyncio
async def test_quantum_state_vector(test_vector):
    """Test QuantumStateVector class."""
    state = QuantumStateVector(test_vector)

    # Test coherence calculation
    coherence = state.coherence
    assert 0 <= coherence <= 1

    # Test serialization
    serialized = state.serialize()
    assert isinstance(serialized, bytes)

    # Test caching
    assert state._cache is not None
    state.clear_cache()
    assert len(state._cache) == 0


@pytest.mark.asyncio
async def test_large_vector_handling():
    """Test handling of vectors larger than MAX_VECTOR_SIZE."""
    large_vector = np.random.random(1025)  # Exceeds MAX_VECTOR_SIZE
    large_vector = large_vector / np.linalg.norm(large_vector)

    with pytest.warns(UserWarning):
        state = QuantumStateVector(large_vector)
        assert len(state.coordinates) == 1025

'''
def test_thread_safety(qzkp_instance, test_vectors):
    """Test thread safety of proof generation and verification."""
    results = queue.Queue()

    def worker(vector, identifier):
        try:
            commitment, proof = asyncio.run(
                qzkp_instance.prove_vector_knowledge(vector, identifier)
            )
            is_valid = qzkp_instance.verify_proof(commitment, proof, identifier)
            results.put(is_valid)
        except Exception as e:
            logger.error(f"Thread error: {e}")
            results.put(False)

    # Use ThreadPoolExecutor for better thread management
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        futures = [
            executor.submit(worker, vector, f"thread_test_{i}")
            for i, vector in enumerate(test_vectors)
        ]
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    # Collect results
    result_list = []
    while not results.empty():
        result_list.append(results.get())

    assert len(result_list) == len(test_vectors)
    assert all(result_list)
'''

@pytest.mark.benchmark
def test_performance_prove_verify(benchmark, qzkp_instance, test_vector):
    """Benchmark proof generation and verification."""
    identifier = "test_vector_1"

    def run_proof_cycle():
        commitment, proof = asyncio.run(
            qzkp_instance.prove_vector_knowledge(test_vector, identifier)
        )
        return qzkp_instance.verify_proof(commitment, proof, identifier)

    result = benchmark(run_proof_cycle)
    assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])