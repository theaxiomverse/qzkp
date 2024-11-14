import json
import sys

import numpy as np
import hashlib
import logging
import time
import os
import psutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from numba import cuda
from oqs import Signature
from collections import deque
import threading
import asyncio
import warnings

from .numba_utils import calculate_entropy_numba, generate_measurement_data_numba, \
    verify_measurement_data_numba, verify_coefficients_numba

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Constants
MAX_VECTOR_SIZE = 1024
ENTROPY_EPSILON = 1e-10
PROBABILITY_TOLERANCE = 1e-5
DEFAULT_BATCH_SIZE = 1000
MAX_CACHE_SIZE = 10000
THREAD_COUNT = min(32, (os.cpu_count() or 1) * 4)

# GPU Support Check
USE_GPU = cuda.is_available()
if USE_GPU:
    logger.info("CUDA GPU support enabled")


def set_cpu_affinity():
    """Set CPU affinity with platform-specific handling."""
    try:
        import psutil
        p = psutil.Process()

        if sys.platform == "linux":
            # Linux-specific CPU affinity
            p.cpu_affinity(list(range(psutil.cpu_count())))
        elif sys.platform == "win32":
            # Windows-specific CPU affinity
            p.cpu_affinity(list(range(psutil.cpu_count())))
        elif sys.platform == "darwin":
            # macOS doesn't support CPU affinity, use thread priority instead
            os.nice(0)  # Set normal priority

        logger.info(f"Process configured to use {psutil.cpu_count()} CPUs")
    except Exception as e:
        logger.warning(f"Could not set process affinity: {e}")
        logger.warning("Continuing with default CPU configuration")


class ResultCache:
    """Thread-safe cache for computation results."""

    def __init__(self, maxsize: int = MAX_CACHE_SIZE):
        self.cache = {}
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.access_times = {}
        self.access_queue = deque()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.access_queue.append(key)
                return self.cache[key]
        return None

    def put(self, key: str, value: Any):
        with self.lock:
            if len(self.cache) >= self.maxsize:
                self._evict_oldest()
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_queue.append(key)

    def _evict_oldest(self):
        while self.access_queue and len(self.cache) >= self.maxsize:
            oldest_key = self.access_queue.popleft()
            if oldest_key in self.cache:
                del self.cache[oldest_key]
                del self.access_times[oldest_key]


class QuantumStateVector:
    """Represents a quantum state vector with optimized memory usage."""

    __slots__ = ['coordinates', '_entanglement', '_coherence', '_state_type',
                 '_timestamp', '_cache']

    def __init__(self, coordinates: np.ndarray):
        """Initialize the quantum state vector."""
        self.coordinates = coordinates
        self._entanglement = 0.0
        self._coherence = None  # Will be calculated on demand
        self._state_type = "SUPERPOSITION"
        self._timestamp = time.time()
        self._cache = {}

        # Validate vector size
        if len(coordinates) > MAX_VECTOR_SIZE:
            warnings.warn(f"Vector size exceeds MAX_VECTOR_SIZE ({MAX_VECTOR_SIZE})")

    @property
    def entanglement(self) -> float:
        """Get the entanglement value."""
        return self._entanglement

    @entanglement.setter
    def entanglement(self, value: float):
        """Set the entanglement value."""
        self._entanglement = value

    @property
    def coherence(self) -> float:
        """Get the coherence value with caching."""
        if self._coherence is None:
            self._coherence = self.calculate_coherence()
        return self._coherence

    @coherence.setter
    def coherence(self, value: float):
        """Set the coherence value."""
        self._coherence = value

    @property
    def state_type(self) -> str:
        """Get the state type."""
        return self._state_type

    @state_type.setter
    def state_type(self, value: str):
        """Set the state type."""
        self._state_type = value

    @property
    def timestamp(self) -> float:
        """Get the timestamp."""
        return self._timestamp

    def calculate_coherence(self) -> float:
        """Calculate coherence value."""
        if 'coherence' not in self._cache:
            self._cache['coherence'] = float(np.mean(np.abs(self.coordinates)))
        return self._cache['coherence']

    def serialize(self) -> bytes:
        """Serialize the state vector to bytes."""
        try:
            if 'serialized' not in self._cache:
                data = {
                    "coordinates": self.coordinates.tolist(),
                    "entanglement": float(self.entanglement),
                    "coherence": float(self.coherence),
                    "state_type": str(self.state_type),
                    "timestamp": float(self.timestamp)
                }
                self._cache['serialized'] = json.dumps(
                    data,
                    separators=(',', ':'),
                    default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else None
                )
            return self._cache['serialized'].encode('utf-8')
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise ValueError(f"Failed to serialize state vector: {e}")

    def clear_cache(self):
        """Clear all cached values."""
        self._cache.clear()
        self._coherence = None

    def __eq__(self, other) -> bool:
        """Implement equality comparison."""
        if not isinstance(other, QuantumStateVector):
            return False
        return np.array_equal(self.coordinates, other.coordinates)

    def __hash__(self) -> int:
        """Implement hashing for the state vector."""
        return hash(self.coordinates.tobytes())


class QuantumZKP:
    def __init__(self, dimensions: int = 8, security_level: int = 128):
        set_cpu_affinity()

        self.dimensions = min(dimensions, MAX_VECTOR_SIZE)
        self.security_level = security_level
        self._falcon = Signature("Falcon-512")
        self.public_key = self._falcon.generate_keypair()
        self._result_cache = ResultCache()

        logger.info(f"Initialized QuantumZKP with dimensions={dimensions}, security_level={security_level}")

    def _generate_commitment(self, state: QuantumStateVector, identifier: str) -> bytes:
        """Generate a commitment using SHA3-256."""
        hasher = hashlib.sha3_256()
        hasher.update(state.coordinates.tobytes())
        hasher.update(str(state.coherence).encode())
        hasher.update(identifier.encode())
        return hasher.digest()

    def _calculate_entropy(self, state_coordinates: np.ndarray) -> float:
        """Calculate entropy using Numba-optimized function."""
        return float(calculate_entropy_numba(state_coordinates))

    def _generate_measurements(self, state_coordinates: np.ndarray, security_level: int) -> List[Dict]:
        """Generate measurements using Numba-optimized arrays."""
        num_measurements = security_level // 8

        # Generate data using Numba
        indices, probabilities, phases = generate_measurement_data_numba(
            state_coordinates,
            num_measurements
        )

        # Convert to list of dictionaries (outside of Numba)
        return [
            {
                'basis_index': int(idx),
                'probability': float(prob),
                'phase': float(phase)
            }
            for idx, prob, phase in zip(indices, probabilities, phases)
        ]

    def _verify_measurements(self, measurements: List[Dict], state_size: int) -> bool:
        """Verify measurements using Numba-optimized functions."""
        # Convert to arrays for Numba processing
        indices = np.array([m['basis_index'] for m in measurements], dtype=np.int64)
        probabilities = np.array([m['probability'] for m in measurements], dtype=np.float64)
        phases = np.array([m['phase'] for m in measurements], dtype=np.float64)

        return verify_measurement_data_numba(indices, probabilities, phases, state_size)

    def _verify_coefficients(self, coefficients: np.ndarray) -> bool:
        """Verify coefficients using Numba-optimized function."""
        return verify_coefficients_numba(coefficients)

    def _prepare_message_for_signing(self, proof: Dict, commitment: bytes) -> bytes:
        """Prepare message for signing."""
        proof_copy = {k: v for k, v in proof.items() if k != "signature"}
        message = json.dumps(
            proof_copy,
            sort_keys=True,
            default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else None,
            separators=(',', ':')
        ).encode('utf-8')
        return message + commitment

    def _validate_proof_structure(self, proof: Dict) -> bool:
        """Validate proof structure."""
        required_fields = {
            'quantum_dimensions',
            'basis_coefficients',
            'measurements',
            'state_metadata',
            'signature',
            'identifier'
        }
        return all(field in proof for field in required_fields)

    async def prove_vector_knowledge(self, vector: np.ndarray, identifier: str) -> Tuple[bytes, Dict]:
        """Generate proof with Numba-optimized operations."""
        try:
            # Normalize vector
            vector = vector / np.linalg.norm(vector)

            # Create quantum state
            state = QuantumStateVector(vector)

            # Calculate entropy and commitment
            entropy = self._calculate_entropy(state.coordinates)
            commitment = self._generate_commitment(state, identifier)

            # Generate measurements
            measurements = self._generate_measurements(
                state.coordinates,
                self.security_level
            )

            # Create proof
            proof = {
                'quantum_dimensions': self.dimensions,
                'basis_coefficients': state.coordinates.tolist(),
                'measurements': measurements,
                'state_metadata': {
                    'coherence': state.coherence,
                    'entanglement': entropy,
                    'timestamp': time.time()
                },
                'identifier': identifier
            }

            # Sign proof
            message = self._prepare_message_for_signing(proof, commitment)
            signature = self._falcon.sign(message)
            proof['signature'] = signature.hex()

            return commitment, proof

        except Exception as e:
            logger.error(f"Error in prove_vector_knowledge: {e}")
            raise

    def verify_proof(self, commitment: bytes, proof: Dict, identifier: str) -> bool:
        """Verify proof using Numba-optimized operations."""
        try:
            # Cache check
            cache_key = f"{commitment.hex()}-{identifier}"
            cached_result = self._result_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Verify structure
            if not self._validate_proof_structure(proof):
                return False

            # Verify signature
            signature = bytes.fromhex(proof['signature'])
            message = self._prepare_message_for_signing(proof, commitment)
            if not self._falcon.verify(message, signature, self.public_key):
                return False

            # Verify measurements
            if not self._verify_measurements(proof['measurements'], len(proof['basis_coefficients'])):
                return False

            # Verify coefficients
            coefficients = np.array(proof['basis_coefficients'], dtype=np.complex128)
            if not self._verify_coefficients(coefficients):
                return False

            # Cache result
            self._result_cache.put(cache_key, True)
            return True

        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            return False

    async def prove_vector_knowledge_batch(
            self,
            vectors: List[np.ndarray],
            identifiers: List[str],
            batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[Tuple[bytes, Dict]]:
        """Generate proofs in batches."""
        results = []

        async def process_batch(batch_vectors, batch_ids):
            tasks = []
            for vector, id_ in zip(batch_vectors, batch_ids):
                tasks.append(self.prove_vector_knowledge(vector, id_))
            return await asyncio.gather(*tasks)

        # Process in batches
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_ids = identifiers[i:i + batch_size]
            batch_results = await process_batch(batch_vectors, batch_ids)
            results.extend(batch_results)

        return results

    def verify_proof_batch(
            self,
            commitments_proofs_ids: List[Tuple[bytes, Dict, str]],
            batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[bool]:
        """Verify proofs in batches."""
        results = []

        # Process in batches
        for i in range(0, len(commitments_proofs_ids), batch_size):
            batch = commitments_proofs_ids[i:i + batch_size]
            batch_results = [
                self.verify_proof(commitment, proof, identifier)
                for commitment, proof, identifier in batch
            ]
            results.extend(batch_results)

        return results