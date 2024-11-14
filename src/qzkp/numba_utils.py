import numpy as np
from numba import njit, prange
import math

# Constants
EPSILON = 1e-10
PROBABILITY_TOLERANCE = 1e-5


@njit(fastmath=True, parallel=True)
def calculate_entropy_numba(amplitudes: np.ndarray) -> float:
    """Calculate entropy using Numba-optimized parallel processing."""
    probabilities = np.zeros(len(amplitudes), dtype=np.float64)

    # Calculate probabilities in parallel
    for i in prange(len(amplitudes)):
        probabilities[i] = abs(amplitudes[i]) ** 2

    # Calculate entropy
    entropy = 0.0
    for i in prange(len(probabilities)):
        if probabilities[i] > EPSILON:
            entropy -= probabilities[i] * math.log2(probabilities[i] + EPSILON)

    return entropy


@njit(fastmath=True, parallel=True)
def generate_measurement_data_numba(state_coordinates: np.ndarray, num_measurements: int):
    """Generate measurement data using parallel processing."""
    indices = np.zeros(num_measurements, dtype=np.int64)
    probabilities = np.zeros(num_measurements, dtype=np.float64)
    phases = np.zeros(num_measurements, dtype=np.float64)

    for i in prange(num_measurements):
        # Generate random index
        idx = np.random.randint(0, len(state_coordinates))
        indices[i] = idx

        # Calculate probability and phase
        amplitude = state_coordinates[idx]
        probabilities[i] = abs(amplitude) ** 2
        phases[i] = math.atan2(amplitude.imag, amplitude.real)

    return indices, probabilities, phases


@njit(fastmath=True)
def verify_coefficients_numba(coefficients: np.ndarray) -> bool:
    """Verify coefficients using vectorized operations."""
    total = 0.0
    for i in range(len(coefficients)):
        total += abs(coefficients[i]) ** 2

    return abs(total - 1.0) < PROBABILITY_TOLERANCE


@njit(fastmath=True)
def verify_measurement_data_numba(indices: np.ndarray, probabilities: np.ndarray,
                                  phases: np.ndarray, state_size: int) -> bool:
    """Verify measurement data using vectorized operations."""
    # Check array lengths match
    if not (len(indices) == len(probabilities) == len(phases)):
        return False

    # Check index bounds
    for i in range(len(indices)):
        if indices[i] < 0 or indices[i] >= state_size:
            return False

    # Check probability ranges
    for i in range(len(probabilities)):
        if probabilities[i] < 0.0 or probabilities[i] > 1.0:
            return False

    # Check phase ranges
    for i in range(len(phases)):
        if phases[i] < -math.pi or phases[i] > math.pi:
            return False

    return True


@njit(fastmath=True, parallel=True)
def matrix_multiply_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized matrix multiplication using Numba."""
    m, n = a.shape[0], b.shape[1]
    p = a.shape[1]
    result = np.zeros((m, n), dtype=np.float64)

    for i in prange(m):
        for j in range(n):
            tmp = 0.0
            for k in range(p):
                tmp += a[i, k] * b[k, j]
            result[i, j] = tmp

    return result


@njit(fastmath=True, parallel=True)
def normalize_vector_numba(vector: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Normalize a vector using parallel processing."""
    norm_squared = 0.0
    for i in prange(len(vector)):
        norm_squared += abs(vector[i]) ** 2

    norm = math.sqrt(norm_squared)

    if norm < tolerance:
        return vector

    result = np.zeros_like(vector)
    for i in prange(len(vector)):
        result[i] = vector[i] / norm

    return result


@njit(fastmath=True)
def vector_distance_numba(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between vectors."""
    if len(a) != len(b):
        return float('inf')

    distance_squared = 0.0
    for i in range(len(a)):
        diff = a[i] - b[i]
        distance_squared += diff * diff

    return math.sqrt(distance_squared)


@njit(fastmath=True)
def hadamard_product_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate Hadamard (element-wise) product of vectors."""
    result = np.zeros_like(a)
    for i in range(len(a)):
        result[i] = a[i] * b[i]
    return result


@njit(fastmath=True)
def inner_product_numba(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate inner product of vectors."""
    result = 0.0
    for i in range(len(a)):
        result += a[i] * np.conj(b[i])
    return result


@njit(fastmath=True, parallel=True)
def process_vector_batch_numba(vectors: np.ndarray, batch_size: int) -> np.ndarray:
    """Process a batch of vectors in parallel."""
    num_vectors = vectors.shape[0]
    vector_size = vectors.shape[1]
    result = np.zeros((num_vectors, vector_size), dtype=np.float64)

    for i in prange(0, num_vectors, batch_size):
        end_idx = min(i + batch_size, num_vectors)
        batch = vectors[i:end_idx]

        # Process each vector in the batch
        for j in range(batch.shape[0]):
            result[i + j] = normalize_vector_numba(batch[j])

    return result