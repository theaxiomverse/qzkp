"""
QZKP - Quantum Zero-Knowledge Proof Implementation

A high-performance implementation of quantum-inspired zero-knowledge proofs
using Numba acceleration and parallel processing.
"""

from .optimized_qzkp import QuantumZKP, QuantumStateVector
from .numba_utils import (
    calculate_entropy_numba,
    generate_measurement_data_numba,
    verify_coefficients_numba,
    verify_measurement_data_numba,
    matrix_multiply_numba,
    normalize_vector_numba,
    vector_distance_numba,
    hadamard_product_numba,
    inner_product_numba,
    process_vector_batch_numba
)

__version__ = '0.1.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

__all__ = [
    'QuantumZKP',
    'QuantumStateVector',
    'calculate_entropy_numba',
    'generate_measurement_data_numba',
    'verify_coefficients_numba',
    'verify_measurement_data_numba',
    'matrix_multiply_numba',
    'normalize_vector_numba',
    'vector_distance_numba',
    'hadamard_product_numba',
    'inner_product_numba',
    'process_vector_batch_numba'
]
