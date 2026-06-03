"""OrthoCache GPU test suite configuration.

Disables torch.compile (sets Dynamo to eager mode) so that all tests can run
on CPU without requiring a C++ compiler (cl, gcc) or CUDA.  The algorithm
correctness is identical in eager vs compiled mode — compilation only
affects performance.
"""

import torch._dynamo


def pytest_configure(config):
    """Called after command line options have been parsed and all plugins loaded."""
    # Suppress torch.compile errors: fall back to eager execution.
    # This lets us test algorithmic correctness on any machine without
    # requiring MSVC (cl.exe) or GCC for the Inductor backend.
    torch._dynamo.config.suppress_errors = True
