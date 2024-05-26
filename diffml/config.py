# config.py
from .light_propagation import RiemannBackend

BACKENDS = {
    "riemann": RiemannBackend,
    # Add other backends as they become available
}
