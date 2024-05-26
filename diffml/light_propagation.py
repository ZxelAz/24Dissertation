import torch
from torch.fft import fft, ifft


class LightPropagation(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        pixel_size,
        wavelength,
        layer_distance,
        samples_per_pixel=1,
        propagation_algorithm="riemann",
    ):
        from .config import BACKENDS

        super().__init__()

        self.zero_propagation = layer_distance == 0
        if self.zero_propagation and dim_in != dim_out:
            raise ValueError(f"Input and ouput dimensions must be same if layer distance is zero.")
        self.samples_per_pixel = samples_per_pixel
        self.backend = BACKENDS[propagation_algorithm](
            dim_in,
            dim_out,
            pixel_size,
            wavelength,
            layer_distance,
            samples_per_pixel,
        )

    def forward(self, field, modulation_profile=None):
        if self.zero_propagation:
            return (
                modulate_field(field, modulation_profile, self.samples_per_pixel)
                if modulation_profile is not None
                else field
            )
        return self.backend(field, modulation_profile)

    def get_H(self):
        return self.backend.H


class RiemannBackend(torch.nn.Module):
    def __init__(self, dim_in, dim_out, pixel_size, wavelength, layer_distance, samples_per_pixel):
        super().__init__()

        self.samples_per_pixel = samples_per_pixel
        self.n_in_samples = dim_in * samples_per_pixel
        self.H = self._calculate_transfer_function(
            dim_in, dim_out, pixel_size, wavelength, layer_distance, samples_per_pixel
        )
        self.register_buffer("H_fr", fft(self.H), persistent=False)

    def forward(self, field, modulation_profile):
        self._check_input_dimensions(field)
        if modulation_profile is not None:
            field = modulate_field(field, modulation_profile, self.samples_per_pixel)
        return conv1d_fft(self.H_fr, field)

    def _calculate_transfer_function(
        self, dim_in, dim_out, pixel_size, wavelength, layer_distance, samples_per_pixel
    ):
        n_in_samples = dim_in * samples_per_pixel
        n_out_samples = dim_out * samples_per_pixel
        distance_offset = (n_out_samples + n_in_samples) / 2
        dx = pixel_size / samples_per_pixel
        differential_x = (torch.arange(-distance_offset + 1, distance_offset) * dx).double()

        r_squared = differential_x**2 + layer_distance**2
        r = torch.sqrt(r_squared)
        lambda_r_sqrt = torch.sqrt(r)*wavelength**0.5
        transfer_function = (
            ((layer_distance / r) * (1 / lambda_r_sqrt) * (1-1j) / 2**0.5)
            * torch.exp(2j * torch.pi * r / wavelength)
            * dx
        )
        return transfer_function.cfloat()

    def _check_input_dimensions(self, field):
        if field.shape[-1] != self.n_in_samples:
            raise ValueError(
                f"Field has incorrect size. Expected dimensions of {self.n_in_samples} but got {field.size(-1)}."
            )


@torch.jit.script
def conv1d_fft(H_fr: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Performs a 1D convolution using Fast Fourier Transforms (FFT).

    Args:
        H_fr (torch.Tensor): Fourier-transformed transfer function.
        x (torch.Tensor): Input complex field.

    Returns:
        torch.Tensor: Output field after convolution.
    """
    output_size = H_fr.size(-1) - x.size(-1) + 1
    x_fr = fft(x.conj(), dim=-1, n=H_fr.size(-1))
    output_fr = H_fr * x_fr.conj()
    output = ifft(output_fr)[..., :output_size].clone()
    return output


def scale_tensor(tensor, samples_per_pixel):
    return tensor.repeat_interleave(samples_per_pixel, dim=-1)


def modulate_field(field, modulation_profile, samples_per_pixel):
    return field * scale_tensor(modulation_profile, samples_per_pixel)
