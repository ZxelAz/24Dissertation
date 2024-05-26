from typing import Iterable, List, Type, Union

import torch

from .layers import DiffractionLayer, InitialLayer, LensLayer


class BaseOpticalSystem(torch.nn.Module):
    def __init__(
        self,
        system_dimensions: List[int],
        pixel_size: float,
        wavelength: float,
        layer_distances: Union[float, List[float]],
        samples_per_pixel: int,
        propagation_method: str,
    ) -> None:
        super().__init__()

        self.system_dimensions = system_dimensions
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.layer_distances = self._init_layer_distances(layer_distances)
        self.samples_per_pixel = samples_per_pixel
        self.propagation_method = propagation_method
        self.layers = None  # Must be initialized in subclass

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)

    def get_layer_modulation_profiles(self):
        return [
            layer.get_modulation_profile()
            for layer in self.layers
            if layer.get_modulation_profile() is not None
        ]

    def _init_layer_distances(self, layer_distances):
        target_length = len(self.system_dimensions) - 1
        if not isinstance(layer_distances, Iterable):
            return [layer_distances] * target_length
        if len(layer_distances) != target_length:
            raise ValueError(f"Length of `layer_distances` must be {target_length}")
        return list(layer_distances)


class DONN(BaseOpticalSystem):
    def __init__(
        self,
        system_dimensions: List[int],
        pixel_size: float,
        wavelength: float,
        layer_distances: Union[float, List[float]],
        samples_per_pixel: int,
        propagation_method: str,
        phase_modulation_init: str = "constant",
        amplitude_modulation_init: str = None,
    ) -> None:
        super().__init__(
            system_dimensions,
            pixel_size,
            wavelength,
            layer_distances,
            samples_per_pixel,
            propagation_method
        )

        self.phase_modulation_init = phase_modulation_init
        self.amplitude_modulation_init = amplitude_modulation_init

        self.layers = torch.nn.Sequential(
            *(
                [
                    InitialLayer(
                        self.system_dimensions[0],
                        self.system_dimensions[1],
                        self.pixel_size,
                        self.wavelength,
                        self.layer_distances[0],
                        self.samples_per_pixel,
                        self.propagation_method,
                    )
                ]
                + [
                    DiffractionLayer(
                        self.system_dimensions[i],
                        self.system_dimensions[i + 1],
                        self.pixel_size,
                        self.wavelength,
                        self.layer_distances[i],
                        self.samples_per_pixel,
                        self.propagation_method,
                        self.phase_modulation_init,
                        self.amplitude_modulation_init,
                    )
                    for i in range(1, len(self.system_dimensions) - 1)
                ]
            )
        )


class ImagingSystem(BaseOpticalSystem):
    def __init__(
        self,
        system_dimensions: List[int],
        pixel_size: float,
        wavelength: float,
        layer_distances: Union[float, List[float]],
        samples_per_pixel: int,
        propagation_method: str,
        focal_length: int,
    ) -> None:
        # Validate system dimensions
        if len(system_dimensions) != 3:
            raise ValueError("ImagingSystem must have exactly 3 layers")

        super().__init__(
            system_dimensions,
            pixel_size,
            wavelength,
            layer_distances,
            samples_per_pixel,
            propagation_method,
        )

        self.focal_length = focal_length
        self.layers = torch.nn.Sequential(
            InitialLayer(
                self.system_dimensions[0],
                self.system_dimensions[1],
                self.pixel_size,
                self.wavelength,
                self.layer_distances[0],
                self.samples_per_pixel,
                self.propagation_method,
            ),
            LensLayer(
                self.system_dimensions[1],
                self.system_dimensions[2],
                self.pixel_size,
                self.wavelength,
                self.layer_distances[1],
                self.samples_per_pixel,
                self.propagation_method,
                self.focal_length,
            ),
        )

    @staticmethod
    def calculate_focal_length(object_distance: float, image_distance: float) -> float:
        """
        Calculate the focal length given object and image distances.

        Args:
        - object_distance (float): Distance from the object to the lens.
        - image_distance (float): Distance from the image to the lens.

        Returns:
        - float: Focal length.
        """
        return object_distance * image_distance / (object_distance + image_distance)
