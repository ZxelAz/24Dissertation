import torch

from .light_propagation import LightPropagation


class BaseLayer(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        pixel_size: float,
        wavelength: float,
        layer_distance: float,
        samples_per_pixel: int,
        propagation_method: str,
    ) -> None:
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.layer_distance = layer_distance
        self.samples_per_pixel = samples_per_pixel
        self.propagation_method = propagation_method
        self.light_propagation = LightPropagation(
            self.dim_in,
            self.dim_out,
            self.pixel_size,
            self.wavelength,
            self.layer_distance,
            self.samples_per_pixel,
            self.propagation_method,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # Must be initialized in subclass
        raise NotImplementedError

    def get_modulation_profile(self):
        # Must be initialized in subclass
        raise NotImplementedError


class InitialLayer(BaseLayer):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        pixel_size: float,
        wavelength: float,
        layer_distance: float,
        samples_per_pixel: int,
        propagation_method: str,
    ):
        super().__init__(
            dim_in,
            dim_out,
            pixel_size,
            wavelength,
            layer_distance,
            samples_per_pixel,
            propagation_method,
        )

        self.register_buffer(
            "initial_field",
            torch.ones(self.dim_in * self.samples_per_pixel),
            persistent=False,
        )

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        return self.light_propagation(self.initial_field, input_image)

    def get_modulation_profile(self, to_cpu=True):
        return None


class DiffractionLayer(BaseLayer):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        pixel_size: float,
        wavelength: float,
        layer_distance: float,
        samples_per_pixel: int,
        propagation_method: str,
        phase_modulation_init: str,
        amplitude_modulation_init: str,
    ):
        super().__init__(
            dim_in,
            dim_out,
            pixel_size,
            wavelength,
            layer_distance,
            samples_per_pixel,
            propagation_method,
        )

        # Phase-only modulation profile

        self.phase_modulation_init = phase_modulation_init
        self.amplitude_modulation_init = amplitude_modulation_init

        self.phase_modulation = self.phase_modulation_init is not None
        self.amplitude_modulation = self.amplitude_modulation_init is not None

        self.phase_params = self._init_phase_modulation()
        self.amp_params = self._init_amplitude_modulation()

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.light_propagation(field, self.get_modulation_profile(to_cpu=False))

    def get_modulation_profile(self, to_cpu=True):
        if self.phase_modulation and not self.amplitude_modulation:
            modulation_profile = torch.exp(1j * self.phase_params)
        elif self.amplitude_modulation and not self.phase_modulation:
            modulation_profile = torch.sigmoid(self.amp_params)
        else:
            modulation_profile = torch.exp(1j * self.phase_params) * (torch.sigmoid(self.amp_params))

        if to_cpu:
            return modulation_profile.detach().cpu()
        return modulation_profile

    def _init_phase_modulation(self):
        if self.phase_modulation:
            if self.phase_modulation_init == "random":
                return torch.nn.Parameter(
                     torch.randn(self.dim_in)*0.1
                )
            elif self.phase_modulation_init == "constant":
                return torch.nn.Parameter(torch.zeros(self.dim_in))
            else:
                raise ValueError(f"Unknown phase modulation initialization: {self.phase_modulation_init}")
        return None

    def _init_amplitude_modulation(self):
        if self.amplitude_modulation:
            if self.amplitude_modulation_init == "random":
                return torch.nn.Parameter(torch.randn(self.dim_in))
            elif self.amplitude_modulation_init == "constant":
                return torch.nn.Parameter(torch.zeros(self.dim_in))
            else:
                raise ValueError(
                    f"Unknown amplitude modulation initialization: {self.amplitude_modulation_init}"
                )
        return None


class LensLayer(BaseLayer):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        pixel_size: float,
        wavelength: float,
        layer_distance: float,
        samples_per_pixel: int,
        propagation_method: str,
        focal_length: float,
    ):
        super().__init__(
            dim_in,
            dim_out,
            pixel_size,
            wavelength,
            layer_distance,
            samples_per_pixel,
            propagation_method,
        )

        self.focal_length = focal_length
        modulation_profile = self._calculate_lens_modulation_profile()
        self.register_buffer("modulation_profile", modulation_profile, persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.light_propagation(field, self.get_modulation_profile(to_cpu=False))

    def get_modulation_profile(self, to_cpu=True):
        if to_cpu:
            return self.modulation_profile.detach().cpu()
        return self.modulation_profile

    def _calculate_lens_modulation_profile(self):
        lens_size = self.pixel_size * self.dim_in
        xv = torch.linspace(
            (-lens_size + self.pixel_size) / 2, (lens_size - self.pixel_size) / 2, self.dim_in
        ).double()

        radius = torch.sqrt(xv**2)
        phase_profile = torch.exp(-1j * torch.pi / (self.wavelength * self.focal_length) * radius**2)

        mask = radius > (lens_size - self.pixel_size) / 2
        phase_profile[mask] = 0
        return phase_profile.cfloat()
