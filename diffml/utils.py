import inspect
import itertools
import os
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from diffml import DONN, ImagingSystem, LightPropagation


def get_fresnel_number(aperture_length, wavelength, propagation_distance):
    return (aperture_length / 2) ** 2 / (wavelength * propagation_distance)


def clone_system(original_system, **kwargs):
    # Get the class of the original system
    class_type = type(original_system)

    # Check if the class type is supported
    if class_type not in [DONN, ImagingSystem]:
        raise ValueError("Unsupported system type!")

    # Extract the attribute names from the constructor signature
    sig = inspect.signature(class_type.__init__)
    attribute_names = list(sig.parameters.keys())[1:]  # skip 'self'

    # Gather attributes and update with additional arguments
    new_args = {attr: getattr(original_system, attr) for attr in attribute_names}
    new_args.update(kwargs)

    # Create a new system instance with the gathered arguments
    new_system = class_type(**new_args)

    # Clone parameters and state dict from the original to the new instance
    original_state_dict = original_system.state_dict()
    filtered_state_dict = {k: v for k, v in original_state_dict.items() if k in new_system.state_dict()}
    new_system.load_state_dict(filtered_state_dict)

    # Transfer to the appropriate device
    device = next(itertools.chain(original_system.parameters(), original_system.buffers())).device
    return new_system.to(device)


def calculate_field_after_propagation(
        system,
        input,
        propagation_distance,
        output_dim,
):
    original_layer_distances = system.layer_distances
    original_system_dimensions = system.system_dimensions

    new_layer_distances = []
    new_system_dimensions = [original_system_dimensions[0]]
    current_distance = 0
    for i, layer_distance in enumerate(original_layer_distances):
        is_last_element = i == len(original_layer_distances) - 1
        is_distance_exceeded = current_distance + layer_distance > propagation_distance

        if is_distance_exceeded or is_last_element:
            new_layer_distances.append(propagation_distance - current_distance)
            new_system_dimensions.append(output_dim)
            break
        else:
            new_layer_distances.append(layer_distance)
            new_system_dimensions.append(original_system_dimensions[i + 1])
        current_distance += layer_distance

    new_system = clone_system(
        system, system_dimensions=new_system_dimensions, layer_distances=new_layer_distances
    )
    return new_system(input)


# TODO: Update
def plot_field(
        field: torch.Tensor,
        title: str = None,
        show: bool = True,
        filename: str = None,
        angle_twopi: bool = True,
        dx: float = None,
):
    field = field.squeeze().detach().cpu()

    titles = ["Intensity", "Angle", "Real", "Imaginary"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # increased figure size for better spacing
    field_plot = [field.abs().pow(2), field.angle(), field.real, field.imag]

    if dx is not None:
        grid_length = dx * (field.size(-1) - 1)
        extent = [-grid_length / 2, grid_length / 2]
    else:
        extent = None

    for i, ax in enumerate(axes.flat):
        im = ax.plot(field_plot[i])
        ax.set_title(titles[i])

        # Setting x-axis limits based on extent
        if extent is not None:
            ax.set_xlim(extent)

        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)

    # Setting the main title for the figure
    fig.suptitle(title, y=0.92)  # Adjust y to move the title down

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0)  # adjust only the width spacing

    if filename is not None:
        if not os.path.splitext(filename)[1]:
            filename += ".pdf"
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.02, dpi=500)

    if show:
        plt.show()


# TODO: Fix for complex values!
def plot_system_layers(
        system: DONN,
        title: str = None,
        show: bool = True,
        filename: str = None,
        angle_twopi: bool = True,
        dx: float = None,
):
    layer_profiles = system.get_layer_modulation_profiles()
    for i, layer_profile in enumerate(layer_profiles):
        plot_field(
            layer_profile,
            title=f"{title}\nLayer {i + 1} Modulation Profile" if title is not None else f"Layer {i + 1} Modulation Profile",
            show=show,
            filename=filename,
            angle_twopi=angle_twopi,
            dx=dx,
        )
