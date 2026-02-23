import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np
from tqdm import tqdm
"""
BLT_VS Architecture

This file implements the full BLT-VS (Bottom-Up, Lateral, Top-Down Visual Stream) model.

The architecture is biologically inspired and mimics the hierarchical structure
of the human ventral visual stream:

Retina → LGN → V1 → V2 → V3 → V4 → LOC → Readout

Key ideas of the model:

1. Bottom-Up Processing:
   Information flows upward like in a normal CNN.

2. Top-Down Feedback:
   Higher visual areas send signals back to lower areas
   to modulate and refine representations.

3. Lateral Connections:
   Each area processes its own previous activation,
   giving the network temporal memory.

4. Recurrent Processing over Timesteps:
   The model runs for multiple timesteps.
   Representations are gradually refined.

5. Multiplicative Gating:
   Bottom-up signals are modulated (gated) by top-down signals
   using a sigmoid-based multiplicative interaction.
   This is the core biological mechanism of the architecture.

Main Classes:

- BLT_VS:
    The full multi-area recurrent network.

- BLT_VS_Layer:
    Implements a single cortical area (e.g., V1, V2, etc.)
    with bottom-up, lateral, and top-down processing.

- BLT_VS_Readout:
    Final classifier that produces class logits
    and also generates top-down feedback.

The forward pass simulates signal flow over time,
either in a fully synchronous recurrent mode
or in a biologically unrolled mode where signals
propagate gradually across areas.
"""


class BLT_VS(nn.Module):
    """
    BLT_VS model simulates the ventral stream of the visual cortex.

    Parameters:
    -----------
    timesteps : int
        Number of time steps for the recurrent computation.
    num_classes : int
        Number of output classes for classification.
    add_feats : int
        Additional features to maintain orientation, color, etc.
    lateral_connections : bool
        Whether to include lateral connections.
    topdown_connections : bool
        Whether to include top-down connections.
    skip_connections : bool
        Whether to include skip connections.
    bio_unroll : bool
        Whether to use biological unrolling.
    image_size : int
        Size of the input image (height and width) - should be 224 or 128px
    hook_type : str
        What kind of area/timestep hooks to register. Options are 'concat' (concat BU/TD), 'separate', 'None'.
    readout_type : str
        Type of readout layer. Options are 'multi' (multi-class) or 'single' (weighted sum of readouts).
    """

    def __init__(
        self,
        timesteps=12,
        num_classes=565,
        add_feats=100,
        lateral_connections=True,
        topdown_connections=True,
        skip_connections=True,
        bio_unroll=True,
        image_size=224,
        hook_type='None',
        readout_type='multi'
    ):
        super(BLT_VS, self).__init__()  # Initialize PyTorch nn.Module

        # Store all configuration parameters inside the model object
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.add_feats = add_feats
        self.lateral_connections = lateral_connections
        self.topdown_connections = topdown_connections
        self.skip_connections = skip_connections
        self.bio_unroll = bio_unroll
        self.image_size = image_size
        self.hook_type = hook_type
        self.readout_type = readout_type

        # Names of all visual areas in the model
        self.areas = ["Retina", "LGN", "V1", "V2", "V3", "V4", "LOC", "Readout"]

        # Only allow supported image sizes
        if image_size not in [224, 128]:
            raise ValueError("Image size must be 224 or 128.")

        # Kernel sizes depend on input resolution
        # Larger images use slightly larger kernels
        if image_size == 224:
            self.kernel_sizes = [7, 7, 5, 1, 5, 3, 3, 5]  # BU kernel sizes
            self.kernel_sizes_lateral = [0, 0, 5, 5, 5, 5, 5, 0]  # Lateral kernels
        elif image_size == 128:
            self.kernel_sizes = [5, 3, 3, 1, 3, 3, 3, 3]
            self.kernel_sizes_lateral = [0, 0, 3, 3, 3, 3, 3, 0]

        # Stride defines downsampling between areas
        self.strides = [2, 2, 2, 1, 1, 1, 2, 2]

        # Compute padding such that output size roughly stays aligned ("same" padding)
        self.paddings = (np.array(self.kernel_sizes) - 1) // 2  

        # Number of feature channels per area
        # Last layer has num_classes + add_feats channels
        self.channel_sizes = [
            32,   # Retina
            32,   # LGN
            576,  # V1
            480,  # V2
            352,  # V3
            256,  # V4
            352,  # LOC
            int(num_classes + add_feats),  # Readout layer
        ]

        # Define which areas receive top-down feedback
        self.topdown_connections_layers = [
            False,  # Retina
            True,   # LGN
            True,   # V1
            True,   # V2
            True,   # V3
            True,   # V4
            True,   # LOC
            False,  # Readout
        ]

        # Create dictionary that will store all cortical areas
        self.connections = nn.ModuleDict()

        # Create BLT_VS_Layer for each area except Readout
        for idx in range(len(self.areas) - 1):
            area = self.areas[idx]

            self.connections[area] = BLT_VS_Layer(
                layer_n=idx,  # index of the area
                channel_sizes=self.channel_sizes,
                strides=self.strides,
                kernel_sizes=self.kernel_sizes,
                kernel_sizes_lateral=self.kernel_sizes_lateral,
                paddings=self.paddings,

                # Enable lateral only if globally allowed AND kernel > 0
                lateral_connections=self.lateral_connections
                and (self.kernel_sizes_lateral[idx] > 0),

                # Enable top-down only if globally allowed AND area supports it
                topdown_connections=self.topdown_connections
                and self.topdown_connections_layers[idx],

                # Skip connection from V1 → V4 (only idx == 5)
                skip_connections_bu=self.skip_connections and (idx == 5),

                # Skip connection from V4 → V1 (only idx == 2)
                skip_connections_td=self.skip_connections and (idx == 2),

                image_size=image_size,
            )

        # Create final readout layer (classifier)
        self.connections["Readout"] = BLT_VS_Readout(
            layer_n=7,
            channel_sizes=self.channel_sizes,
            kernel_sizes=self.kernel_sizes,
            strides=self.strides,
            num_classes=num_classes,
        )

        # If using single readout type, create learnable weights over timesteps
        if self.readout_type == 'single':
            if self.bio_unroll:
                # LOC output becomes available at timestep 4
                self.readout_weights = nn.Parameter(torch.ones(timesteps-4))
            else:
                self.readout_weights = nn.Parameter(torch.ones(timesteps))

        # Create Identity layers so we can attach forward hooks later
        # These allow us to capture activations per area per timestep
        if self.hook_type != 'None':
            for area in self.areas:
                for t in range(timesteps):
                    if self.hook_type == 'concat' and area != 'Readout':
                        setattr(self, f"{area}_{t}", nn.Identity())
                    elif self.hook_type == 'separate':
                        setattr(self, f"{area}_{t}_BU", nn.Identity())
                        setattr(self, f"{area}_{t}_TD", nn.Identity())

        # Precompute output spatial shapes for each area
        self.output_shapes = self.compute_output_shapes(image_size)

    def compute_output_shapes(self, image_size):
        """
        Compute the output shapes for each area based on the image size.

        Parameters:
        -----------
        image_size : int
            The input image size.

        Returns:
        --------
        output_shapes : list of tuples
            The output height and width for each area.

        Why this is needed:
        -------------------
        The model contains top-down (feedback) connections where higher areas
        send signals back to lower areas using transposed convolutions.
        For these feedback signals to work correctly, the spatial dimensions
        (height and width) must exactly match the lower area's feature maps.
        This function precomputes the expected output sizes for every area
        so that bottom-up, top-down, and skip connections can be constructed
        without spatial mismatches.
        """
        output_shapes = []  # Will store (height, width) for each area

        height = width = image_size  # Start with the input image size

        # Loop over all visual areas (Retina → ... → Readout)
        for idx in range(len(self.areas)):

            kernel_size = self.kernel_sizes[idx]  # Kernel size used in this area
            stride = self.strides[idx]            # Stride used in this area
            padding = self.paddings[idx]          # Padding used in this area

            # Apply standard convolution output size formula:
            # output = (input + 2*padding - kernel_size) // stride + 1
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1

            # Store the computed spatial size for this area
            output_shapes.append((int(height), int(width)))

        # Return list like:
        # [(H_retina, W_retina), (H_LGN, W_LGN), ..., (H_readout, W_readout)]
        return output_shapes

    def forward(
        self,
        img_input,
        extract_actvs=False,
        areas=None,
        timesteps=None,
        bu=True,
        td=True,
        concat=False,
    ):
        """
        Forward pass for the BLT_VS model.

        Parameters:
        -----------
        img_input : torch.Tensor
            Input image tensor.
        extract_actvs : bool
            Whether to extract activations.
        areas : list of str
            List of area names to retrieve activations from.
        timesteps : list of int
            List of timesteps to retrieve activations at.
        bu : bool
            Whether to retrieve bottom-up activations.
        td : bool
            Whether to retrieve top-down activations.
        concat : bool
            Whether to concatenate BU and TD activations.

        Returns:
        --------
        If extract_actvs is False:
            readout_output : list of torch.Tensor
                The readout outputs at each timestep.
        If extract_actvs is True:
            (readout_output, activations) : tuple
                readout_output is as above.
                activations is a dict with structure activations[area][timestep] = activation

        Why this is needed:
        -------------------
        This function performs the actual recurrent computation of the model.
        It simulates signal flow through the visual hierarchy over multiple
        timesteps, including bottom-up processing, lateral recurrence,
        and top-down feedback. Without this method, the architecture defined
        in __init__ would only exist structurally — no image would be processed.
        """

        # Ensure input image has correct spatial size
        if img_input.size(2) != self.image_size or img_input.size(3) != self.image_size:
            raise ValueError(
                f"Input image size must be {self.image_size}x{self.image_size}."
            )

        # If user wants activations, prepare storage dictionary
        if extract_actvs:
            if areas is None or timesteps is None:
                raise ValueError(
                    "When extract_actvs is True, areas and timesteps must be specified."
                )
            activations = {area: {} for area in areas}
        else:
            activations = None

        readout_output = []  # Store classification outputs per timestep

        # Store bottom-up and top-down activations per area
        bu_activations = [None for _ in self.areas]
        td_activations = [None for _ in self.areas]

        batch_size = img_input.size(0)

        # ===============================
        # BIOLOGICAL UNROLL MODE
        # ===============================
        if self.bio_unroll:

            # Store previous timestep activations
            bu_activations_old = [None for _ in self.areas]
            td_activations_old = [None for _ in self.areas]

            # Retina processes input first
            bu_activations_old[0], _ = self.connections["Retina"](bu_input=img_input)
            bu_activations[0] = bu_activations_old[0]

            # Timestep 0 activation collection
            t = 0
            activations = self.activation_shenanigans(
                extract_actvs, areas, timesteps, bu, td, concat,
                batch_size, bu_activations, td_activations, activations, t
            )

            # Iterate through remaining timesteps
            for t in range(1, self.timesteps):

                # Update intermediate areas (LGN → LOC)
                for idx, area in enumerate(self.areas[1:-1]):

                    # Only update area if it receives some input signal
                    should_update = any(
                        [
                            bu_activations_old[idx] is not None,  # bottom-up
                            (bu_activations_old[2] is not None and (idx + 1) == 5),  # skip BU
                            td_activations_old[idx + 2] is not None,  # top-down
                            (td_activations_old[5] is not None and (idx + 1) == 2),  # skip TD
                        ]
                    )

                    if should_update:
                        # Compute BU and TD outputs for this area
                        bu_act, td_act = self.connections[area](
                            bu_input=bu_activations_old[idx],
                            bu_l_input=bu_activations_old[idx + 1],
                            td_input=td_activations_old[idx + 2],
                            td_l_input=td_activations_old[idx + 1],
                            bu_skip_input=bu_activations_old[2]
                            if (idx + 1) == 5 else None,
                            td_skip_input=td_activations_old[5]
                            if (idx + 1) == 2 else None,
                        )

                        # Store new activations
                        bu_activations[idx + 1] = bu_act
                        td_activations[idx + 1] = td_act

                # Move current activations to old for next timestep
                bu_activations_old = bu_activations[:]
                td_activations_old = td_activations[:]

                # Activate readout once LOC has produced output
                if bu_activations_old[-2] is not None:
                    bu_act, td_act = self.connections["Readout"](
                        bu_input=bu_activations_old[-2]
                    )
                    bu_activations_old[-1] = bu_act
                    td_activations_old[-1] = td_act
                    readout_output.append(bu_act)

                    bu_activations[-1] = bu_act
                    td_activations[-1] = td_act

                # Store activations if requested
                activations = self.activation_shenanigans(
                    extract_actvs, areas, timesteps, bu, td, concat,
                    batch_size, bu_activations, td_activations, activations, t
                )

        # ===============================
        # STANDARD RECURRENT MODE
        # ===============================
        else:

            # Initial bottom-up sweep
            bu_activations[0], _ = self.connections["Retina"](bu_input=img_input)

            for idx, area in enumerate(self.areas[1:-1]):
                bu_act, _ = self.connections[area](
                    bu_input=bu_activations[idx],
                    bu_skip_input=bu_activations[2] if idx + 1 == 5 else None,
                )
                bu_activations[idx + 1] = bu_act

            # Compute initial readout
            bu_act, td_act = self.connections["Readout"](bu_input=bu_activations[-2])
            bu_activations[-1] = bu_act
            td_activations[-1] = td_act
            readout_output.append(bu_act)

            # Initial top-down sweep
            for idx, area in enumerate(reversed(self.areas[1:-1])):
                _, td_act = self.connections[area](
                    bu_input=bu_activations[-(idx + 2) - 1],
                    td_input=td_activations[-(idx + 2) + 1],
                    td_skip_input=td_activations[5] if idx + 1 == 2 else None,
                )
                td_activations[-(idx + 2)] = td_act

            _, td_act = self.connections["Retina"](
                bu_input=img_input,
                td_input=td_activations[1],
            )
            td_activations[0] = td_act

            # Store timestep 0 activations
            t = 0
            activations = self.activation_shenanigans(
                extract_actvs, areas, timesteps, bu, td, concat,
                batch_size, bu_activations, td_activations, activations, t
            )

            # Repeat recurrent refinement
            for t in range(1, self.timesteps):

                # Bottom-up update
                for idx, area in enumerate(self.areas[1:-1]):
                    bu_act, _ = self.connections[area](
                        bu_input=bu_activations[idx],
                        bu_l_input=bu_activations[idx + 1],
                        td_input=td_activations[idx + 2],
                        bu_skip_input=bu_activations[2] if idx + 1 == 5 else None,
                    )
                    bu_activations[idx + 1] = bu_act

                # Readout update
                bu_act, td_act = self.connections["Readout"](bu_input=bu_activations[-2])
                bu_activations[-1] = bu_act
                td_activations[-1] = td_act
                readout_output.append(bu_act)

                # Top-down update
                for idx, area in enumerate(reversed(self.areas[1:-1])):
                    _, td_act = self.connections[area](
                        bu_input=bu_activations[-(idx + 2) - 1],
                        td_input=td_activations[-(idx + 2) + 1],
                        td_l_input=td_activations[-(idx + 2)],
                        td_skip_input=td_activations[5] if idx + 1 == 2 else None,
                    )
                    td_activations[-(idx + 2)] = td_act

                _, td_act = self.connections["Retina"](
                    bu_input=img_input,
                    td_input=td_activations[1],
                    td_l_input=td_activations[0],
                )
                td_activations[0] = td_act

                # Store activations if requested
                activations = self.activation_shenanigans(
                    extract_actvs, areas, timesteps, bu, td, concat,
                    batch_size, bu_activations, td_activations, activations, t
                )

        # ===============================
        # READOUT HANDLING
        # ===============================

        if self.readout_type == 'single':
            # Stack outputs across time
            outputs = torch.stack(readout_output, dim=0)

            # Reshape to (batch_size, timesteps, num_classes)
            outputs = outputs.permute(1, 0, 2)

            # Learnable time weights (softmax normalized)
            readout_weights = F.softmax(self.readout_weights, dim=0)

            # Reshape weights for broadcasting
            if self.bio_unroll:
                readout_weights = readout_weights.view(1, self.timesteps-4, 1)
            else:
                readout_weights = readout_weights.view(1, self.timesteps, 1)

            # Weighted temporal integration
            weighted_outputs = outputs * readout_weights
            final_outputs = [weighted_outputs.sum(dim=1)]
        else:
            # Return list of logits per timestep
            final_outputs = readout_output

        # Return outputs (and activations if requested)
        if extract_actvs:
            return final_outputs, activations
        else:
            return final_outputs

        
        
    def activation_shenanigans(
            self, extract_actvs, areas, timesteps, bu, td, concat, batch_size, bu_activations, td_activations, activations, t
    ):
        """
        Helper function to implement activation collection and compute relevant for hook registration.

        Parameters:
        -----------
        extract_actvs : bool
            Whether to extract activations.
        areas : list of str
            List of area names to retrieve activations from.
        timesteps : list of int
            List of timesteps to retrieve activations at.
        bu : bool
            Whether to retrieve bottom-up activations.
        td : bool
            Whether to retrieve top-down activations.
        concat : bool
            Whether to concatenate BU and TD activations.
        batch_size : int
            Batch size of the input data.
        bu_activations : list of torch.Tensor
            List of bottom-up activations.
        td_activations : list of torch.Tensor
            List of top-down activations.
        activations : dict
            Dictionary to store activations.
        t : int
            Current timestep.

        Returns:
        --------
        activations : dict
            Updated activations dictionary.

        Why this is needed:
        -------------------
        This function centralizes all activation extraction logic so that the
        main forward() method remains clean and readable. It allows researchers
        to retrieve intermediate bottom-up and/or top-down activations at
        specific areas and timesteps without modifying the core computation.
        Additionally, it handles optional hook execution for debugging or
        feature visualization purposes.
        """
        if extract_actvs and t in timesteps:

            # Loop through all areas (Retina → Readout)
            for idx, area in enumerate(self.areas):

                # Only process areas requested by the user
                if area in areas:

                    # If concatenation is requested but area is Readout,
                    # skip it (Readout does not support concatenation of BU/TD)
                    if concat and area == 'Readout':
                        continue

                    # Collect activation using helper function
                    # This handles BU only, TD only, or concatenated
                    activation = self.collect_activation(
                        bu_activations[idx],   # Bottom-up activation of this area
                        td_activations[idx],   # Top-down activation of this area
                        bu,                    # Whether BU should be included
                        td,                    # Whether TD should be included
                        concat,                # Whether to concatenate BU and TD
                        idx,                   # Index of the area
                        batch_size,            # Needed for safe fallback shapes
                    )

                    # Store activation under activations[area][timestep]
                    activations[area][t] = activation

        # ======================================================
        # Hook Execution (for automatic PyTorch hooks)
        # ======================================================

        # If hooks are enabled, push activations through Identity layers
        if self.hook_type != 'None':

            # Loop through all areas
            for idx, area in enumerate(self.areas):

                # CONCAT MODE:
                # Send concatenated BU+TD through identity layer
                if self.hook_type == 'concat' and area != 'Readout':

                    _ = getattr(self, f"{area}_{t}")(
                        concat_or_not(bu_activations[idx], td_activations[idx], dim=1)
                    )

                # SEPARATE MODE:
                # Send BU and TD separately through identity layers
                elif self.hook_type == 'separate':

                    _ = getattr(self, f"{area}_{t}_BU")(bu_activations[idx])
                    _ = getattr(self, f"{area}_{t}_TD")(td_activations[idx])

        # Return possibly updated activation dictionary
        return activations

def collect_activation(
    self, bu_activation, td_activation, bu_flag, td_flag, concat, area_idx, batch_size
):
    """
    Helper function to collect activations, handling None values and concatenation.

    Parameters:
    -----------
    bu_activation : torch.Tensor or None
        Bottom-up activation.
    td_activation : torch.Tensor or None
        Top-down activation.
    bu_flag : bool
        Whether to collect BU activations.
    td_flag : bool
        Whether to collect TD activations.
    concat : bool
        Whether to concatenate BU and TD activations.
    area_idx : int
        Index of the area in self.areas.
    batch_size : int
        Batch size of the input data.

    Returns:
    --------
    activation : torch.Tensor or dict
        The collected activation. If concat is True, returns a single tensor.
        If concat is False, returns a dict with keys 'bu' and/or 'td'.

    Why this is needed:
    -------------------
    During recurrent computation, some areas may not yet have valid
    bottom-up or top-down activations (they may be None).
    This function guarantees that activation extraction never crashes
    by safely replacing missing activations with zero tensors of the
    correct shape. It also standardizes how BU and TD activations are
    returned (either concatenated or separate), ensuring consistent
    behavior for analysis and visualization.
    """

    # Determine the device (CPU or GPU) where the model parameters live
    # This ensures created zero tensors are placed on the correct device
    device = next(self.parameters()).device  

    # ======================================================
    # CONCAT MODE (Return a single tensor)
    # ======================================================
    if concat:

        # Case 1: Both BU and TD are None
        # → create a zero tensor with correct spatial and channel size
        if bu_activation is None and td_activation is None:

            # Channel size doubled because we concatenate BU and TD
            channels = self.channel_sizes[area_idx] * 2  

            # Get spatial resolution for this area
            height, width = self.output_shapes[area_idx]

            # Create zero tensor of expected shape
            zeros = torch.zeros(
                (batch_size, channels, height, width),
                device=device
            )
            return zeros

        # Case 2: BU is missing → replace with zeros shaped like TD
        if bu_activation is None:
            bu_activation = torch.zeros_like(td_activation)

        # Case 3: TD is missing → replace with zeros shaped like BU
        if td_activation is None:
            td_activation = torch.zeros_like(bu_activation)

        # Concatenate along channel dimension (dim=1)
        activation = torch.cat([bu_activation, td_activation], dim=1)

        return activation

    # ======================================================
    # SEPARATE MODE (Return dictionary with 'bu' and/or 'td')
    # ======================================================
    else:

        activation = {}

        # ---------------------------
        # Handle Bottom-Up (BU)
        # ---------------------------
        if bu_flag:

            # If BU exists, use it directly
            if bu_activation is not None:
                activation['bu'] = bu_activation

            # If BU is missing but TD exists,
            # create zero tensor shaped like TD
            elif td_activation is not None:
                activation['bu'] = torch.zeros_like(td_activation)

            # If both are None → create zero tensor from scratch
            else:
                channels = self.channel_sizes[area_idx]
                height, width = self.output_shapes[area_idx]

                activation['bu'] = torch.zeros(
                    (batch_size, channels, height, width),
                    device=device
                )

        # ---------------------------
        # Handle Top-Down (TD)
        # ---------------------------
        if td_flag:

            # If TD exists, use it directly
            if td_activation is not None:
                activation['td'] = td_activation

            # If TD missing but BU exists → create zero tensor shaped like BU
            elif bu_activation is not None:
                activation['td'] = torch.zeros_like(bu_activation)

            # If both are None → create zero tensor from scratch
            else:
                channels = self.channel_sizes[area_idx]
                height, width = self.output_shapes[area_idx]

                activation['td'] = torch.zeros(
                    (batch_size, channels, height, width),
                    device=device
                )

        return activation



class BLT_VS_Layer(nn.Module):
    """
    A single layer in the BLT_VS model, representing a cortical area.

    Parameters:
    -----------
    layer_n : int
        Layer index.
    channel_sizes : list
        List of channel sizes for each layer.
    strides : list
        List of strides for each layer.
    kernel_sizes : list
        List of kernel sizes for each layer.
    kernel_sizes_lateral : list
        List of lateral kernel sizes for each layer.
    paddings : list
        List of paddings for each layer.
    lateral_connections : bool
        Whether to include lateral connections.
    topdown_connections : bool
        Whether to include top-down connections.
    skip_connections_bu : bool
        Whether to include bottom-up skip connections.
    skip_connections_td : bool
        Whether to include top-down skip connections.
    image_size : int
        Size of the input image (height and width).

    Why this is needed:
    -------------------
    This class implements the computation of a single cortical area
    in the ventral visual stream model. Each area performs bottom-up
    processing, optional lateral recurrence (within-area processing),
    and optional top-down feedback from higher areas. It encapsulates
    all convolutional operations required for that area so that the
    full BLT_VS model can be built by stacking multiple such layers.
    """

    def __init__(
        self,
        layer_n,
        channel_sizes,
        strides,
        kernel_sizes,
        kernel_sizes_lateral,
        paddings,
        lateral_connections=True,
        topdown_connections=True,
        skip_connections_bu=False,
        skip_connections_td=False,
        image_size=224,
    ):
        super(BLT_VS_Layer, self).__init__()

        # --------------------------------------------------
        # Determine input and output channel sizes
        # --------------------------------------------------

        # First layer (Retina) receives RGB input → 3 channels
        # All other layers receive previous layer's channels
        in_channels = 3 if layer_n == 0 else channel_sizes[layer_n - 1]

        # Output channels defined by architecture configuration
        out_channels = channel_sizes[layer_n]

        # --------------------------------------------------
        # Bottom-Up Convolution (standard forward pathway)
        # --------------------------------------------------

        # This is the main feedforward convolution:
        # previous area → current area
        self.bu_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[layer_n],
            stride=strides[layer_n],
            padding=paddings[layer_n],
        )

        # --------------------------------------------------
        # Lateral Connections (within-area recurrence)
        # --------------------------------------------------

        if lateral_connections:

            kernel_size_lateral = kernel_sizes_lateral[layer_n]

            # Depthwise convolution:
            # each channel processed independently
            self.bu_l_conv_depthwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size_lateral,
                stride=1,
                padding='same',
                groups=out_channels,  # depthwise operation
            )

            # Pointwise convolution:
            # mixes information across channels
            self.bu_l_conv_pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        else:
            # If lateral disabled → do nothing
            self.bu_l_conv_depthwise = NoOpModule()
            self.bu_l_conv_pointwise = NoOpModule()

        # --------------------------------------------------
        # Top-Down Connections (feedback pathway)
        # --------------------------------------------------

        if topdown_connections:

            # Transposed convolution:
            # higher area → current area (upsampling)
            self.td_conv = nn.ConvTranspose2d(
                in_channels=channel_sizes[layer_n + 1],
                out_channels=out_channels,
                kernel_size=kernel_sizes[layer_n + 1],
                stride=strides[layer_n + 1],
                padding=(kernel_sizes[layer_n + 1] - 1) // 2
            )

            # Optional lateral processing of top-down signal
            if lateral_connections:

                self.td_l_conv_depthwise = nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes_lateral[layer_n],
                    stride=1,
                    padding='same',
                    groups=out_channels,
                )

                self.td_l_conv_pointwise = nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

            else:
                self.td_l_conv_depthwise = NoOpModule()
                self.td_l_conv_pointwise = NoOpModule()

        else:
            # If no top-down → no feedback computation
            self.td_conv = NoOpModule()
            self.td_l_conv_depthwise = NoOpModule()
            self.td_l_conv_pointwise = NoOpModule()

        # --------------------------------------------------
        # Bottom-Up Skip Connection (V1 → V4)
        # --------------------------------------------------

        if skip_connections_bu:

            # Depthwise skip projection from V1
            self.skip_bu_depthwise = nn.Conv2d(
                in_channels=channel_sizes[2],  # From V1
                out_channels=out_channels,
                kernel_size=7 if image_size == 224 else 5,
                stride=1,
                padding='same',
                groups=np.gcd(channel_sizes[2], out_channels),
            )

            # Channel mixing
            self.skip_bu_pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        else:
            self.skip_bu_depthwise = NoOpModule()
            self.skip_bu_pointwise = NoOpModule()

        # --------------------------------------------------
        # Top-Down Skip Connection (V4 → V1)
        # --------------------------------------------------

        if skip_connections_td:

            self.skip_td_depthwise = nn.Conv2d(
                in_channels=channel_sizes[5],  # From V4
                out_channels=out_channels,
                kernel_size=3,  # V4 to V1 skip
                stride=1,
                padding='same',
                groups=np.gcd(channel_sizes[5], out_channels),
            )

            self.skip_td_pointwise = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        else:
            self.skip_td_depthwise = NoOpModule()
            self.skip_td_pointwise = NoOpModule()

        # --------------------------------------------------
        # Normalization Layers
        # --------------------------------------------------

        # GroupNorm with 1 group behaves similar to LayerNorm over channels
        # Applied separately for BU and TD pathways
        self.layer_norm_bu = nn.GroupNorm(num_groups=1, num_channels=out_channels)
        self.layer_norm_td = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def forward(
        self,
        bu_input,
        bu_l_input=None,
        td_input=None,
        td_l_input=None,
        bu_skip_input=None,
        td_skip_input=None,
    ):
        """
        Forward pass for a single BLT_VS layer.

        Parameters:
        -----------
        bu_input : torch.Tensor or None
            Bottom-up input tensor.
        bu_l_input : torch.Tensor or None
            Bottom-up lateral input tensor.
        td_input : torch.Tensor or None
            Top-down input tensor.
        td_l_input : torch.Tensor or None
            Top-down lateral input tensor.
        bu_skip_input : torch.Tensor or None
            Bottom-up skip connection input.
        td_skip_input : torch.Tensor or None
            Top-down skip connection input.

        Returns:
        --------
        bu_output : torch.Tensor
            Bottom-up output tensor.
        td_output : torch.Tensor
            Top-down output tensor.

        Why this is needed:
        -------------------
        This function defines the core computation of one cortical area.
        It integrates bottom-up input, lateral recurrence, top-down feedback,
        and optional skip connections. The key biological mechanism implemented
        here is multiplicative gating: bottom-up signals are modulated by
        top-down signals and vice versa. This interaction allows higher areas
        to dynamically refine and control lower-level representations over time.
        """

        # ======================================================
        # Process Bottom-Up Input (feedforward pathway)
        # ======================================================

        # If BU input exists, apply bottom-up convolution
        # Otherwise return 0 (no signal yet)
        bu_processed = self.bu_conv(bu_input) if bu_input is not None else 0

        # ======================================================
        # Process Top-Down Input (feedback pathway)
        # ======================================================

        # If TD input exists, apply transposed convolution (upsampling)
        # Ensure output matches BU spatial size
        td_processed = (
            self.td_conv(td_input, output_size=bu_processed.size())
            if td_input is not None
            else 0
        )

        # ======================================================
        # Process Bottom-Up Lateral Input (within-area recurrence)
        # ======================================================

        # Apply depthwise + pointwise conv if lateral signal exists
        bu_l_processed = (
            self.bu_l_conv_pointwise(self.bu_l_conv_depthwise(bu_l_input))
            if bu_l_input is not None
            else 0
        )

        # ======================================================
        # Process Top-Down Lateral Input
        # ======================================================

        td_l_processed = (
            self.td_l_conv_pointwise(self.td_l_conv_depthwise(td_l_input))
            if td_l_input is not None
            else 0
        )

        # ======================================================
        # Process Skip Connections
        # ======================================================

        skip_bu_processed = (
            self.skip_bu_pointwise(self.skip_bu_depthwise(bu_skip_input))
            if bu_skip_input is not None
            else 0
        )

        skip_td_processed = (
            self.skip_td_pointwise(self.skip_td_depthwise(td_skip_input))
            if td_skip_input is not None
            else 0
        )

        # ======================================================
        # Combine Signals
        # ======================================================

        # BU drive = all bottom-up sources
        bu_drive = bu_processed + bu_l_processed + skip_bu_processed

        # BU modulator = only main BU pathway (used to modulate TD)
        bu_mod = bu_processed + skip_bu_processed

        # TD drive = all top-down sources
        td_drive = td_processed + td_l_processed + skip_td_processed

        # TD modulator = main TD pathway (used to modulate BU)
        td_mod = td_processed + skip_td_processed

        # ======================================================
        # Compute Bottom-Up Output (Gated by Top-Down)
        # ======================================================

        if isinstance(td_mod, torch.Tensor):

            if isinstance(bu_drive, torch.Tensor):

                # Core biological gating:
                # ReLU(bu_drive) scaled by sigmoid(td_mod)
                bu_output = F.relu(bu_drive) * 2 * torch.sigmoid(td_mod)

            else:
                # If no BU drive but TD exists → output zeros
                bu_output = torch.zeros_like(td_mod)

        else:
            # If no TD modulation → standard ReLU
            bu_output = F.relu(bu_drive)

        # ======================================================
        # Compute Top-Down Output (Gated by Bottom-Up)
        # ======================================================

        if isinstance(bu_mod, torch.Tensor):

            if isinstance(td_drive, torch.Tensor):

                # Symmetric gating mechanism
                td_output = F.relu(td_drive) * 2 * torch.sigmoid(bu_mod)

            else:
                td_output = torch.zeros_like(bu_mod)

        else:
            td_output = F.relu(td_drive)

        # ======================================================
        # Normalize Outputs
        # ======================================================

        bu_output = self.layer_norm_bu(bu_output)
        td_output = self.layer_norm_td(td_output)

        return bu_output, td_output


class BLT_VS_Readout(nn.Module):
    """
    Readout layer for the BLT_VS model.

    Parameters:
    -----------
    layer_n : int
        Layer index.
    channel_sizes : list
        List of channel sizes for each layer.
    kernel_sizes : list
        List of kernel sizes for each layer.
    strides : list
        List of strides for each layer.
    num_classes : int
        Number of output classes for classification.

    Why this is needed:
    -------------------
    This layer converts the high-level feature representation (LOC output)
    into class scores for classification. It performs a final convolution,
    followed by global average pooling to collapse spatial dimensions,
    and outputs logits for each class. For each class, a single multi-channel
    convolutional filter looks at all previous feature maps together and produces
    a class-specific activation map whose global average becomes the final class score.
    Additionally, it generates a top-down signal that is sent back into the network during
    recurrent processing.
    """

    def __init__(self, layer_n, channel_sizes, kernel_sizes, strides, num_classes):
        super(BLT_VS_Readout, self).__init__()

        # Store number of classification categories
        self.num_classes = num_classes

        # Input channels come from previous layer (LOC)
        in_channels = channel_sizes[layer_n - 1]

        # Output channels = num_classes + add_feats
        # (add_feats are extra features used for top-down feedback)
        out_channels = channel_sizes[layer_n]

        # --------------------------------------------------
        # Final convolution
        # --------------------------------------------------
        # Transforms high-level features into class-specific channels
        self.readout_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes[layer_n],
            stride=strides[layer_n],
            padding=(kernel_sizes[layer_n] - 1) // 2,
        )

        # --------------------------------------------------
        # Global Average Pooling
        # --------------------------------------------------
        # Reduces spatial dimensions (H x W → 1 x 1)
        # Produces one value per channel
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --------------------------------------------------
        # Normalization for top-down signal
        # --------------------------------------------------
        # Used before sending feedback to earlier layers
        self.layer_norm_td = nn.GroupNorm(
            num_groups=1,
            num_channels=out_channels
        )


    def forward(self, bu_input):
        """
        Forward pass for the Readout layer.

        Parameters:
        -----------
        bu_input : torch.Tensor
            Bottom-up input tensor.

        Returns:
        --------
        output : torch.Tensor
            Class scores for classification.
        td_output : torch.Tensor
            Top-down output tensor.

        Why this is needed:
        -------------------
        This function converts spatial feature maps into classification
        logits while also producing a normalized feature map that can
        act as top-down feedback. The classification output is used for
        loss computation, while the td_output participates in recurrent
        refinement of earlier visual areas.
        """

        # --------------------------------------------------
        # Apply final convolution
        # --------------------------------------------------
        # Shape: [B, in_channels, H, W] → [B, out_channels, H, W]
        output_intermediate = self.readout_conv(bu_input)

        # --------------------------------------------------
        # Global average pooling
        # --------------------------------------------------
        # Collapse spatial dimensions to 1x1
        # Result shape: [B, out_channels]
        output_pooled = self.global_avg_pool(output_intermediate).view(
            output_intermediate.size(0), -1
        )

        # --------------------------------------------------
        # Extract only class logits
        # --------------------------------------------------
        # The first num_classes channels are class scores.
        # Extra channels (add_feats) are NOT used for classification.
        output = output_pooled[:, : self.num_classes]

        # --------------------------------------------------
        # Prepare top-down feedback signal
        # --------------------------------------------------
        # Apply ReLU and normalization to intermediate feature maps
        # This becomes the feedback signal sent into the network
        td_output = self.layer_norm_td(F.relu(output_intermediate))

        return output, td_output


class NoOpModule(nn.Module):
    """
    A no-operation module that returns zero regardless of the input.

    This is used in places where an operation is conditionally skipped.

    Why this is needed:
    -------------------
    Many parts of the BLT_VS architecture (lateral, top-down, skip connections)
    are optional. Instead of writing complex conditional logic inside the
    forward pass, this module acts as a placeholder that safely returns 0.
    This allows the rest of the computation (e.g., additions) to remain clean
    and uniform without checking whether a connection exists.
    """

    def __init__(self):
        super(NoOpModule, self).__init__()

    def forward(self, *args, **kwargs):
        """
        Forward pass that returns zero.

        Returns:
        --------
        Zero tensor or zero value as appropriate.
        """
        # Always return 0 so that adding this to tensors
        # does not change the result
        return 0

    
def concat_or_not(bu_activation, td_activation, dim=1):
    """
    Concatenates bottom-up and top-down activations safely.

    Parameters:
    -----------
    bu_activation : torch.Tensor or None
        Bottom-up activation tensor.
    td_activation : torch.Tensor or None
        Top-down activation tensor.
    dim : int
        Dimension along which concatenation should happen (default: channel dimension).

    Returns:
    --------
    torch.Tensor or None
        Concatenated tensor if at least one activation exists.
        Returns None if both inputs are None.

    Why this is needed:
    -------------------
    During recurrent computation, bottom-up and top-down activations may
    be missing (None). This function ensures that concatenation is robust
    by automatically replacing missing activations with zero tensors of
    matching shape. This prevents runtime errors and keeps activation
    extraction consistent.
    """

    # If both activations are None → nothing to concatenate
    if bu_activation is None and td_activation is None:
        return None
    
    # If bottom-up is missing, create zeros matching TD shape
    if bu_activation is None:
        bu_activation = torch.zeros_like(td_activation)
    
    # If top-down is missing, create zeros matching BU shape
    if td_activation is None:
        td_activation = torch.zeros_like(bu_activation)
    
    # Concatenate along specified dimension (usually channel dim=1)
    return torch.cat([bu_activation, td_activation], dim=dim)