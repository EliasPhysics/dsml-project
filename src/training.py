import torch.nn as nn
import torch
import utils


def run_encoder_decoder_inference(
        model: nn.Module,
        src: torch.Tensor,
        forecast_window: int,
        batch_size: int,
        device
) -> torch.Tensor:
    """
        bla
    """

    # Take the last value of thetarget variable in all batches in src and make it tgt
    # as per the Influenza paper
    tgt = src[:, -1, 0]  # shape [1, batch_size, 1]

    # Iteratively concatenate tgt with the first element in the prediction
    for _ in range(forecast_window - 1):

        # Create masks
        dim1 = tgt.shape[1]

        dim2 = src.shape[1]

        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=dim1,
            dim2=dim1,
            device=device
        )

        src_mask = utils.generate_square_subsequent_mask(
            dim1=dim1,
            dim2=dim2,
            device=device
        )

        # Make prediction
        prediction = model(src, tgt, src_mask, tgt_mask)

        # If statement simply makes sure that the predicted value is
        # extracted and reshaped correctly
        if batch_first == False:

            # Obtain the predicted value at t+1 where t is the last time step
            # represented in tgt
            last_predicted_value = prediction[-1, :, :]

            # Reshape from [batch_size, 1] --> [1, batch_size, 1]
            last_predicted_value = last_predicted_value.unsqueeze(0)

        else:

            # Obtain predicted value
            last_predicted_value = prediction[:, -1, :]

            # Reshape from [batch_size, 1] --> [batch_size, 1, 1]
            last_predicted_value = last_predicted_value.unsqueeze(-1)

        # Detach the predicted element from the graph and concatenate with
        # tgt in dimension 1 or 0
        tgt = torch.cat((tgt, last_predicted_value.detach()), 1)

    # Create masks
    dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

    dim_b = src.shape[1] if batch_first == True else src.shape[0]

    tgt_mask = utils.generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_a,
        device=device
    )

    src_mask = utils.generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_b,
        device=device
    )

    # Make final prediction
    final_prediction = model(src, tgt, src_mask, tgt_mask)

    return final_prediction