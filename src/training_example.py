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


    tgt = src[-1]
    print(f"src: {src.shape}, tgt: {tgt.shape}")

    # Iteratively concatenate tgt with the first element in the prediction
    for _ in range(forecast_window - 1):

        # Create masks
        dim1 = tgt.shape[0]

        dim2 = src.shape[1]

        tgt_mask = utils.generate_square_subsequent_mask(
            dim1=dim1,
            dim2=dim1
        )

        src_mask = utils.generate_square_subsequent_mask(
            dim1=dim1,
            dim2=dim2
        )

        # Make prediction
        prediction = model(src, tgt, src_mask, tgt_mask)

        # If statement simply makes sure that the predicted value is
        # extracted and reshaped correctly


        # Obtain predicted value
        last_predicted_value = prediction[:, -1, :]

        # Reshape from [batch_size, 1] --> [batch_size, 1, 1]
        last_predicted_value = last_predicted_value.unsqueeze(-1)

        # Detach the predicted element from the graph and concatenate with
        # tgt in dimension 1 or 0
        tgt = torch.cat((tgt, last_predicted_value.detach()), 1)

    # Create masks
    dim1 = tgt.shape[1]

    dim2 = src.shape[1]

    tgt_mask = utils.generate_square_subsequent_mask(
        dim1=dim1,
        dim2=dim1
    )

    src_mask = utils.generate_square_subsequent_mask(
        dim1=dim1,
        dim2=dim2
    )

    # Make final prediction
    final_prediction = model(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)

    return final_prediction