from torchinfo import summary


def get_model_summary(model, input_shape):
    return summary(model, input_size=(input_shape[0], input_shape[1]))