import torch

def predict(model, data_loader):
    model.eval()

    outputs = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            output = model(inputs)
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)  # Concatenate all outputs
    return outputs