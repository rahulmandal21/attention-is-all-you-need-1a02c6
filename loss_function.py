import torch
import torch.nn as nn

class CrossEntropyLossFunction(nn.Module):
    """
    A class to calculate the cross-entropy loss between the predicted output and the true output.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1) -> None:
        """
        Initializes the CrossEntropyLossFunction class.

        Args:
            num_classes (int): The number of classes in the classification problem.
            smoothing (float, optional): The label smoothing value. Defaults to 0.1.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cross-entropy loss between the predicted output and the true output.

        Args:
            predictions (torch.Tensor): The predicted output.
            targets (torch.Tensor): The true output.

        Returns:
            torch.Tensor: The cross-entropy loss.
        """
        return self.criterion(predictions, targets)


if __name__ == "__main__":
    # Create a dummy model
    model = nn.Linear(5, 3)

    # Create a CrossEntropyLossFunction instance
    loss_function = CrossEntropyLossFunction(num_classes=3)

    # Generate dummy data
    inputs = torch.randn(1, 5)
    labels = torch.tensor([1])

    # Forward pass
    outputs = model(inputs)

    # Calculate the loss
    loss = loss_function(outputs, labels)

    print(f"Loss: {loss.item()}")