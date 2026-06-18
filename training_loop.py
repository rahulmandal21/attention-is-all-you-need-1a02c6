import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

class TrainingLoop:
    """
    A class used to manage the training loop of a PyTorch model.

    Attributes:
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    dataloader : torch.utils.data.DataLoader
        The data loader for the training data.
    loss_fn : nn.Module
        The loss function to be used for training.
    device : torch.device
        The device on which the model will be trained.
    d_model : int
        The dimensionality of the model.
    warmup_steps : int
        The number of warmup steps for the learning rate schedule.
    max_grad_norm : float
        The maximum gradient norm for gradient clipping.

    Methods:
    -------
    train_one_epoch()
        Trains the model for one epoch.
    get_lr()
        Gets the current learning rate based on the step number and warmup steps.
    """

    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, device: torch.device, d_model: int, warmup_steps: int, max_grad_norm: float = 1.0):
        """
        Initializes the TrainingLoop class.

        Args:
        ----
        model (nn.Module): The PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): The data loader for the training data.
        loss_fn (nn.Module): The loss function to be used for training.
        device (torch.device): The device on which the model will be trained.
        d_model (int): The dimensionality of the model.
        warmup_steps (int): The number of warmup steps for the learning rate schedule.
        max_grad_norm (float, optional): The maximum gradient norm for gradient clipping. Defaults to 1.0.
        """
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = LambdaLR(self.optimizer, self.get_lr)

    def train_one_epoch(self) -> float:
        """
        Trains the model for one epoch.

        Returns:
        -------
        float
            The average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in self.dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(self.dataloader)

    def get_lr(self, step_num: int) -> float:
        """
        Gets the current learning rate based on the step number and warmup steps.

        Args:
        ----
        step_num (int): The current step number.

        Returns:
        -------
        float
            The current learning rate.
        """
        return self.d_model ** -0.5 * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)


if __name__ == "__main__":
    # Create a dummy model, data loader, and loss function
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel()
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 10)), batch_size=10)
    loss_fn = nn.MSELoss()

    # Create a TrainingLoop instance and train the model for one epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loop = TrainingLoop(model, dataloader, loss_fn, device, d_model=10, warmup_steps=100)
    loss = training_loop.train_one_epoch()
    print(f"Loss: {loss}")