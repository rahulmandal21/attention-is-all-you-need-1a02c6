import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Tuple

class BLEUEvaluator:
    """
    Evaluates the performance of a machine translation model using the BLEU score.
    """

    def __init__(self):
        """
        Initializes the BLEUEvaluator.
        """
        pass

    def calculate_bleu_score(self, predictions: List[List[str]], references: List[List[List[str]]]) -> float:
        """
        Calculates the BLEU score for the given predictions and references.

        Args:
            predictions (List[List[str]]): The predicted translations.
            references (List[List[List[str]]]): The reference translations.

        Returns:
            float: The BLEU score.
        """
        bleu_scores = []
        for prediction, reference in zip(predictions, references):
            bleu_score = sentence_bleu(reference, prediction, smoothing_function=SmoothingFunction().method4)
            bleu_scores.append(bleu_score)
        return sum(bleu_scores) / len(bleu_scores)

    def evaluate_model(self, model: nn.Module, test_data: List[Tuple[List[str], List[List[str]]]]) -> float:
        """
        Evaluates the performance of the given model on the test data.

        Args:
            model (nn.Module): The model to evaluate.
            test_data (List[Tuple[List[str], List[List[str]]]]): The test data.

        Returns:
            float: The BLEU score.
        """
        predictions = []
        references = []
        for input_sequence, reference in test_data:
            # Assuming the model has a forward method that takes in a list of strings and returns a list of strings
            prediction = model(input_sequence)
            predictions.append(prediction)
            references.append(reference)
        return self.calculate_bleu_score(predictions, references)


if __name__ == "__main__":
    # Create a dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_sequence: List[str]) -> List[str]:
            # For demonstration purposes, just return the input sequence
            return input_sequence

    # Create a BLEUEvaluator
    evaluator = BLEUEvaluator()

    # Create some dummy test data
    test_data = [
        (["hello", "world"], [["hello", "world"]]),
        (["foo", "bar"], [["foo", "bar"]]),
    ]

    # Create a dummy model
    model = DummyModel()

    # Evaluate the model
    bleu_score = evaluator.evaluate_model(model, test_data)
    print("BLEU score:", bleu_score)