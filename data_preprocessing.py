import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

class DataPreprocessing:
    """
    A class used to preprocess input and output data for a sequence-to-sequence model.
    
    Attributes:
    ----------
    input_embedding : nn.Embedding
        The embedding layer for the input data.
    output_embedding : nn.Embedding
        The embedding layer for the output data.
    input_dim : int
        The dimension of the input data.
    output_dim : int
        The dimension of the output data.
    d_model : int
        The dimension of the embedding vectors.
    """

    def __init__(self, input_dim: int, output_dim: int, d_model: int):
        """
        Initializes the DataPreprocessing class.
        
        Parameters:
        ----------
        input_dim : int
            The dimension of the input data.
        output_dim : int
            The dimension of the output data.
        d_model : int
            The dimension of the embedding vectors.
        """
        self.input_embedding = nn.Embedding(input_dim, d_model)
        self.output_embedding = nn.Embedding(output_dim, d_model)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

    def preprocess(self, input_data: List[List[int]], output_data: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocesses the input and output data by tokenizing the sequences and converting them to indices.
        
        Parameters:
        ----------
        input_data : List[List[int]]
            The input data.
        output_data : List[List[int]]
            The output data.
        
        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The preprocessed input and output data.
        """
        input_sequences = [torch.tensor(seq) for seq in input_data]
        output_sequences = [torch.tensor(seq) for seq in output_data]
        input_padded = pad_sequence(input_sequences, batch_first=True, padding_value=0)
        output_padded = pad_sequence(output_sequences, batch_first=True, padding_value=0)
        return input_padded, output_padded

    def embed(self, input_data: torch.Tensor, output_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds the input and output data using the input and output embeddings.
        
        Parameters:
        ----------
        input_data : torch.Tensor
            The input data.
        output_data : torch.Tensor
            The output data.
        
        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The embedded input and output data.
        """
        input_embedded = self.input_embedding(input_data)
        output_embedded = self.output_embedding(output_data)
        return input_embedded, output_embedded


class TokenizedTextDataset(Dataset):
    """
    A class used to create a PyTorch dataset from tokenized text sequences.
    
    Attributes:
    ----------
    sequences : List[torch.Tensor]
        The tokenized text sequences.
    """

    def __init__(self, sequences: List[List[int]]):
        """
        Initializes the TokenizedTextDataset class.
        
        Parameters:
        ----------
        sequences : List[List[int]]
            The tokenized text sequences.
        """
        self.sequences = [torch.tensor(seq) for seq in sequences]

    def __len__(self) -> int:
        """
        Returns the number of sequences in the dataset.
        
        Returns:
        -------
        int
            The number of sequences in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns the sequence at the specified index.
        
        Parameters:
        ----------
        idx : int
            The index of the sequence.
        
        Returns:
        -------
        torch.Tensor
            The sequence at the specified index.
        """
        return self.sequences[idx]


def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Pads the sequences in the batch to the same length.
    
    Parameters:
    ----------
    batch : List[torch.Tensor]
        The batch of sequences.
    
    Returns:
    -------
    torch.Tensor
        The padded batch of sequences.
    """
    return pad_sequence(batch, batch_first=True, padding_value=0)


if __name__ == "__main__":
    input_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    output_data = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
    data_preprocessing = DataPreprocessing(20, 20, 128)
    input_padded, output_padded = data_preprocessing.preprocess(input_data, output_data)
    input_embedded, output_embedded = data_preprocessing.embed(input_padded, output_padded)
    print(input_embedded.shape)
    print(output_embedded.shape)
    dataset = TokenizedTextDataset(input_data)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch.shape)