from typing import Union
import torch
from transformers.models.roberta.modeling_roberta import RobertaOutput, RobertaSelfOutput

class Adapter(torch.nn.Module):
    def __init__(self, input_size: int, bottleneck_size: int):
        super().__init__()
        self._model = torch.nn.Sequential(
            torch.nn.Linear(input_size, bottleneck_size),
            torch.nn.GELU(),
            torch.nn.Linear(bottleneck_size, input_size),
        )
  
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self._model(input_tensor) + input_tensor


def roberta_output_forward_patched(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    ''' Patched version of a forward function'''
    hidden_states = self.dense(hidden_states)
    hidden_states = self.adapter(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states

def inject_adapter(module: Union[RobertaOutput, RobertaSelfOutput], bottleneck_size: int) -> None:
    '''
        Do monkey patch. We use knoweledge about Roberta implementations:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
    '''
    hidden_size = module.dense.out_features
    module.adapter = Adapter(hidden_size, bottleneck_size)
    module.forward = roberta_output_forward_patched.__get__(module, type(module))
