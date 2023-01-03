import torch
from adapter_for_roberta import inject_adapter
from transformers import RobertaModel, RobertaConfig
from parameters import disable_parameters, print_params_stat


config = RobertaConfig.from_pretrained('microsoft/unixcoder-base')
model = RobertaModel.from_pretrained('microsoft/unixcoder-base', config=config)

disable_parameters(model)
print_params_stat(model)

for t_layer in model.encoder.layer:
    # inject adapter twice for each layer
    inject_adapter(t_layer.attention.output, bottleneck_size=1)
    inject_adapter(t_layer.output, bottleneck_size=1)

    # to enable grad for normalization layer, as said in the paper 
    t_layer.attention.output.LayerNorm.requires_grad_(True)
    t_layer.output.LayerNorm.requires_grad_(True)

print_params_stat(model)
output = model(torch.tensor([[1]]))
