# Parameter Efficient Transfer Learning
Inspired by a paper ["Parameter-Efficient Transfer Learning for NLP"](https://arxiv.org/abs/1902.00751). It's an attempt to apply the described approach for a transformer-based pretrained model.



In my example, I take HuggingFace implementation of a Roberta model and do monkey patch.

```python
for t_layer in model.encoder.layer:
    # inject adapter twice for each layer
    inject_adapter(t_layer.attention.output, bottleneck_size=1)
    inject_adapter(t_layer.output, bottleneck_size=1)
```

For more details see `main.py`
