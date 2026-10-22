import torch

from vcfdconformer import VariableContextFDConformer


def test_variable_context_and_query_shapes():
    model = VariableContextFDConformer(n_freq=106, d_model=32, d_ff=64, n_heads=4, n_layers=2)
    hrtf = torch.randn(2, 7, 2, 106)
    context_dirs = torch.nn.functional.normalize(torch.randn(2, 7, 3), dim=-1)
    query_dirs = torch.nn.functional.normalize(torch.randn(2, 11, 3), dim=-1)
    mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])

    output = model(hrtf, context_dirs, mask, query_dirs)

    assert output.shape == (2, 11, 2, 106)
    assert torch.isfinite(output).all()
