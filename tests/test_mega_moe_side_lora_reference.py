"""CPU numerical regression for the native driver's shared-gradient oracle."""

import ast
from pathlib import Path

import torch
import pytest


def _reference_helpers():
    # Load only the pure-Torch helpers: the GPU driver imports the extension.
    source = Path(__file__).with_name("test_mega_moe_native_side_lora.py")
    tree = ast.parse(source.read_text())
    names = {"_reference_adapters", "_reference_expert_adapters"}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
         namespace)
    return tuple(namespace[name] for name in (
        "_reference_adapters", "_reference_expert_adapters"))


@pytest.mark.parametrize("combined_backward", [False, True])
def test_shared_reference_preserves_bf16_operands_and_fp32_accumulation(combined_backward):
    make_refs, expert_operands = _reference_helpers()
    experts = 257
    shared = (0, 2, 5)
    adapters = tuple(
        torch.full((2, 2) if index in shared else (experts, 2, 2),
                   0.25, dtype=torch.bfloat16)
        for index in range(6))
    refs = make_refs(adapters)
    old_refs = tuple(t.clone().requires_grad_(True) for t in adapters)
    losses, old_losses = [], []
    for expert in range(experts):
        operands = expert_operands(refs, expert)
        old_operands = tuple(t if index in shared else t[expert]
                             for index, t in enumerate(old_refs))
        for actual, expected in zip(operands, old_operands):
            assert actual.dtype == torch.bfloat16
            assert torch.equal(actual, expected)
        # All contributions are exactly representable. After a unit gradient,
        # 1/256 contributions are lost by a repeatedly rounded BF16 leaf.
        contribution = 1.0 if expert == 0 else 1.0 / 256
        loss = sum(t.float().sum() * contribution for t in operands)
        old_loss = sum(t.float().sum() * contribution for t in old_operands)
        if combined_backward:
            losses.append(loss)
            old_losses.append(old_loss)
        else:
            loss.backward()
            old_loss.backward()
    if combined_backward:
        torch.stack(losses).sum().backward()
        torch.stack(old_losses).sum().backward()
    for index in range(6):
        if index in shared:
            assert refs[index].grad.dtype == torch.float32
            assert torch.equal(refs[index].grad, torch.full((2, 2), 2.0))
            # The old oracle's combined-graph accumulation order is engine
            # dependent; the ordered-backward case proves its rounding loss.
            if not combined_backward:
                assert torch.equal(old_refs[index].grad,
                                   torch.ones((2, 2), dtype=torch.bfloat16))
        else:
            assert refs[index].grad.dtype == torch.bfloat16
            assert torch.equal(refs[index].grad, old_refs[index].grad)
        assert adapters[index].grad is None
