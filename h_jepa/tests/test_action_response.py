import pytest
import torch

from action_response import ActionResponseNet, ResponseConfig, response_loss


def test_zero_effect_for_identical_actions():
    cfg = ResponseConfig(state_dim=192, action_dim=2, horizon=5, hidden_dim=64)
    model = ActionResponseNet(cfg).eval()
    z = torch.randn(4, 192)
    a = torch.randn(4, 5, 2)
    torch.testing.assert_close(
        model(z, a, a), torch.zeros(4, 192), atol=0, rtol=0
    )


def test_antisymmetry_and_shape():
    cfg = ResponseConfig(state_dim=192, action_dim=2, horizon=5, hidden_dim=64)
    model = ActionResponseNet(cfg).eval()
    z = torch.randn(3, 192)
    a = torch.randn(3, 5, 2)
    b = torch.randn(3, 5, 2)
    r_ab = model(z, a, b)
    r_ba = model(z, b, a)
    assert r_ab.shape == (3, 192)
    torch.testing.assert_close(r_ab, -r_ba, atol=0, rtol=0)


def test_wrong_action_time_shape_fails_fast():
    model = ActionResponseNet(ResponseConfig())
    z = torch.zeros(2, 192)
    wrong = torch.zeros(2, 2, 5)
    correct = torch.zeros(2, 5, 2)
    with pytest.raises(ValueError):
        model(z, wrong, correct)


def test_gradient_and_tiny_overfit():
    torch.manual_seed(7)
    cfg = ResponseConfig(state_dim=12, action_dim=2, horizon=5, hidden_dim=48)
    model = ActionResponseNet(cfg)
    z = torch.randn(16, 12)
    a, b = torch.randn(16, 5, 2), torch.randn(16, 5, 2)
    weight = torch.randn(10, 12) / 4
    target = (a.reshape(16, -1) - b.reshape(16, -1)) @ weight

    optim = torch.optim.Adam(model.parameters(), lr=0.008)
    with torch.no_grad():
        before = response_loss(model(z, a, b), target).item()
    for _ in range(45):
        optim.zero_grad(set_to_none=True)
        loss = response_loss(model(z, a, b), target)
        loss.backward()
        optim.step()
    with torch.no_grad():
        after = response_loss(model(z, a, b), target).item()
    assert after < before * 0.5, (before, after)
    assert all(torch.isfinite(p).all() for p in model.parameters())
