import pytest
import torch

from oc_gsn.models.ocgsn import OCGSN


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_raw_tensors_are_moved_to_model_device():
    z = torch.tensor([6, 1, 1, 1])
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        requires_grad=True,
    )
    batch = torch.zeros(4, dtype=torch.long)

    model = OCGSN(
        hidden_dim=16,
        num_layers=1,
        rbf_dim=4,
        cutoff=5.0,
        max_dim=3,
    ).cuda()

    y = model(z=z, pos=pos, batch=batch)

    assert y.device.type == "cuda"

    loss = y.sum()
    loss.backward()

    # The original CPU pos may not receive grad if pos is moved via .to(cuda),
    # but the test must not fail for that. Device movement is the main check here.
    assert torch.isfinite(y).all()
