import pytest

from oc_gsn.models.coface import CofaceAggregator


def test_coface_invalid_aggr_raises():
    with pytest.raises(ValueError, match="aggr"):
        CofaceAggregator(
            hidden_dim=8,
            inc_dim=1,
            out_dim=8,
            mlp_hidden_dim=8,
            aggr="bad",
        )


def test_coface_valid_aggr_modes_construct():
    CofaceAggregator(hidden_dim=8, inc_dim=1, out_dim=8, mlp_hidden_dim=8, aggr="sum")
    CofaceAggregator(hidden_dim=8, inc_dim=1, out_dim=8, mlp_hidden_dim=8, aggr="mean")
