import pytest
from colossalai.testing import rerun_if_address_is_in_use
from test_engine.boring_model_utils import run_boring_model


@pytest.mark.dist
@pytest.mark.standalone
@rerun_if_address_is_in_use()
def test_tp():
    run_boring_model(2, 1)


if __name__ == '__main__':
    test_tp()
