import sys
import pytest
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trading_engine.risk import RiskConfig


def pytest_configure(config):
    """Add custom markers for Phase 7 tests."""
    config.addinivalue_line(
        "markers", "phase7_asset: mark test to use CSV_BACKTEST asset in RiskConfig"
    )


@pytest.fixture
def phase7_risk_config(monkeypatch):
    """
    Fixture that patches RiskConfig.default to allow CSV_BACKTEST asset
    and set a permissive max_notional_per_order for Phase 7 tests.
    """
    from trading_engine.risk import RiskConfig
    import dataclasses

    # Get the original classmethod function
    original_default_func = RiskConfig.default.__func__

    def patched_default(cls):
        # Call the original classmethod function directly to avoid recursion
        cfg = original_default_func(cls)
        # Modify the config: add CSV_BACKTEST to allowed_assets and increase max_notional_per_order
        new_allowed = cfg.allowed_assets + ("CSV_BACKTEST",)
        return dataclasses.replace(
            cfg,
            allowed_assets=new_allowed,
            max_notional_per_order=100000.0,
        )

    # Patch the classmethod descriptor on the class
    monkeypatch.setattr(
        RiskConfig,
        "default",
        classmethod(patched_default),
    )
    try:
        yield
    finally:
        # Restore original classmethod
        monkeypatch.setattr(RiskConfig, "default", classmethod(original_default_func))


def pytest_configure(config):
    """Add custom markers for Phase 7 tests."""
    config.addinivalue_line(
        "markers", "phase7_asset: mark test to use CSV_BACKTEST asset in RiskConfig"
    )


# Auto-use the fixture for tests marked with phase7_asset
@pytest.fixture(autouse=True)
def _auto_use_phase7_fixture(request, phase7_risk_config):
    if request.node.get_closest_marker("phase7_asset"):
        # Fixture is activated by the mark
        pass