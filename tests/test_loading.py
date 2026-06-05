import mojito.download
import pytest

import typed_lisa_toolkit as tlt


# Keep this mark until caching is configured in CI
@pytest.mark.slow("requires downloading data")
def test_load_mojito():
    combined = mojito.download.download_brick("combined", reduced=False)
    # Reduced data has no quality flag so we must use the non-reduced version.
    data = tlt.load_mojito(combined, time_interval=(None, tlt.shop.month2second(6)))
    assert data.channel_names == ("X", "Y", "Z", "flag")


# Keep this mark until caching is configured in CI
@pytest.mark.slow("requires downloading data")
def test_load_mojito_orbits():
    combined = mojito.download.download_brick("combined", reduced=False)
    orbits = tlt.load_mojito_orbits(
        combined, time_interval=(None, tlt.shop.month2second(6))
    )
    assert orbits.channel_names == (
        "SAT1_POS1",
        "SAT1_POS2",
        "SAT1_POS3",
        "SAT2_POS1",
        "SAT2_POS2",
        "SAT2_POS3",
        "SAT3_POS1",
        "SAT3_POS2",
        "SAT3_POS3",
        "SAT1_VEL1",
        "SAT1_VEL2",
        "SAT1_VEL3",
        "SAT2_VEL1",
        "SAT2_VEL2",
        "SAT2_VEL3",
        "SAT3_VEL1",
        "SAT3_VEL2",
        "SAT3_VEL3",
    )
