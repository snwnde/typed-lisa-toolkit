import lisaorbits
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


@pytest.mark.slow("requires downloading data")
def test_load_reduced_mojito():
    combined = mojito.download.download_brick("combined", reduced=True)
    data = tlt.load_mojito(combined, time_interval=(None, tlt.shop.month2second(6)))
    assert data.channel_names == ("X", "Y", "Z")


@pytest.mark.slow("requires downloading data")
def test_load_orbits():
    orbits = tlt.load_mojito_orbits(
        mojito.download.download_brick("combined", reduced=False)
    )
    assert isinstance(orbits, lisaorbits.Orbits)


@pytest.mark.slow("requires downloading data")
def test_load_reduced_orbits():
    orbits = tlt.load_mojito_orbits(
        mojito.download.download_brick("combined", reduced=True)
    )
    assert isinstance(orbits, lisaorbits.Orbits)
