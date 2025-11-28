import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

from pyseis_io.gpkg import export_headers_to_gis
from pyseis_io.models import SeismicData

@pytest.fixture
def mock_seismic_data():
    headers = pd.DataFrame({
        'trace_num': range(10),
        'source_coordinate_x': np.arange(10) * 100.0,
        'source_coordinate_y': np.arange(10) * 100.0,
        'elevation': np.random.random(10) * 50
    })
    # Mock SeismicData since we don't want to depend on full implementation details
    sd = MagicMock(spec=SeismicData)
    sd.headers = headers
    return sd, headers

def test_export_headers_to_gis_success(mock_seismic_data, tmp_path):
    sd, headers = mock_seismic_data
    output_path = tmp_path / "test.gpkg"
    
    # Mock geopandas and shapely to verify logic without requiring installation
    with patch.dict(sys.modules, {'geopandas': MagicMock(), 'shapely.geometry': MagicMock()}):
        mock_gpd = sys.modules['geopandas']
        mock_shapely = sys.modules['shapely.geometry']
        
        # Mock GeoDataFrame constructor and to_file
        mock_gdf_instance = MagicMock()
        mock_gpd.GeoDataFrame.return_value = mock_gdf_instance
        
        export_headers_to_gis(sd, str(output_path))
        
        # Verify GeoDataFrame was created
        assert mock_gpd.GeoDataFrame.called
        _, kwargs = mock_gpd.GeoDataFrame.call_args
        assert kwargs['crs'] == "EPSG:4326"
        assert 'geometry' in kwargs
        
        # Verify to_file was called
        mock_gdf_instance.to_file.assert_called_with(str(output_path), driver="GPKG")

def test_missing_dependency():
    # Simulate missing geopandas
    with patch.dict(sys.modules, {'geopandas': None}):
        with pytest.raises(ImportError, match="geopandas is required"):
            export_headers_to_gis(pd.DataFrame(), "out.gpkg")

def test_invalid_geometry_columns(mock_seismic_data):
    sd, _ = mock_seismic_data
    # We need geopandas to be "present" to pass the import check
    with patch.dict(sys.modules, {'geopandas': MagicMock(), 'shapely.geometry': MagicMock()}):
        with pytest.raises(ValueError, match="must contain exactly two column names"):
            export_headers_to_gis(sd, "out.gpkg", geometry_cols=["x"])

def test_missing_columns(mock_seismic_data):
    sd, _ = mock_seismic_data
    with patch.dict(sys.modules, {'geopandas': MagicMock(), 'shapely.geometry': MagicMock()}):
        with pytest.raises(ValueError, match="Geometry columns .* not found"):
            export_headers_to_gis(sd, "out.gpkg", geometry_cols=["missing_x", "missing_y"])
