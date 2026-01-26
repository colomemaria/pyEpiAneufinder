import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from pyEpiAneufinder.plotting import karyo_gainloss

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        'seq': ['chr1', 'chr1', 'chr2', 'chr2'],
        'start': [0, 1000000, 0, 1000000],
        'end': [1000000, 2000000, 1000000, 2000000],
        'cell1': [2, 1, 0, 1],
        'cell2': [2, 1, 1, 0],
        'cell3': [1, 1, 1, 1],
        'cell4': [0, 2, 2, 1],
    }
    return pd.DataFrame(data)

#@patch templates for mocking a real function or object during a test
#avoids side effects and allows verification of interactions
@patch("matplotlib.pyplot.savefig")
@patch("pyEpiAneufinder.plotting.dendrogram")

def test_karyo_gainloss(mock_dendrogram, mock_savefig, sample_dataframe, tmp_path):
    """Test the karyo_gainloss function with a sample DataFrame."""
    # Call the function with the sample DataFrame
    mock_dendrogram.return_value = {"leaves": [0, 1, 2]}
    outdir = str(tmp_path)+"/"
    karyo_gainloss(sample_dataframe, outdir, "Title")

    # Verify that the savefig function was called with the correct filename
    mock_savefig.assert_called_once()
    assert mock_savefig.call_args[0][0].endswith("Karyogram.png")

    # Verify that the dendrogram function was called
    mock_dendrogram.assert_called() 

def test_karyo_gainloss_empty_dataframe(tmp_path):
    """Test the karyo_gainloss function with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=['seq', 'start', 'end', 'cell1', 'cell2'])

    outdir = str(tmp_path)+"/"
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        karyo_gainloss(empty_df, outdir, "Title")

def test_karyo_gainloss_invalid_dataframe(tmp_path):
    df_invalid = pd.DataFrame({
        "seq": ["chr1", "chr2"],
        "start": [0, 0],
        "end": [1000000000, 100000],
        "cell1": ["a", "b"],  # invalid string values
        "cell2": ["x", "y"]
    })
    outdir = str(tmp_path) + "/"

    with pytest.raises(ValueError, match="All CNV columns must be numeric"):
        karyo_gainloss(df_invalid, outdir, "Invalid Test")


def test_karyo_gainloss_no_gainloss(tmp_path):
    df_no_gainloss = pd.DataFrame({
        "seq": ["chr1", "chr2"],
        "start": [1, 1],
        "end": [10, 10],
        "cell1": [1, 1],
        "cell2": [1, 1]
    })
    outdir = str(tmp_path) + "/"

    with pytest.raises(ValueError, match="No gain or loss detected"):
        karyo_gainloss(df_no_gainloss, outdir, "No Gain/Loss Test")

def test_column_reordering_after_clustering(tmp_path):
    df = pd.DataFrame({
        "seq": ["chr1"]*4,
        "start": [1,2,3,4],
        "end": [5,6,7,8],
        "cell1":[0,1,0,1],
        "cell2":[1,0,1,0],
        "cell3":[0,1,1,0],
        "cell4":[1,0,0,1]
    })
    outdir = str(tmp_path)+"/"

    with patch("pyEpiAneufinder.plotting.dendrogram") as mock_dendrogram, patch("matplotlib.pyplot.savefig") as mock_savefig:
        mock_dendrogram.return_value = {'leaves':[0,1,2,3]}
        result = karyo_gainloss(df, outdir, "Test")
        mock_savefig.assert_called_once()
        expected_order = ["seq","start","end","cell4","cell3","cell2","cell1"]
        assert list(result.columns) == expected_order