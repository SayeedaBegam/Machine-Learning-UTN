import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Loads the data from disk to ram

    Args:
        path (str): File path of the file

    Returns:
        pd.DataFrame: Dataframe containing the data
    """
    ############################################################################
    # YOUR CODE @T
    # TODO: Load the CSV data into a pandas dataframe @T
    ############################################################################
    pass  
    ############################################################################
    # END OF YOUR CODE @T
    ############################################################################

def compute_summary(data: pd.DataFrame) -> dict:
    """Computes the value counts, the mean and the standard deviation, grouped
     by label, and returns them in a dictionary
    
    Args:
        data (pd.DataFrame): The data on which to comput the summary statistics.
    Returns:
        dict: A dictionary with keys 'cnt', 'avg', 'std' containing the value
        counts, the value mean and the standard deviation respectively, all
        grouped by label."""
    ############################################################################
    # YOUR CODE @T
    # TODO: Compute summary statistics. @T
    ############################################################################
    pass  
    ############################################################################
    # END OF YOUR CODE @T
    ############################################################################