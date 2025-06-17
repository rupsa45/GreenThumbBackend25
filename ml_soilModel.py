import pandas as pd

# Load and preprocess the dataset
file_path = "SoiltypesandNPKofStates.csv" 
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()  # Strip any whitespace from column names
data['Percentage'] = data['Percentage'].str.replace('%', '').astype(float) / 100  # Convert percentages to decimal

def get_soil_analysis(state: str):
    """
    Get soil type percentages and NPK values for a given state.
    :param state: Name of the state (case-insensitive)
    :return: A dictionary with soil types and NPK values
    """
    state = state.strip().lower()

    # Filter the data for the given state
    state_data = data[data['States'].str.lower() == state]

    if state_data.empty:
        return {"error": f"No data found for the state '{state}'. Please check the input."}

    # Extract soil type percentages
    soil_types = state_data[['Soil_Types', 'Percentage']].to_dict(orient='records')

    # Extract NPK values (first row if duplicates exist)
    npk_values = state_data[['Nitrogen', 'Phosphorus', 'Potassium']].iloc[0].to_dict()

    return {
        "state": state.title(),
        "soil_types": soil_types,
        "npk_values": npk_values
    }
