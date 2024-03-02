
import pandas as pd
def load_data(file):
    data = pd.read_csv(file)
    return data