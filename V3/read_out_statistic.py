import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

additive_df = pd.DataFrame(pd.read_csv('./V3/processed_data/additives_order.csv'))
additives = additive_df['additives'].tolist()
print((set(additives)))
