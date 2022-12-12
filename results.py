# %%
import pandas as pd

results = pd.read_csv("benchmark.csv")

# %%
results.head()
# %%
results.sort_values(by="avg_test_losses")
# %%
