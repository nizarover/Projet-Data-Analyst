import numpy as np
import pandas as pd

# Example contingency table (observed frequencies)
data = pd.read_excel("C:\\Users\\exe\\Desktop\\Scripts\\Coding\\Python\\Projects\\Data Analyst\\AFC\\nobel.xlsx")

# Convert to DataFrame for better readability
df = pd.DataFrame(Effectifs)

# Step 1: Compute row totals, column totals, and grand total
row_totals = df.sum(axis=1)
col_totals = df.sum(axis=0)
grand_total = df.sum().sum()

# Step 2: Compute the expected frequencies
expected = np.outer(row_totals, col_totals) / grand_total

# Step 3: Compute the correspondence matrix (O/E)
correspondence_matrix = df.values / expected

# Convert the result back to DataFrame for readability
correspondence_df = pd.DataFrame(correspondence_matrix, columns=df.columns, index=df.index)

print("Correspondence Matrix:")
print(correspondence_df)