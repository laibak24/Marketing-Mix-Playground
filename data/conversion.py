import pyreadr

result = pyreadr.read_r("raw/dt_simulated_weekly.RData")
df = list(result.values())[0]

df.to_csv("raw/weekly_media_data.csv", index=False)

print(df.head())
print(df.columns.tolist())