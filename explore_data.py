from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from data.splits import CzechLynxSplitter



DATA_ROOT = Path("data/kaggle-data")
CSV_PATH = DATA_ROOT / "CzechLynxDataset-Metadata-Real.csv"
splitter = CzechLynxSplitter(DATA_ROOT, CSV_PATH)
df = splitter.df

# 1. Rozkład liczby zdjęć na osobnika
counts = df.groupby("identity").size().sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
axes[0].hist(counts.values, bins=40, color="#534AB7", edgecolor="white")
axes[0].set_title("Rozkład zdjęć na osobnika")
axes[0].set_xlabel("Liczba zdjęć")
axes[0].set_ylabel("Liczba osobników")
axes[0].axvline(counts.median(), color="red", linestyle="--", label=f"mediana={counts.median():.0f}")
axes[0].legend()

# 2. Rozkład po regionach
region_counts = df["source"].value_counts()
axes[1].bar(region_counts.index, region_counts.values, color=["#1D9E75","#534AB7","#D85A30"])
axes[1].set_title("Zdjęcia per region")
axes[1].set_ylabel("Liczba zdjęć")

# 3. Top 20 osobników
top20 = counts.head(20)
axes[2].barh(top20.index[::-1], top20.values[::-1], color="#1D9E75")
axes[2].set_title("Top 20 osobników (najwięcej zdjęć)")
axes[2].set_xlabel("Liczba zdjęć")

plt.tight_layout()
plt.savefig("eda_distribution.png", dpi=150)
plt.show()
print(f"Min zdjęć na osobnika: {counts.min()}, Max: {counts.max()}, Mediana: {counts.median()}")

# 4. Przykładowe zdjęcia — 8 losowych
sample = df.sample(8, random_state=42)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, (_, row) in zip(axes.flat, sample.iterrows()):
    img = mpimg.imread(DATA_ROOT / row["path"])
    ax.imshow(img)
    ax.set_title(f"{row['identity']}\n{row['source']}", fontsize=8)
    ax.axis("off")
plt.suptitle("Przykładowe zdjęcia z datasetu", fontsize=13)
plt.tight_layout()
plt.savefig("eda_samples.png", dpi=150)
plt.show()

print(f"Osobników z < 10 zdjęć:  {(counts < 10).sum()}")
print(f"Osobników z < 20 zdjęć:  {(counts < 20).sum()}")
print(f"Osobników z > 200 zdjęć: {(counts > 200).sum()}")
print(f"Osobników z > 500 zdjęć: {(counts > 500).sum()}")
print(f"\nTop 5 osobników:\n{counts.head()}")

#Rozkład splitów czasowych
for split in ["time_open", "time_closed"]:
    train_df = splitter.get_predefined(split, "train")
    test_df  = splitter.get_predefined(split, "test")

    train_ids = set(train_df["identity"])
    test_ids  = set(test_df["identity"])

    overlap = train_ids & test_ids

    print(f"\n{split}:")
    print(f"  train: {len(train_df)} zdjęć, {len(train_ids)} osobników")
    print(f"  test:  {len(test_df)} zdjęć, {len(test_ids)} osobników")
    print(f"  overlap osobników: {len(overlap)}")

splitter.analyze("geo_aware")