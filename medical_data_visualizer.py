import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv("medical_examination.csv")

# 2. Add 'overweight' column (BMI > 25)
df['overweight'] = (
    df['weight'] / ((df['height'] / 100) ** 2) > 25
).astype(int)

# 3. Normalize cholesterol and gluc (0 = good, 1 = bad)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# =========================
# CATEGORICAL PLOT
# =========================
def draw_cat_plot():
    # Create melted DataFrame
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # Group and count
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )

    # Draw plot
    g = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    )

    g.set_axis_labels("variable", "total")
    g.set_titles("cardio = {col_name}")

    fig = g.fig
    return fig


# =========================
# HEAT MAP
# =========================
def draw_heat_map():
    # Clean data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Correlation matrix
    corr = df_heat.corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set figure size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    return fig