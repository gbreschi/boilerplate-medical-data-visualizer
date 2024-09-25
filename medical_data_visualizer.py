import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data from medical_examination.csv
df = pd.read_csv('medical_examination.csv')

# 2. Create 'overweight' column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3. Normalize cholesterol and gluc columns (0 = normal, 1 = above normal)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Draw the Categorical Plot in the draw_cat_plot function
def draw_cat_plot():
    # 5. Create a DataFrame for the cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6. Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()

    # Rename 'size' column to 'total' as expected by the test
    df_cat = df_cat.rename(columns={'size': 'total'})

    # 7. Convert the data into long format and create a categorical plot
    sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')

    # 8. Get the figure for the output and store it in the fig variable
    fig = plt.gcf()

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# 9. Draw the Heat Map in the draw_heat_map function
def draw_heat_map():
    # 10. Clean the data in the df_heat variable
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 11. Calculate the correlation matrix and store it in the corr variable
    corr = df_heat.corr()

    # 12. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13. Set up the matplotlib figure
    plt.figure(figsize=(12, 8))

    # 14. Plot the heatmap
    sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, square=True, linewidths=1, cbar_kws={'shrink': 0.5})

    # 15. Get the figure for the output and store it in the fig variable
    fig = plt.gcf()

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig


