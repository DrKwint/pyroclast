import seaborn as sns
sns.set(context="notebook")
import matplotlib.pyplot as plt
import pandas as pd
import json

with open('metrics.json', 'r') as json_file:
    metrics_dict = json.load(json_file)
metrics_dict = eval(metrics_dict)
dfs = {
    k: pd.DataFrame.from_dict(v, orient='index')
    for k, v in metrics_dict.items()
}

for i, df in dfs.items():
    print(df)
    mdf = pd.melt(
        df.reset_index(),
        id_vars=['index'],
        value_vars=['hypothesis accuracy', 'boosted accuracy'])
    ax = sns.lineplot(x='index', y='value', hue='variable', data=mdf)
    ax.set(
        title='Accuracy after Training Module ' + str(i),
        xlabel='module',
        ylabel='accuracy')
    plt.savefig('accuracy-posttrain_module_' + str(i) + '.png')
    plt.close()
