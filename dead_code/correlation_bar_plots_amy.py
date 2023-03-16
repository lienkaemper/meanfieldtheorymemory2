import pandas as pd
import seaborn as sb
from matplotlib import rcParams
import matplotlib.pyplot as plt

hue_order = ['Non-tagged vs Non-tagged','Tagged vs Tagged','Tagged vs Non-tagged']
FC = pd.read_csv("data/pairwise_correlations_4binned_FC.csv") # path to csv file
#%% for baseline + post on same graph
rcParams['axes.linewidth']=2
rcParams['font.family']='Arial'
rcParams['font.size']='18'
plt.figure(figsize=(7,5))
g=sb.barplot(data = FC,x='Session',y='Spearmans R',hue='Pair Group',hue_order=hue_order, palette='Set2',errorbar='se',
             capsize=.1,errwidth=1,errcolor='0',edgecolor='0',linewidth=2)

sb.move_legend(g,loc='upper left',bbox_to_anchor=(1,1),fontsize=12)
plt.ylim([0,.006])
sb.despine()
plt.tight_layout()
plt.savefig("results/correlation_bar_plot_data.pdf")

plt.show()




df = pd.read_csv("results/pairwise_covariances_from_sim.csv")
#%% for individual bar plots ################
### baseline only
print(df.head())
plt.figure(figsize=(7,5))
rcParams['axes.linewidth']=2
rcParams['font.family']='Arial'
rcParams['font.size']='15'
ax=sb.barplot(data = df,  x='session',y='correlation', hue = 'pair_group',hue_order=hue_order, palette='Set2',errorbar='se',
              width=.5,capsize=.1,errwidth=1,errcolor='0',edgecolor='0',linewidth=2)
sb.despine()
plt.ylim([0,0.4])
sb.move_legend(g,loc='upper left',bbox_to_anchor=(1,1),fontsize=12)
plt.tight_layout()
plt.savefig("results/correlation_bar_plot_sim.pdf")

plt.show()
