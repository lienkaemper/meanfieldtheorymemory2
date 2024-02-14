import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle as pkl

def raster_plot(spktimes, neurons, t_start, t_stop):
    df = pd.DataFrame(spktimes, columns = ["time", "neuron"])
    df = df[(df["time"] < t_stop) &( df["time"] > t_start) ]
    df = df[df["neuron"].isin(neurons)]
    fig, ax = plt.subplots()
    s = 1000
    sns.scatterplot(data = df, x = "time", y = "neuron", marker = "|" , s = s/len(neurons), ax = ax, hue = "neuron",  palette = ["black"])
    plt.legend([],[], frameon=False)
    return fig, ax


with open("results/spikes_h={}i={}.pkl".format(1.75, 0), "rb") as f:
    spktimes = pkl.load(f)

with open("results/index_dict.pkl", "rb") as f:
    index_dict = pkl.load(f)

CA1E = index_dict["CA1E"]
CA1P = index_dict["CA1P"]

CA3E = index_dict["CA3E"]
CA3P = index_dict["CA3P"]    
fig, ax = raster_plot(spktimes,  CA1E[0:10], 1, 1000)

plt.savefig("results/CA1Eraster_after.pdf")

fig, ax = raster_plot(spktimes,  CA1P[0:10], 1, 1000)

plt.savefig("results/CA1Praster_after.pdf")

with open("results/spikes_h={}i={}.pkl".format(1, 0), "rb") as f:
    spktimes = pkl.load(f)

with open("results/index_dict.pkl", "rb") as f:
    index_dict = pkl.load(f)
CA1E = index_dict["CA1E"]
CA1P = index_dict["CA1P"]

CA3E = index_dict["CA3E"]
CA3P = index_dict["CA3P"]    
fig, ax = raster_plot(spktimes,  CA1E[0:10], 1, 1000)

plt.savefig("results/CA1Eraster_before.pdf")

fig, ax = raster_plot(spktimes,  CA1P[0:10], 1, 1000)

plt.savefig("results/CA1Praster_before.pdf")