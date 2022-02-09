import plotly.graph_objects as go
import matplotlib.pyplot as plt

from metaplex import reverse_metaplex_attributes, avg_attributes, sum_attributes, attrs_difference

def attribute_breakdown(attrs: dict, 
            title: str="", metaplex: bool=True) -> None:

    if metaplex:
        attrs = reverse_metaplex_attributes(attrs)

    display_items = [i for i in attrs.items() if i[1] > 0]
    labels = [i[0] for i in display_items]
    values = [i[1] for i in display_items]
    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title = title)
    fig.show()

def graph_parent_child_genetics(attrs_parent_1, attrs_parent_2, attrs_child):
    # attrs_1_sort = collections.OrderedDict(sorted(attrs_parent_1.items()))
    # attrs_2_sort = collections.OrderedDict(sorted(attrs_parent_2.items()))
    # child_attr_sort = collections.OrderedDict(sorted(attrs_child.items()))
    keys = list(attrs_parent_1.keys())
    values = range(len(keys))
    plt.figure(figsize=(12,3))
    # plt.subplot(3,1,1)
    plt.plot(attrs_parent_1.values(), label="nft-1", color='red')
    plt.plot(attrs_parent_2.values(), label="nft-2", color='orange')
    plt.plot(attrs_child.values(), label="child-generated", color='blue')
    plt.xticks(values, keys, rotation=75)
    plt.legend(loc="upper right")
    plt.show()
    parent_avgs = attrs_difference(attrs_1_sort, attrs_2_sort)
    parent_avgs = {k: (v[0]+v[1])/2 for k, v in parent_avgs.items()}
    plt.figure(figsize=(12,3))
    plt.plot(parent_avgs.values(), label="parent-avg", color='red')
    plt.plot(child_attr_sort.values(), label="child-generated", color='green')
    plt.xticks(values, keys, rotation=75)
    plt.legend(loc="upper right")
    plt.show()