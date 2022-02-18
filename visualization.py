import plotly.graph_objects as go
import matplotlib.pyplot as plt
from utils import print_pretty
from metaplex import reverse_metaplex_attributes

def imshow(img, title='', size=(10,10)):
    plt.figure(figsize=size)
    plt.title(title)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_histogram(hist):
    plt.plot(hist[0, 1:], color='red')
    plt.plot(hist[1, 1:], color='green')
    plt.plot(hist[2, 1:], color='blue')
    plt.show()

def plot_bar(keys, values):
    plt.bar(keys, values)
    plt.show()

def display_multiple_images(images=[], titles=[], size=(20,10)):
    plt.figure(figsize=size)
    for i in range(len(images)):
        plt.subplot(1,len(images), i+1)
        plt.title(titles[i])
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()
    plt.close()

def display_image_meta(image, metadata, size=(10,10)):
    imshow(
        image, 
        f"{metadata['symbol']} : {metadata['name']}",
        size)
    print_pretty(metadata)

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

def graph_attributes(attributes=[], names=[], figsize=(12,3)):
    keys = list(attributes[0].keys())
    values = range(len(keys))
    plt.figure(figsize=figsize)
    for name, attr in zip(names, attributes):

        plt.plot(attr.values(), label=name)
    plt.xticks(values, keys, rotation=75)
    plt.legend(loc="upper right")
    plt.show()