import bokeh
from bokeh.io import show
from bokeh.plotting import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import LinearColorMapper, Slider, Button
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, BoxZoomTool, ResetTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges
import  pandas as pd
import numpy as np
from networkx import nx
from pathlib import Path

# create a new DataFrame that describes the graph
def readDB(df):
    df.sort_values(['UserID', 'ItemID'], inplace=True)
    column_names = ['U', 'V', 'ItemID', 'Weight']
    new_df = pd.DataFrame(columns=column_names)
    for i in range(1, df['UserID'].max()):
        u = df[df['UserID'] == i]

        for j in range(i + 1, df['UserID'].max() + 1):
            v = df[df['UserID'] == j]
            SharedGroup1 = v[v['ItemID'].isin(u['ItemID'].values)]

            if SharedGroup1.shape[0] < 2:
                continue

            SharedGroup2 = u[u['ItemID'].isin(v['ItemID'].values)]
            avg = np.mean(abs(SharedGroup1['Rating'].values - SharedGroup2['Rating'].values))

            for k in range(len(SharedGroup1)):
                # add the edges to the new DataFrame
                tempDF = pd.DataFrame([[i, j, SharedGroup1['ItemID'].values.tolist(), avg]], columns=column_names)
                new_df = new_df.append(tempDF, ignore_index=True)

    # sort the new DataFrame by weight and save it
    new_df = new_df.sort_values(['Weight'], ascending=False)
    new_df.to_pickle("graph.csv")


def create_graph_rendrer(threshold,df,graph_df):
    plot_new = Plot(plot_width=1100, plot_height=800,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    plot_new.title.text = "Dash-Graph"

    color_dict = {}

    for i in range(1, df['UserID'].max() + 1):
        color_dict[i] = df.groupby('UserID').groups[i].size * 1.9

    pal_hex_lst = bokeh.palettes.viridis(21)
    mapper = LinearColorMapper(palette=pal_hex_lst, low=0, high=21)

    G = create_graph(graph_df, threshold, True)

    # add graph's attributes
    node_size = {k: ((5 * v) + 5) for k, v in G.degree()}
    nx.set_node_attributes(G, color_dict, 'node_color')
    nx.set_node_attributes(G, node_size, 'node_size')
    graph_renderer = from_networkx(G, nx.spring_layout)
    graph_renderer.node_renderer.glyph = Circle(size='node_size',
                                                fill_color={'field': 'node_color', 'transform': mapper})
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="black", line_alpha=0.8, line_width=0.5)
    graph_renderer.selection_policy = NodesAndLinkedEdges()

    plot.renderers.append(graph_renderer)




def create_graph(graph_df,threshold,showdegzero):
    filtered_df = graph_df[graph_df['Weight'] <= threshold]

    # Build Graph DB
    G = nx.from_pandas_edgelist(filtered_df, 'U', 'V', ['ItemID'])
    G.edges(data=True)
    if showdegzero is True:
        # show under threshold nodes
        for node_key in range(1, df['UserID'].max() + 1):
            if G.has_node(node_key) is not True:
                G.add_node(node_key)
    return G


plot = Plot(plot_width=1100, plot_height=800,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
plot.title.text = "EMC-Graph"

my_file = Path("graph.csv")
#read db
df = pd.read_csv('ml1.data', sep='\t', skiprows=1, names=['UserID', 'ItemID', 'Rating', 'TimeStamp'])

if not (my_file.is_file()):
    readDB(df)

graph_df = pd.read_pickle(my_file)

max_rate = df['Rating'].max()
slider = Slider(start=0, end=max_rate, value=5, step=.1,title="Threshold")
avg_th_up_btn = Button(label = "Change Threshold")
hide_nodes =  Button(label = "Hide nodes with no neighbours")
show_nodes = Button(label = "Show nodes with no neighbours")
def callback(attr, old, new):
    pass
    # avg_threshold_up()
def avg_threshold_up():
    if slider.value == 0:
        return
    slider.value = slider.value-1
    create_graph_rendrer(slider.value-1, df, graph_df)


# Render default graph
create_graph_rendrer(slider.value,df,graph_df)
node_hover_tool = HoverTool(tooltips=[("User ID", "@index")])
plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())
layout = row(column(avg_th_up_btn,hide_nodes,show_nodes),plot, widgetbox(slider))
avg_th_up_btn.on_click(avg_threshold_up)
# show(layout)
curdoc().add_root(layout)
