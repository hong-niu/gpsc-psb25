import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable



# before naming color column
def plot_tracts_OLD(cluster_filename, title, out_path):
    data = pd.read_excel(cluster_filename, index_col=0)
    block_2010 = pd.read_csv("CensusPlotting/2010_Census_Tracts.csv")
    block_2010.shape

    print(block_2010.head().T)

    blocks_map_2010 = gpd.read_file("./CensusPlotting/2010_Census_Tracts/2010_Census_Tracts.shp")
    print(blocks_map_2010.head())
    blocks_map_2010['Tract'] = blocks_map_2010['geoid10'].astype(int)
    data['Tract'] = data['Tract'].astype(int)
    data['new_clust'] = data['new_clust'].astype(str)
    cols = ['Tract', 'new_clust']
    state_blocks_map = blocks_map_2010.merge(data.loc[:, cols],
                                             how='outer',
                                             on='Tract',
                                             validate='many_to_many')

    state_blocks_map = state_blocks_map.to_crs("EPSG:4326")
    outline = state_blocks_map.dissolve(by=state_blocks_map.columns[0],
                                        aggfunc='first')

    fig, ax = plt.subplots(figsize=(10, 8))
    divider = make_axes_locatable(ax)

    state_blocks_map.plot(ax=ax,
                          column='new_clust',
                          cmap= 'Set2_r',   # 'Set2_r',  # coolwarm_r
                          legend=True,
                          missing_kwds={"color": "white"},
                          alpha=0.85
                          )

    outline.plot(ax=ax,
                 facecolor='none',
                 edgecolor='gray',
                 linewidth=.5)
    ax.set_title(title, fontsize=20)

    cities = pd.read_excel('./data/cities-NC.xlsx', sheet_name='zoomOUT')
    for i in range(len(cities.index)):
        if (cities['City'][i]=='Winston-Salem'):
            ax.scatter(cities['Longitude'][i], cities['Latitude'][i], s=5, color='red')
            ax.annotate(cities['City'][i], (cities['Longitude'][i]-1.3, cities['Latitude'][i]), color='black', size=10, weight='bold')
            continue
        if (cities['City'][i]=='Elizabeth City'):
            ax.scatter(cities['Longitude'][i], cities['Latitude'][i], s=5, color='red')
            ax.annotate(cities['City'][i], (cities['Longitude'][i]-1.2, cities['Latitude'][i]), color='black', size=10, weight='bold')
            continue
        ax.scatter(cities['Longitude'][i], cities['Latitude'][i], s=5, color='red')
        ax.annotate(cities['City'][i], (cities['Longitude'][i], cities['Latitude'][i]), color='black', size=10, weight='bold')

    fig.patch.set_visible(True)
    ax.axis('on')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_tracts(cluster_filename, title, out_path, colorBy = "new_clust"):
    data = pd.read_excel(cluster_filename, index_col=0)
    block_2010 = pd.read_csv("CensusPlotting/2010_Census_Tracts.csv")
    block_2010.shape

    print(block_2010.head().T)

    blocks_map_2010 = gpd.read_file("./CensusPlotting/2010_Census_Tracts/2010_Census_Tracts.shp")
    print(blocks_map_2010.head())
    blocks_map_2010['Tract'] = blocks_map_2010['geoid10'].astype(int)
    data['Tract'] = data['Tract'].astype(int)
    data[colorBy] = data[colorBy].astype(str)
    cols = ['Tract', colorBy]
    state_blocks_map = blocks_map_2010.merge(data.loc[:, cols],
                                             how='outer',
                                             on='Tract',
                                             validate='many_to_many')

    state_blocks_map = state_blocks_map.to_crs("EPSG:4326")
    outline = state_blocks_map.dissolve(by=state_blocks_map.columns[0],
                                        aggfunc='first')

    fig, ax = plt.subplots(figsize=(10, 8))
    divider = make_axes_locatable(ax)

    state_blocks_map.plot(ax=ax,
                          column = colorBy,
                          cmap= 'Set2_r',   # 'Set2_r',  # coolwarm_r
                          legend=True,
                          missing_kwds={"color": "white"},
                          alpha=0.85
                          )

    outline.plot(ax=ax,
                 facecolor='none',
                 edgecolor='gray',
                 linewidth=.5)
    ax.set_title(title, fontsize=20)

    cities = pd.read_excel('./data/cities-NC.xlsx', sheet_name='zoomOUT')
    for i in range(len(cities.index)):
        if (cities['City'][i]=='Winston-Salem'):
            ax.scatter(cities['Longitude'][i], cities['Latitude'][i], s=5, color='red')
            ax.annotate(cities['City'][i], (cities['Longitude'][i]-1.3, cities['Latitude'][i]), color='black', size=10, weight='bold')
            continue
        if (cities['City'][i]=='Elizabeth City'):
            ax.scatter(cities['Longitude'][i], cities['Latitude'][i], s=5, color='red')
            ax.annotate(cities['City'][i], (cities['Longitude'][i]-1.2, cities['Latitude'][i]), color='black', size=10, weight='bold')
            continue
        ax.scatter(cities['Longitude'][i], cities['Latitude'][i], s=5, color='red')
        ax.annotate(cities['City'][i], (cities['Longitude'][i], cities['Latitude'][i]), color='black', size=10, weight='bold')

    fig.patch.set_visible(True)
    ax.axis('on')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

# before naming color column
def plot_tracts_subfig_OLD(cluster_filename, title, out_path, x_limits, y_limits):
    data = pd.read_excel(cluster_filename, index_col=0)
    block_2010 = pd.read_csv("CensusPlotting/2010_Census_Tracts.csv")
    block_2010.shape

    print(block_2010.head().T)

    blocks_map_2010 = gpd.read_file("./CensusPlotting/2010_Census_Tracts/2010_Census_Tracts.shp")
    print(blocks_map_2010.head())
    # blocks_map_2010.plot()
    # plt.show()
    blocks_map_2010['Tract'] = blocks_map_2010['geoid10'].astype(int)
    data['Tract'] = data['Tract'].astype(int)
    data['new_clust'] = data['new_clust'].astype(str)
    cols = ['Tract', 'new_clust']
    state_blocks_map = blocks_map_2010.merge(data.loc[:, cols],
                                             how='outer',
                                             on='Tract',
                                             # validate='one_to_one')
                                             validate='many_to_many')

    state_blocks_map = state_blocks_map.to_crs("EPSG:4326")
    outline = state_blocks_map.dissolve(by=state_blocks_map.columns[0],
                                        aggfunc='first')

    fig, ax = plt.subplots(figsize=(10, 8))
    divider = make_axes_locatable(ax)
    state_blocks_map.plot(ax=ax,
                          column="new_clust",
                          cmap='Set2_r',  # 'Set2_r',  # coolwarm_r
                          legend=True,
                          # cax=cax,
                          missing_kwds={"color": "white"},
                          alpha=0.85
                          )

    outline.plot(ax=ax,
                 facecolor='none',
                 edgecolor='gray',
                 linewidth=.5)
    cities = pd.read_excel('./data/cities-NC.xlsx', sheet_name='zoomIN')
    for i in range(len(cities.index)):
        ax.scatter(cities['Longitude'][i], cities['Latitude'][i], s=5, color='red')
        ax.annotate(cities['City'][i], (cities['Longitude'][i], cities['Latitude'][i]), color='black', size=8, weight='bold')

    ax.set_title(title, fontsize=20)
    fig.patch.set_visible(True)
    ax.axis('on')
    plt.tight_layout()
    plt.xlim(x_limits[0], x_limits[1])
    plt.ylim(y_limits[0], y_limits[1])
    plt.savefig(out_path, dpi=300)
    plt.show()

def plot_tracts_subfig(cluster_filename, title, out_path, x_limits, y_limits, colorBy = "new_clust"):
    data = pd.read_excel(cluster_filename, index_col=0)
    block_2010 = pd.read_csv("CensusPlotting/2010_Census_Tracts.csv")
    block_2010.shape

    print(block_2010.head().T)

    blocks_map_2010 = gpd.read_file("./CensusPlotting/2010_Census_Tracts/2010_Census_Tracts.shp")
    print(blocks_map_2010.head())
    # blocks_map_2010.plot()
    # plt.show()
    blocks_map_2010['Tract'] = blocks_map_2010['geoid10'].astype(int)
    data['Tract'] = data['Tract'].astype(int)
    data[colorBy] = data[colorBy].astype(str)
    cols = ['Tract', colorBy]
    state_blocks_map = blocks_map_2010.merge(data.loc[:, cols],
                                             how='outer',
                                             on='Tract',
                                             # validate='one_to_one')
                                             validate='many_to_many')

    state_blocks_map = state_blocks_map.to_crs("EPSG:4326")
    outline = state_blocks_map.dissolve(by=state_blocks_map.columns[0],
                                        aggfunc='first')

    fig, ax = plt.subplots(figsize=(10, 8))
    divider = make_axes_locatable(ax)
    state_blocks_map.plot(ax=ax,
                          column=colorBy,
                          cmap='Set2_r',  # 'Set2_r',  # coolwarm_r
                          legend=True,
                          # cax=cax,
                          missing_kwds={"color": "white"},
                          alpha=0.85
                          )

    outline.plot(ax=ax,
                 facecolor='none',
                 edgecolor='gray',
                 linewidth=.5)
    cities = pd.read_excel('./data/cities-NC.xlsx', sheet_name='zoomIN')
    for i in range(len(cities.index)):
        ax.scatter(cities['Longitude'][i], cities['Latitude'][i], s=5, color='red')
        ax.annotate(cities['City'][i], (cities['Longitude'][i], cities['Latitude'][i]), color='black', size=8, weight='bold')

    ax.set_title(title, fontsize=20)
    fig.patch.set_visible(True)
    ax.axis('on')
    plt.tight_layout()
    plt.xlim(x_limits[0], x_limits[1])
    plt.ylim(y_limits[0], y_limits[1])
    plt.savefig(out_path, dpi=300)
    plt.show()



if __name__ == '__main__':
    plot_tracts()


