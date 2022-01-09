
import os
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import dash
#from dash import html, dcc
import dash_core_components as dcc
import dash_html_components as html


"""
## Define your path
PATH = r"D:\francisco_madrigal\Desktop\Tesis\modelo"

def create_path(file_path, path=PATH):
    
    return os.path.join(path, file_path)


## Read neighborhoods DataFrame for later use
colonias = gpd.read_file(create_path(r"colonias_fixed\colonias_fixed.shp"))

## Declare the used coordinate system
colonias.crs = "epsg:4326"

## Read Crimes Probabilities DataFrame adn transform
results_brf = pd.read_csv(create_path(r"df_results\results_brf.csv"), low_memory=False,
                          parse_dates =['Hora'])

results_brf['geometry'] = gpd.GeoSeries.from_wkt(results_brf['geometry'])

results_brf = gpd.GeoDataFrame(results_brf, geometry='geometry')


mean_dayp = results_brf.groupby(['id_colonia', 'day_period'])[['proba_crimen']]\
    .mean().reset_index()

mean_dayp['day_period'].replace({0:'early_morning',
                                 1:'morning',
                                 2:'afternoon',
                                 3:'evening'},
                               inplace=True)

mean_dayp = mean_dayp.pivot_table(index=['id_colonia'], columns='day_period',
                                  values='proba_crimen', aggfunc='first').reset_index()

mean_dayp = mean_dayp.merge(colonias[['id_colonia', 'colonia', 'alcaldi', 'geometry']],
                            how='left',
                            on='id_colonia')

"""
mean_dayp = pd.read_csv(r"/home/fjmadrigal/mysite/mean_predict_graph_prueba.csv")

mean_dayp['geometry'] = gpd.GeoSeries.from_wkt(mean_dayp['geometry'])

mean_dayp = gpd.GeoDataFrame(mean_dayp, geometry='geometry')


mean_dayp = mean_dayp.set_crs(epsg=4326, inplace=True)

mean_daypjson = mean_dayp[['id_colonia', 'geometry']].__geo_interface__

fig = go.Figure(go.Choroplethmapbox(geojson=mean_daypjson,
                                    locations=mean_dayp['id_colonia'], 
                                    z=mean_dayp['morning'],
                                    customdata=mean_dayp['colonia'],
                                    colorscale="Viridis",
                                    marker_opacity=0.7,
                                    zmax=1, zmin=0,
                                    name='',
                                    featureidkey="properties.id_colonia",
                                    hovertemplate= '<b>%{customdata} </b><br><br>Probability=%{z}'
                   )
               )
# fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.update_layout(mapbox=dict(style='carto-positron',
                              zoom=9.9, 
                              center = {"lat": 19.355, "lon": -99.18})) 

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(
    width=1500,
    height=800,
    autosize=False,
    margin=dict(t=100, b=0, l=0, r=0),
)


button1 = dict(method='update',
              label='Morning [6 to 12 hours)',
              args=[{'z':[mean_dayp['morning']]},
                   {'colorscale':'Viridis'}])
                
button2 = dict(method='update',
              label='Afternoon [12 to 18 hours)',
              args=[{'z':[mean_dayp['afternoon']]},
                   {'colorscale':'Viridis'}])
                
    
button3 = dict(method='update',
              label='Evening [18 to 24 hours)',
              args=[{'z':[mean_dayp['evening']]},
                   {'colorscale':'Viridis'}])

button4 = dict(method='update',
              label='Early morning [0 to 6 hours)',
              args=[{'z':[mean_dayp['early_morning']]},
                   {'colorscale':'Viridis'}])

fig.update_layout(updatemenus=[dict(active=0,
                                    buttons=[button1, button2, button3, button4],
                                    direction="down",
                                    pad={"r": 10, "t": 10},
                                    showactive=True,
                                    x=0.01,
                                    xanchor="left",
                                    y=1.1,
                                    yanchor="top")]
                 )

fig.update_layout(
    annotations=[
        dict(text="Time of the day:", showarrow=False,
        x=0.01, y=1.12, yref="paper", align="left")
    ]
)


#fig.show()
#fig.write_html(create_path("map.html"))

# http://127.0.0.1:8050/
# https://dash.plotly.com/deployment
# https://plotly.com/python/scattermapbox/

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Crime Probability by neighborhood in Mexico City',
            style={'textAlign': 'center', 'color': '#444444'
                   }
            ),
    
    html.Div(children='Using a Balanced Random Forest to predict crimes in Mexico City,\
             this map shows the resulting probabilities for each neighborhood.', 
             style={
                 'textAlign': 'center',
                 'color': '#444444'
                 }
             ),
    
    dcc.Graph(figure=fig)
])

#if __name__ == '__main__':
#app.run_server(debug=True)


# https://dash.plotly.com/dash-core-components/datepickerrange
# https://dash.plotly.com/basic-callbacks
# https://towardsdatascience.com/plotly-dash-from-development-to-deployment-c9500d16581a
