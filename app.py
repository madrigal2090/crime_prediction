
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
import dash
#from dash import html, dcc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

## Read Crimes Probabilities DataFrame and transform
results_brf = pd.read_csv(r"modelo\df_results\reduced_results_brf.csv")
colonias = pd.read_csv(r"modelo\df_results\reduced_colonias.csv")

results_brf = results_brf.merge(colonias[['id_colonia', 'colonia']],
                                how='left',
                                on='id_colonia')


colonias['geometry'] = gpd.GeoSeries.from_wkt(colonias['geometry'])

colonias = gpd.GeoDataFrame(colonias, geometry='geometry')

colonias = colonias.set_crs(epsg=4326, inplace=True)

colonias = colonias[['id_colonia', 'geometry']].__geo_interface__

results_brf = results_brf.sort_values(by='day_period')

results_brf['day_period'].replace({0:'Early morning [0 to 6 hours)',
                                  1:'Morning [6 to 12 hours)',
                                  2:'Afternoon [12 to 18 hours)',
                                  3:'Evening [18 to 24 hours)'},
                               inplace=True)

day_period = results_brf['day_period'].unique()

day_period = np.insert(day_period, 0, 'Aggregate')

results_brf = results_brf.sort_values(by='dia_semana')

results_brf['dia_semana'].replace({0:'Sunday',
                                  1:'Monday',
                                  2:'Tuesday',
                                  3:'Wednesday',
                                  4:'Thursday',
                                  5:'Friday',
                                  6:'Saturday'},
                               inplace=True)

dia_semana = results_brf['dia_semana'].unique()

dia_semana = np.insert(dia_semana, 0, 'Aggregate')


## Create dash plot
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
        
    html.Br(),
    
    html.Div(children=[
        
        html.Label(['Select day of the week:'], 
                   style={'font-weight': 'bold', "text-align": "center"}),
        
        dcc.Dropdown(
            id='dia_semana-column',
            options=[{'label': i, 'value': i} for i in dia_semana],
            value='Aggregate',
            clearable=False, searchable=False
            ),
        ], style={'width': '20%'}
        ),

    html.Div(children=[
        
        html.Label(['Select time of the day'], 
                   style={'font-weight': 'bold', "text-align": "center"}),
        dcc.Dropdown(
            id='day_period-column',
            options=[{'label': i, 'value': i} for i in day_period],
            value='Aggregate',
            clearable=False, searchable=False
        ),
    ], style={'width': '20%'}
        
        ),
    
    dcc.Graph(id='map-graph')
    ])
    

@app.callback(
    Output('map-graph', 'figure'),
    Input('dia_semana-column', 'value'),
    Input('day_period-column', 'value'))
def update_graph(dia_semana_val, day_period_val):
    
    if dia_semana_val == 'Aggregate' and day_period_val == 'Aggregate':
        
        df = results_brf.groupby(['id_colonia', 'colonia'])[['proba_crimen']]\
        .mean().reset_index()
        
    if dia_semana_val != 'Aggregate' and day_period_val == 'Aggregate':
        
        df = results_brf.groupby(['id_colonia', 'colonia', 'dia_semana'])[['proba_crimen']]\
            .mean().reset_index()
            
        df = df[df['dia_semana'] == dia_semana_val]
        
        
    if dia_semana_val == 'Aggregate' and day_period_val != 'Aggregate':
        
        df = results_brf.groupby(['id_colonia', 'colonia', 'day_period'])[['proba_crimen']]\
            .mean().reset_index()
            
        df = df[df['day_period'] == day_period_val]
        
    if dia_semana_val != 'Aggregate' and day_period_val != 'Aggregate':
        
        df = results_brf.groupby(['id_colonia', 'colonia', 'day_period', 'dia_semana'])\
            [['proba_crimen']].mean().reset_index()
            
        df = df[(df['day_period'] == day_period_val) & (df['dia_semana'] == dia_semana_val)]


    fig = go.Figure(go.Choroplethmapbox(geojson=colonias,
                                        locations=df['id_colonia'], 
                                        z=df['proba_crimen'],
                                        customdata=df['colonia'],
                                        colorscale="Viridis",
                                        marker_opacity=0.7,
                                        zmax=1, zmin=0, zauto=False,
                                        colorbar_title = "<b>Crime<br>Probability</b>",
                                        name='',
                                        featureidkey="properties.id_colonia",
                                        hovertemplate= '<b>%{customdata} </b><br><br>Probability=%{z}'
                       )
                   )
    # fig.update_geos(fitbounds="locations", visible=False)
    #fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_layout(mapbox=dict(style='carto-positron',
                                  zoom=9.9, 
                                  center = {"lat": 19.355, "lon": -99.18})) 

    fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(
        width=1500,
        height=800,
        autosize=False,
    )

    return fig

#if __name__ == '__main__':
#app.run_server(debug=True)
