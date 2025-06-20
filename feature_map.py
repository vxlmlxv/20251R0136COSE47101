from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import json


app = Dash(__name__)

app.layout = html.Div([
    html.H4('Feature Map Visualization'),
    html.P("Select a feature:"),
    dcc.RadioItems(
        id='feature',
        options=["Top1", "Top2", "Top3"],
        value="Top1",
        inline=True
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"),
    Input("feature", "value"))
def display_choropleth(feature):
    df_path = 'insight.csv'
    df = pd.read_csv(df_path)
    json_path = 'dong.geojson'
    geojson = json.load(open(json_path))

    fig = px.choropleth_map(
        df, geojson=geojson, color=feature,
        locations="dong", featureidkey="properties.dong",
        center={"lat": 37.5665, "lon": 126.9780}, zoom=9.5,
        map_style="open-street-map")
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0})

    return fig


app.run(debug=True)