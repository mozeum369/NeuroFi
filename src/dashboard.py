from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import os
import base64

# List of tokens for dropdown filter
TOKENS = ["Pepe", "Zora", "Spell", "LRC", "Townes"]

# Function to encode image to base64
def encode_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{encoded}"
    return None

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], title="Cryptobot Dashboard")

# App layout
app.layout = dbc.Container([
    html.H2("Cryptobot Dashboard"),
    html.P("Live analytics and controls"),

    dbc.Row([
        dbc.Col([
            html.Label("Select Token"),
            dcc.Dropdown(
                id="token-dropdown",
                options=[{"label": token, "value": token} for token in TOKENS],
                value=TOKENS[0],
                clearable=False
            ),
        ], width=4),

        dbc.Col([
            html.Button("Refresh Images", id="refresh-button", n_clicks=0, className="btn btn-primary"),
        ], width=4),

        dbc.Col([
            html.Button("Run Bot Scraper", id="scrape-button", n_clicks=0, className="btn btn-warning"),
        ], width=4),
    ], className="mb-4"),

    dcc.Interval(id="auto-refresh", interval=5000, n_intervals=0),

    html.Div(id="image-container"),

    html.Div(id="scrape-status", className="mt-3 text-success"),
], fluid=True)

# Callback to update images based on refresh or interval
@app.callback(
    Output("image-container", "children"),
    Input("refresh-button", "n_clicks"),
    Input("auto-refresh", "n_intervals"),
    Input("token-dropdown", "value")
)
def update_images(n_clicks, n_intervals, selected_token):
    images = []
    token_img = "token_mentions.png"
    sentiment_img = "sentiment_scores.png"

    token_src = encode_image(token_img)
    sentiment_src = encode_image(sentiment_img)

    if token_src:
        images.append(html.Img(src=token_src, style={"width": "100%", "margin-bottom": "20px"}))
    if sentiment_src:
        images.append(html.Img(src=sentiment_src, style={"width": "100%"}))

    return images

# Callback to simulate bot scraping trigger
@app.callback(
    Output("scrape-status", "children"),
    Input("scrape-button", "n_clicks"),
    prevent_initial_call=True
)
def trigger_scraping(n_clicks):
    # Simulate bot scraping trigger
    return f"✅ Bot scraping triggered at click #{n_clicks}"

# Run server
def run_dashboard(host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
    app.run_server(host=host, port=port, debug=debug)
