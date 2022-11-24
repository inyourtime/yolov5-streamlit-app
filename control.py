import requests

def command(host, port, var, val):
    cmd_query = f'http://{host}:{port}/control?var={var}&val={val}'
    requests.get(cmd_query)