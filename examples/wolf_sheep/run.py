from wolf_sheep.server import *


server = mesa.visualization.ModularServer(
    WolfSheep, [canvas_element, chart_element], "Wolf Sheep Predation", model_params
)
server.port = 8521

server.launch(open_browser=False)


