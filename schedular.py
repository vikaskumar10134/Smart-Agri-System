from langgraph.graph import StateGraph , START , END
from mcp import ClientSession , StdioServerParameters
from mcp.client.stdio import stdio_client
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from dotenv import load_dotenv
from typing import Literal
from zoneinfo import ZoneInfo

from graph import build_resources , DB_URL


from state import *
import sys
import asyncio
import traceback
import datetime
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Scheduler Code'

class State(BaseModel):
    farmer_ids : List[str]

    # Weather mcp tool output
    weather_forcast: Optional[WeatherForecast] = Field(
        default=None, description="Weather forecast fetched via weather_mcp_server, if the intent required it."
    )
    weather_tool_error: Optional[str] = Field(
        default=None, description="Error message if the weather MCP tool call failed, else None."
    )

    caveats: List[str] = Field(
        default_factory=list, description="Accumulated caveat tags (e.g. 'weather_unavailable') attached during the run."
    )

    should_alert : Optional[bool] = None
    alert_reason : Optional[str] = None

    location : dict[str , float]

    whatsapp_alert_sent : Optional[bool] = None
    whatsapp_tool_error : Optional[str] = None






WEATHER_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather-mcp-server.py")
WHATSAPP_SERVER_SCRIPT = r'D:/GenAI/Langchain/mcps/whatsapp-mcp-server'



weather_server_params = StdioServerParameters(
    command=sys.executable,      # uses the same Python/venv running this script
    args=[WEATHER_SERVER_SCRIPT],      
)

# whatsapp_server_params = StdioServerParameters(
#     command=sys.executable,      # uses the same Python/venv running this script
#     args=[WHATSAPP_SERVER_SCRIPT],      
# )


async def load_active_farmers_node(state : State):

    print('From the laod active farmer node')

    checkpointer , store , stack = await build_resources(DB_URL)

    namespace = await store.alist_namespaces(prefix=('farmer_profile',))
    farmer_ids = [ns[-1] for ns in namespace]

    return {'farmer_ids' : farmer_ids}


async def weather_mcp_server(state : State) -> dict:

    print('From the weather mcp server node')

    '''
    Call the weather MCP tool get_forecast(lat, lon, days).

    Args:
        state (AgriAdvisoryState): Current graph state. Reads
            `farmer_profile.location` for coordinates.

    Returns:
        dict: Partial state update with either:
            - weather_forcast (WeatherForecast): On success.
            - weather_tool_error (str), caveats (list[str]): On failure or
              missing location data.
    '''

    lat = state.location.get('lat')
    lon = state.location.get('lon')

    if not lat and not lon:
        return {'weather_tool_error' : 'Missing_farmer_location'}


    try:
        async with stdio_client(weather_server_params) as (read , write):
            async with ClientSession(read , write) as session:
                await session.initialize()
                result = await session.call_tool(
                    'weather_forcasting',
                    arguments={'lattitude': lat, 'longitude': lon, 'days': 5 , 'aqi' : 'no' , 'alert' : 'no'},
                )

                payload = result.structuredContent

                forecast = WeatherForecast(
                    source="weather_mcp_server",
                    fetch_at=datetime.datetime.utcnow().isoformat(),
                    daily=payload.get("result", {}).get("forecast", []),
                )
        

        return {'weather_forcast' : forecast}

    except Exception as e:
        if hasattr(e, "exceptions"):  # BaseExceptionGroup / ExceptionGroup
            for sub in e.exceptions:
                print("Sub-exception:", repr(sub))
                traceback.print_exception(type(sub), sub, sub.__traceback__)
        else:
            traceback.print_exc()
        return {'weather_tool_error': str(e), 'caveats': ['weather_unavailable']}


async def alert_decision_node(state : State):

    print('From the alert decision node')

    forecast = state.weather_forcast

    if not forecast or not forecast.daily:
        return {'should_alert' : False , 'alert_reason' : None}

    tomarrow = forecast.daily[0]

    chance_of_rain = tomarrow.get('chance_of_rain' ,0)
    max_temp = tomarrow.get('max_temp_c',0)

    if chance_of_rain > 70:
        return {'should_alert' : True , 'alert_reason' : f'Havily rain likely {chance_of_rain}% chance'}

    elif max_temp > 42:
        return {'should_alert' : True , 'alert_reason' : f' Exterme heat expected ({max_temp}\u00b0C)'}

    return {'should_alert' : False , 'alert_reason' : None}


def route_after_decision(state : State) -> Literal['whatsapp_mcp_server' , 'END']:

    if state.should_alert:
        return 'END'
    else:
        return 'END'


async def whatsapp_mcp_server(state : State):

    reason = state.alert_reason
    number_id = ''

    try:
        async with stdio_client(WHATSAPP_SERVER_SCRIPT) as (read , write):
            async with ClientSession(read , write) as session:
                await session.initialize()

                result = await session.call_tool(
                    'send_text_message',
                    arguments= {'to' : number_id , 'message' : reason , 'preview_url' : True}
                )

                return {'whatsapp_alert_sent' : result.structuredContent.success , 'alert_reason' : reason}

    except Exception as e:
            return {'whatsapp_tool_error' : str(e) , 'caveats' : ['whatsapp_tool_unavailable'] }

            





scheduler_builder = StateGraph(State)

scheduler_builder.add_node('load_active_farmers_node' , load_active_farmers_node)
scheduler_builder.add_node('weather_mcp_server' , weather_mcp_server)
scheduler_builder.add_node('whatsapp_mcp_server' , whatsapp_mcp_server)
scheduler_builder.add_node('alert_decision_node' , alert_decision_node)


scheduler_builder.add_edge(START , 'load_active_farmers_node')
scheduler_builder.add_edge('load_active_farmers_node' , 'weather_mcp_server')
scheduler_builder.add_edge('weather_mcp_server' , 'alert_decision_node')

scheduler_builder.add_conditional_edges(
    'alert_decision_node',

    route_after_decision,
    {
        'whatsapp_mcp_server' : 'whatsapp_mcp_server',
        'END' : END,
    }
)
scheduler_builder.add_edge('whatsapp_mcp_server' , END)

async def run_graph():

    
    graph = scheduler_builder.compile()

    intial_state = {
        'farmer_ids' : [],
        'caveats' : [],
        'location' : {'lat' : 28.63, 'lon' : 79.81}

    }

    result =  await graph.ainvoke(intial_state)


    for key , value in result.items():
        print(f"{key}: {value}\n")


# async def run_graph():

#     result = 6 + 7

#     print('The sum is ' , result)

def scheduled_job():
    asyncio.run(run_graph())



scheduler = BlockingScheduler(
    timezone = ZoneInfo('Asia/Kolkata')
)

scheduler.add_job(

    scheduled_job,
    trigger='cron',
    hour=20,
    minute=12,
)

try:
    scheduler.start()

except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()

