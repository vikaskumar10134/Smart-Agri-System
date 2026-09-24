from langchain_nvidia_ai_endpoints import ChatNVIDIA , NVIDIAEmbeddings
from langchain_core.messages import SystemMessage , HumanMessage

from langgraph.store.postgres import AsyncPostgresStore
from langgraph.graph import StateGraph , START , END
from langgraph.store.base import BaseStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from mcp import ClientSession , StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from typing import Literal

import json
import datetime
import os
import traceback
import uuid
import asyncio
import sys
import warnings
import logging


asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from state import *
from prompts import ADVISORY_PROMPT , INTENT_ROUTER_PROMPT , LTM_WRITE_PROMPT



warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")
logger = logging.getLogger(__name__)

load_dotenv()


DB_URL = os.getenv('DB_URL')
WEATHER_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weather-mcp-server.py")
MANDI_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandi-mcp-server.py")
os.environ['LANGCHAIN_PROJECT'] = 'Smart Agri Advisory System'




weather_server_params = StdioServerParameters(
    command=sys.executable,      # uses the same Python/venv running this script
    args=[WEATHER_SERVER_SCRIPT],      
)

mandi_server_params = StdioServerParameters(
    command=sys.executable,
    args=[MANDI_SERVER_SCRIPT],
)

llm = ChatNVIDIA(model = 'nvidia/nemotron-3-super-120b-a12b')



builder = StateGraph(AgriAdvisoryState)


async def intent_router_node(state : AgriAdvisoryState) -> dict:

    print('From the intent router node')

    '''
    Classify the farmer's free-text query into one of four supported intents.

    Args:
        state (AgriAdvisoryState): Current graph state. Reads `raw_query` and
            `detected_language` to build the classification prompt.

    Returns:
        dict: Partial state update with keys:
            - intent (str): One of 'crop_selection', 'irrigation_timing',
              'pest_diagnosis', 'sell_or_hold_price'.
            - intent_confidence (float): Confidence score in [0, 1] for the
              classification (currently a naive proxy value).
    '''

    structure_llm = llm.with_structured_output(IntentSchema , method = 'function_calling')
    result : IntentSchema = await structure_llm.ainvoke(
        [
            SystemMessage(content = INTENT_ROUTER_PROMPT),
            HumanMessage(content= f'Farmer query {state.detected_language} : {state.raw_query}')
        ] 
    )

    intent_confidence = 0.9 if result.intent else 0.0

    return {
        'intent' : result.intent , 
        'intent_confidence' : intent_confidence,
        'llm_used' : state.llm_used+1,
    }





async def ltm_write(state : AgriAdvisoryState , store : BaseStore) -> dict:

    print('From the ltm write node')
    
    '''Extract atomic, de-duplicated facts from the turn and persist new ones.

    Args:
        state (AgriAdvisoryState): Current graph state. Reads `raw_query`,
            `final_response`, and `farmer_id`.
        store (BaseStore): LangGraph-injected store, written to directly as a
            side effect (no state fields are returned).

    Returns:
        dict: Always an empty dict — this node mutates long-term memory
            directly via `store.aput` rather than the graph state.
    '''
    
    asyncio.create_task(_write_ltm_background(state, store))
    return {'llm_used': state.llm_used + 1}

async def _write_ltm_background(state, store):
    namespace = ('farmer_profile', state.farmer_id, 'profile')
    items = await store.asearch(namespace)
    user_details_content = '\n'.join(f"-{it.value.get('data', '')}" for it in items) or "No memories on file yet."

    ltm_write_llm = ChatNVIDIA(model="nvidia/nemotron-3.5-lightning-30b-a3b" , max_completion_tokens=8000)
    ltm_write_structure_llm = ltm_write_llm.with_structured_output(MemoryDecision)
    
    try:
        decision = await ltm_write_structure_llm.ainvoke([
            SystemMessage(content=LTM_WRITE_PROMPT.format(existing_memories=user_details_content)),
            HumanMessage(content=f"Query: {state.raw_query}\nResponse: {state.final_response}"),
        ])

        # args = response.tool_calls[0]["args"]
        # decision = MemoryDecision(**args)


        if decision.should_write:
            for memory in decision.memories:
                if memory.is_new:
                    await store.aput(namespace, str(uuid.uuid4()), {'data': memory.text})
    except Exception as e:
        logger.warning(f"LTM write failed for farmer {state.farmer_id}: {e}")


async def context_node(state : AgriAdvisoryState , store : BaseStore) -> dict:
    print('From the context node')

    '''
        Load the farmer's long-term profile from PostgresStore before any tool call.
    
        Args:
            state (AgriAdvisoryState): Current graph state. Reads `farmer_id` and
                `raw_query` (used as the semantic search query against the store).
            config (RunnableConfig): LangGraph-injected run config (thread/run metadata).
            store (BaseStore): LangGraph-injected long-term memory store, keyword-only.
    
        Returns:
            dict: Partial state update with keys:
                - farmer_profile (FarmerProfile | None): The retrieved profile, or
                  None if no profile exists yet for this farmer_id.
                - caveats (list[str]): Only set (['no_stored_profile']) when no
                  profile was found; omitted otherwise.
    '''
    
    namespace = ('farmer_profile' , state.farmer_id, 'profile')

    record = await store.asearch(namespace , query = state.raw_query , limit = 1)

    if not record :
        return {'profile' : {} , 'caveats' : ['missing_profile']}

    profile_dict = record[0].value
    farmer_profile = FarmerProfile(**profile_dict)

    return {'farmer_profile' : farmer_profile}

def route_after_context(state : AgriAdvisoryState) -> Literal['weather_mcp_server' , 'mandi_mcp_server' , 'advisory_node']:

    if state.intent == 'crop_selection':
        return 'weather_mcp_server'
    
    elif state.intent == 'irrigation_timing':
        return 'weather_mcp_server'
    
    elif state.intent == 'sell_or_hold_price':
        return 'mandi_price_mcp_server'
    
    elif state.intent == 'pest_diagnosis':
        return 'advisory_node'

    


async def weather_mcp_server(state : AgriAdvisoryState) -> dict:

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

    if state.farmer_profile is None:
        return {'weather_tool_error' : 'Missing_farmer_location'}

    lat = state.farmer_profile.location.get('lat')
    lon = state.farmer_profile.location.get('lon')

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

def route_after_weather(state : AgriAdvisoryState) -> Literal['mandi_price_mcp_server' , 'advisory_node']:

    if state.intent == 'crop_selection':
        return 'mandi_price_mcp_server'

    if state.intent == 'irrigation_timing':
        return 'advisory_node'


async def mandi_price_mcp_server(state : AgriAdvisoryState) -> dict:

    print('From the mandi price mcp server node')

    '''
    Call the mandi price MCP tool `get_mandi_price(commodity, district)`.

    Args:
        state (AgriAdvisoryState): Current graph state. Reads
            `farmer_profile.current_crop` and `farmer_profile.location['district']`.

    Returns:
        dict: Partial state update with either:
            - mandi_prices (list[MandiPricePoint]): On success.
            - mandi_tool_error (str): On failure or missing crop/district.
    '''

    commodity = state.farmer_profile.current_crop if state.farmer_profile else None
    state_name = state.farmer_profile.location.get('state_name') if state.farmer_profile else None
    market = state.farmer_profile.location.get('market') if state.farmer_profile else None
    variety = state.farmer_profile.variety if state.farmer_profile else None
    grade = state.farmer_profile.grade if state.farmer_profile else None
    district = state.farmer_profile.location.get('district') if state.farmer_profile else None

    if not state_name or not district or not market or not commodity or not variety or not grade:
        return {'mandi_tool_error' : 'Missing parameter'}

    try:
        async with stdio_client(mandi_server_params) as (read , write):
            async with ClientSession(read , write) as session:
                await session.initialize()
                result = await session.call_tool(
                    'get_product_mandi_price',
                    arguments= {'state'  : state_name , 'district' : district , 'market' : market, 'commodity' : commodity , 'variety' : variety , 'grade' : grade}

                )

                data = result.structuredContent
                if not data["success"]:
                    raise RuntimeError(data["error"])


                
                rows = data.get("data").get("records")[0]

        
                price = MandiPricePoint(
                    commodity=rows["commodity"],
                    market=rows["market"],
                    date=rows["arrival_date"],
                    model_price_per_quintal=rows["modal_price"],
                    source = "data.gov.in Agmarknet",
                )

        return {'mandi_prices' : [price]}

    except Exception as e:
        return {'mandi_tool_error' : str(e) , 'caveats' : ['mandi_price_unavailable'] }


async def advisory_node(state : AgriAdvisoryState , store : BaseStore) -> dict:

    print('From the advisory node')

    '''Generate a grounded advisory using whatever context/tool data is present.

    Args:
        state (AgriAdvisoryState): Current graph state. Reads `intent`,
            `farmer_profile`, `weather_forcast`, `mandi_prices`,
            `pest_diagnosis`, and `raw_query`.
        store (BaseStore): LangGraph-injected store, used here for an agronomy
            knowledge-base similarity search (RAG).

    Returns:
        dict: Partial state update with:
            - advisory_draft (str): LLM-generated recommendation text.
            - rag_snippet_used (list[str]): Knowledge-base snippets used as context.
    '''

    rag_hits = await store.asearch(
        ('agronomy_kb' ,) , query= state.raw_query , limit=3
    )

    rag_snippets = [hit.value['text'] for hit in rag_hits]

    context_blob = {

        'intent' : state.intent,
        'farmer_profile' : state.farmer_profile if state.farmer_profile else None,
        'weather' : state.weather_forcast if state.weather_forcast else None,
        'mandi_prices' : [m.model_dump() for m in state.mandi_prices] if state.mandi_prices else None,
        'pest_diagnosis' : state.pest_diagnosis if state.pest_diagnosis else None,
        'reference_notes' : rag_snippets,
    }

    response = await llm.ainvoke(
        [
            SystemMessage(content= ADVISORY_PROMPT),
            HumanMessage(content = f'{state.detected_language} . Context : \n{json.dumps(context_blob , default=str)}')
        ]
    )

    
    return {
        'advisory_draft' : response.content,
        'rag_snippet_used' : rag_snippets,
        'llm_used' : state.llm_used+1,
    }


def route_image_branch(state : AgriAdvisoryState) -> str:

    
    return 'image_branch' if state.query_type == 'image' else 'advisory_node'



async def image_branch(state : AgriAdvisoryState) -> dict:

    print('From the image branch node')

    '''Send the crop-leaf photo to a vision-capable NIM model for pest/disease ID.

    Args:
        state (AgriAdvisoryState): Current graph state. Reads `image_url`.

    Returns:
        dict: Partial state update with:
            - pest_diagnosis (PestDiagnosis): Label, confidence, and model used.
    '''

    vision_model = ChatNVIDIA(model='meta/llama-3.2-90b-vision-instruct' , temperature=0 , dimesions=1024)

    message = HumanMessage(
        content=[
            {'type' : 'text' , 'text' : 'Identify the crop pestt or disease shown'},
            {'type' : 'image_url' , 'image_url' : {'url' : state.image_url}}
        ]
    )

    structure_model = vision_model.with_structured_output(DiagnosisSchema)

    result = await structure_model.ainvoke([message])

    return {
        'pest_diagnosis' : PestDiagnosis(
            label = result.label,
            confidence = result.confidence,
            model_used='nvidia/vila',
            
        ),
        'llm_used' : state.llm_used+1,
    }


async def confidence_gate(state : AgriAdvisoryState) -> dict:

    print('From the confidence gate node')

    '''Decide whether the advisory can go out as-is, or needs a caveat attached.

    Args:
        state (AgriAdvisoryState): Current graph state. Reads `intent_confidence`,
            `weather_tool_error`, `mandi_tool_error`, `pest_diagnosis`, `caveats`.

    Returns:
        dict: Partial state update with:
            - data_freshness_ok (bool): False if any confidence/error check failed.
            - caveats (list[str]): Existing caveats plus any newly appended ones.
    '''

    caveats = list(state.caveats)
    freshness_ok = True

    if state.intent_confidence is not None and state.intent_confidence < 0.6:
        caveats.append("low_intent_confidence")
        freshness_ok = False

    if state.weather_tool_error:
        caveats.append("weather_unavailable")
        freshness_ok = False

    if state.mandi_tool_error:
        caveats.append("mandi_price_unavailable")
        freshness_ok = False

    # only check pest_diagnosis.confidence when a diagnosis actually exists
    if state.pest_diagnosis is not None and state.pest_diagnosis.confidence < 0.5:
        caveats.append("low_pest_diagnosis_confidence")
        freshness_ok = False

    return {
        "data_freshness_ok": freshness_ok,
        "caveats": caveats,
    }



async def respond_node(state : AgriAdvisoryState , store : BaseStore) -> dict:
    print('From the respond node')

    '''Assemble the final farmer-facing text, appending disclaimers if needed.

    Args:
        state (AgriAdvisoryState): Current graph state. Reads `advisory_draft`,
            `data_freshness_ok`, `caveats`, `detected_language`, and which
            optional data fields are populated (to build `tool_used`).

    Returns:
        dict: Partial state update with:
            - final_response (str): Advisory text plus disclaimer if applicable.
            - tool_used (list[str]): Names of tools/models that contributed data.
    '''

    final = state.advisory_draft or ''

    if not state.data_freshness_ok:
        disclaimer = {
            'en' : '\n\n Note : some data used here may be incomplete or outdated.'
        }.get(state.detected_language , '\n\n Note : Some data may be incomplete.')
        final += disclaimer


    tool_used = []

    if state.weather_forcast:
        tool_used.append('weather_mcp_server')

    if state.mandi_prices:
        tool_used.append('mandi_mcp_server')

    if state.pest_diagnosis:
        tool_used.append('vision_model')

    return {
        'final_response' : final,
        'tool_used' : tool_used,
    }
    

builder.add_node('intent_router_node' , intent_router_node)
builder.add_node('ltm_write' , ltm_write)
builder.add_node('context_node' , context_node)
builder.add_node('weather_mcp_server' , weather_mcp_server)
builder.add_node('mandi_price_mcp_server' , mandi_price_mcp_server)
builder.add_node('advisory_node' , advisory_node)
builder.add_node('image_branch' , image_branch)
builder.add_node('confidence_gate' , confidence_gate)
builder.add_node('respond_node' , respond_node)

builder.add_edge(START , 'intent_router_node')
builder.add_edge('intent_router_node' , 'context_node')

builder.add_conditional_edges(
    'context_node',

    route_after_context,
    {
        'weather_mcp_server' : 'weather_mcp_server',
        'mandi_price_mcp_server' : 'mandi_price_mcp_server',
        'image_branch' : 'image_branch',
    }
)

builder.add_conditional_edges(
    'weather_mcp_server',

    route_after_weather,
    {
        'mandi_price_mcp_server' : 'mandi_price_mcp_server',
        'advisory_node' : 'advisory_node',
    }
)

builder.add_edge('mandi_price_mcp_server' , 'advisory_node')
builder.add_edge('image_branch' , 'advisory_node')
builder.add_edge('advisory_node' , 'confidence_gate')
builder.add_edge('confidence_gate' , 'respond_node')
builder.add_edge('respond_node' , 'ltm_write')
builder.add_edge('ltm_write' , END)


async def main():
    async with (

        AsyncPostgresSaver.from_conn_string(
            DB_URL,
        ) as checkpointer,

        AsyncPostgresStore.from_conn_string(
            DB_URL,
            index={
                "embed": NVIDIAEmbeddings(model="nvidia/llama-nemotron-embed-vl-1b-v2" , dimensions=1024),
                "dims": 1024,
            },
        ) as store

        

    ):
        await checkpointer.setup()
        await store.setup()
    
        # compile() is sync — it just builds the graph, no I/O here
        graph = builder.compile(checkpointer=checkpointer , store=store)

        farmer_profile = FarmerProfile(
            location={
                "lat": 28.63,
                "lon": 79.81,
                "state_name": "Punjab",
                "market": "Lalru APMC",
                "district": "Mohali",
            },
            variety="other",
            grade="Local",
            current_crop="Brinjal",
            prefered_language="English",
        )

        initial_state = {
            "farmer_id": "f1",
            'query_type' : 'text',
            "raw_query": (
                "My Brinjal crop needs water. Based on the weather this week, when should I irrigate next? "
            ),
            "farmer_profile": farmer_profile,
            'detected_language' : 'English',
        }

        result = await graph.ainvoke(initial_state)

        print('====================== FINAL RESULT ======================')

        for key, value in result.items():
            print(f"{key}: {value}\n")

if __name__ == '__main__':
    asyncio.run(main())


