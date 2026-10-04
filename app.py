import asyncio
import queue
import threading
import traceback
import uuid

import streamlit as st
from psycopg import OperationalError
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.postgres import AsyncPostgresStore

# graph.py sets the Windows selector loop policy on import, so keep this
# import above the loop creation below
from graph import builder, DB_URL
from state import FarmerProfile


NODE_LABELS = {
    "intent_router_node": "Understanding your question",
    "context_node": "Loading your farm profile",
    "weather_mcp_server": "Fetching weather forecast",
    "mandi_price_mcp_server": "Checking mandi prices",
    "image_branch": "Analysing crop image",
    "advisory_node": "Writing advisory",
    "confidence_gate": "Checking data quality",
    "respond_node": "Preparing final answer",
    "ltm_write": "Saving to memory",
}

MARKS = {"run": "→", "ok": "✓", "warn": "⚠"}

ALLOWED_TYPES = [
    ("state", "FarmerProfile"),
    ("state", "WeatherForecast"),
    ("state", "MandiPricePoint"),
    ("state", "PestDiagnosis"),
]

RUN_TIMEOUT = 180

st.set_page_config(page_title="Smart Agri Advisory", layout="centered")


@st.cache_resource
def get_runtime():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    async def setup():
        pool = AsyncConnectionPool(
            conninfo=DB_URL,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            check=AsyncConnectionPool.check_connection,
        )
        await pool.open()

        serde = JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES)
        checkpointer = AsyncPostgresSaver(pool, serde=serde)
        await checkpointer.setup()

        index = {
            "embed": NVIDIAEmbeddings(
                model="nvidia/llama-nemotron-embed-vl-1b-v2", dimensions=1024
            ),
            "dims": 1024,
        }
        store = AsyncPostgresStore(pool, index=index)
        await store.setup()

        graph = builder.compile(checkpointer=checkpointer, store=store)
        return graph, pool

    try:
        graph, pool = asyncio.run_coroutine_threadsafe(setup(), loop).result(timeout=60)
    except Exception:
        loop.call_soon_threadsafe(loop.stop)
        raise
    return loop, graph, pool


def reset_runtime():
    try:
        loop, graph, pool = get_runtime()
        asyncio.run_coroutine_threadsafe(pool.close(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
    except Exception:
        pass
    get_runtime.clear()


def run_graph(loop, graph, payload, config):
    q = queue.Queue()

    async def worker():
        try:
            async for mode, data in graph.astream(
                payload, config=config, stream_mode=["debug", "messages"]
            ):
                q.put((mode, data))
            snap = await graph.aget_state(config)
            q.put(("final", snap.values))
        except Exception as e:
            q.put(("error", (e, traceback.format_exc())))
        finally:
            q.put(None)

    future = asyncio.run_coroutine_threadsafe(worker(), loop)

    while True:
        try:
            item = q.get(timeout=RUN_TIMEOUT)
        except queue.Empty:
            future.cancel()
            err = TimeoutError(f"No response from the agent for {RUN_TIMEOUT} seconds")
            yield ("error", (err, ""))
            break
        if item is None:
            break
        yield item


def is_connection_error(e):
    text = str(e).lower()
    return (
        isinstance(e, OperationalError)
        or "connection is closed" in text
        or "connection closed" in text
        or "server closed the connection" in text
        or ("pool" in text and "closed" in text)
    )


def friendly_error(e):
    if isinstance(e, TimeoutError):
        return "The agent took too long to respond. Please try again."
    if is_connection_error(e):
        return "Lost the connection to the database and could not reconnect."
    if "api key" in str(e).lower() or "401" in str(e) or "403" in str(e):
        return "The model API rejected the request. Check your NVIDIA API key."
    return "Something went wrong while generating the advisory."


def get_field(values, key):
    if isinstance(values, dict):
        return values.get(key)
    return getattr(values, key, None)


def chunk_text(chunk):
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def result_has_error(result):
    for item in result or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            key, value = item
            if str(key).endswith("_error") and value:
                return True
    return False


def debug_info(values):
    keys = [
        "intent",
        "intent_confidence",
        "caveats",
        "weather_tool_error",
        "mandi_tool_error",
        "tool_used",
    ]
    info = {k: get_field(values, k) for k in keys}
    info["weather_data_found"] = bool(get_field(values, "weather_forcast"))
    info["mandi_data_found"] = bool(get_field(values, "mandi_prices"))
    return info


def make_payload(query, cfg):
    profile = FarmerProfile(
        location={
            "lat": cfg["lat"],
            "lon": cfg["lon"],
            "state_name": cfg["state"],
            "district": cfg["district"],
            "market": cfg["market"],
        },
        variety=cfg["variety"],
        grade=cfg["grade"],
        current_crop=cfg["crop"],
        prefered_language=cfg["language"],
    )
    # clear everything a previous turn may have left in the thread state
    return {
        "farmer_id": cfg["farmer_id"],
        "query_type": "text",
        "raw_query": query,
        "detected_language": cfg["language"],
        "farmer_profile": profile,
        "intent": None,
        "intent_confidence": None,
        "weather_forcast": None,
        "weather_tool_error": None,
        "mandi_prices": None,
        "mandi_tool_error": None,
        "pest_diagnosis": None,
        "image_attach": False,
        "image_url": None,
        "data_freshness_ok": True,
        "caveats": [],
        "advisory_draft": None,
        "rag_snippet_used": None,
        "final_response": None,
        "tool_used": [],
    }


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Farm details")
    farmer_id = st.text_input("Farmer ID", value="f1")
    language = st.selectbox("Language", ["English", "Hindi"])
    crop = st.text_input("Current crop", value="Brinjal")
    variety = st.text_input("Variety", value="other")
    grade = st.text_input("Grade", value="Local")
    state_name = st.text_input("State", value="Punjab")
    district = st.text_input("District", value="Mohali")
    market = st.text_input("Market", value="Lalru APMC")
    lat = st.number_input("Latitude", value=28.63, min_value=-90.0, max_value=90.0)
    lon = st.number_input("Longitude", value=79.81, min_value=-180.0, max_value=180.0)

    show_debug = st.checkbox("Show debug info")

    if st.button("Start over", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    if st.button("Reconnect database", use_container_width=True):
        reset_runtime()
        st.rerun()

cfg = {
    "farmer_id": farmer_id,
    "language": language,
    "crop": crop,
    "variety": variety,
    "grade": grade,
    "state": state_name,
    "district": district,
    "market": market,
    "lat": lat,
    "lon": lon,
}

st.title("Smart Agri Advisory")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Ask about your crop, weather or mandi prices")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        status = st.status("Starting...", expanded=True)
        board = status.empty()
        answer = st.empty()

        try:
            payload = make_payload(query, cfg)
        except Exception as e:
            status.update(label="Invalid farm details", state="error")
            st.error(f"Please check the sidebar values: {e}")
            st.stop()

        steps = []
        buffer = ""
        final_text = None
        final_values = None
        error = None

        def draw():
            lines = []
            for name, state in steps:
                lines.append(f"- {MARKS[state]} {NODE_LABELS.get(name, name)}")
            board.markdown("\n".join(lines))

        for attempt in range(2):
            steps.clear()
            buffer = ""
            final_text = None
            final_values = None
            error = None

            try:
                loop, graph, _ = get_runtime()
            except Exception as e:
                error = (e, traceback.format_exc())
                break

            for mode, data in run_graph(loop, graph, payload, config):
                if mode == "debug":
                    kind = data.get("type")
                    body = data.get("payload", {})
                    name = body.get("name")
                    if kind == "task":
                        steps.append([name, "run"])
                        status.update(label=NODE_LABELS.get(name, name) + "...")
                        draw()
                    elif kind == "task_result":
                        failed = result_has_error(body.get("result"))
                        for s in steps:
                            if s[0] == name and s[1] == "run":
                                s[1] = "warn" if failed else "ok"
                        draw()

                elif mode == "messages":
                    chunk, meta = data
                    if meta.get("langgraph_node") == "advisory_node":
                        buffer += chunk_text(chunk)
                        answer.markdown(buffer + "▌")

                elif mode == "final":
                    final_values = data
                    final_text = get_field(data, "final_response")

                elif mode == "error":
                    error = data

            if error and attempt == 0 and is_connection_error(error[0]):
                status.update(label="Connection lost, reconnecting...")
                answer.empty()
                reset_runtime()
                continue
            break

        if error:
            exc, tb = error
            status.update(label="Failed", state="error", expanded=False)
            answer.empty()
            st.error(friendly_error(exc))
            with st.expander("Error details"):
                st.code(tb or repr(exc))
        else:
            final_text = final_text or buffer
            if final_text:
                answer.markdown(final_text)
                status.update(label="Done", state="complete", expanded=False)
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_text}
                )
            else:
                status.update(label="No answer produced", state="error", expanded=False)
                st.warning("The agent finished but did not return an answer. Try rephrasing your question.")

            if show_debug and final_values is not None:
                with st.expander("Debug info"):
                    st.json(debug_info(final_values))