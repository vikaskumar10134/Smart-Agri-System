from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from graph import builder , DB_URL
from state import FarmerProfile

import asyncio


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