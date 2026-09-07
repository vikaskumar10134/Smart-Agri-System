from fastmcp import FastMCP
from pydantic import BaseModel , Field
from typing import Optional , Dict , List
from datetime import date
from dotenv import load_dotenv
import os
import httpx


load_dotenv()  # Load environment variables from .env file

class MCPResponse(BaseModel):
    success : bool
    data : Optional[dict] = None
    error : Optional[str] = None
    meta : Optional[dict] = None

RESOURCE_ID = os.getenv('RESOURCE_ID')
DATA_GOV_API_KEY = os.getenv('DATA_GOV_API_KEY')

BASE_API_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

mcp = FastMCP('Mandi-mcp-server')



@mcp.tool()
async def get_product_mandi_price(
    state: str = Field(description="State name, e.g. 'Uttar Pradesh'"),
    district: str = Field(description='District of the place where the mandi is situated'),
    market: str = Field(description="Mandi/market name, e.g. 'Hapur'"),
    commodity: str = Field(description="Crop/commodity name, e.g. 'Onion', 'Wheat'"),
    variety: str = Field(description='Variety of the commodity'),
    grade: str = Field(description='Grade of the crop/commodity'),
    limit: int = Field(20, description="Max number of records to return"),
    ) -> dict:
    """
    Fetch live mandi (market) prices for a commodity from the Agmarknet dataset
    on data.gov.in. Returns min/max/modal price per market, per date.
    """

    if not DATA_GOV_API_KEY:
        return MCPResponse(success=False , error = f'Not provided the DATA Government API key')

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    params = {
        'api-key' : DATA_GOV_API_KEY,
        'format' : 'json',
        'limit' : limit,
    }

    if state:
            params['filters[state]'] = state
    
    if district:
        params['filters[district]'] = district

    if market:
        params['filters[market]'] = market

    if commodity:
        params['filters[commodity]'] = commodity

    if variety:
        params['filters[variety]'] = variety

    if grade:
        params['filters[grade]'] = grade

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:

            response = await client.get(BASE_API_URL , params= params , headers=headers)
            response.raise_for_status()
            result = response.json()
            records = result.get('records')

    except httpx.HTTPStatusError as e:
        return MCPResponse(
            success=False, error=f"Upstream API error: {e.response.status_code}"
        )

    except httpx.RequestError as e:
        return MCPResponse(success=False, error=f"Network error: {str(e)}")
    
    except Exception as e:
        return MCPResponse(success=False, error=f"Unexpected error: {str(e)}")


    if not records:
        return MCPResponse(
            success=True,
            data={"records": []},
            meta={"stale_or_missing": True, "message": "No matching mandi price records found"},
        )

    return MCPResponse(
        success=True,
        data={"records": records},
        meta={
            "count": len(records),
            "commodity": commodity,
            "market": market,
            "fetched_on": str(date.today()),
        },
    )


if __name__ == '__main__':
    mcp.run(transport='stdio')


   