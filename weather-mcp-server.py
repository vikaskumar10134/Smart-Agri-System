from pydantic import BaseModel , Field
from typing import Optional , Literal , Any
from fastmcp import FastMCP
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

import os

import httpx

API_KEY = os.getenv('WEATHER_API_KEY')
BASE_URL = 'https://api.weatherapi.com/v1/forecast.json'

mcp = FastMCP('Weather-mcp-server')

class MCPResponse(BaseModel):
    success : bool
    result : Optional[dict] = None
    error : Optional[str] = None
    meta : Optional[dict] = None



def helper_function(result: dict[str, Any], days: int) -> dict[str, Any]:
    """
    Convert the large Weather API response into a compact day-based response.

    Parameters
    ----------
    result : dict
        Raw response returned by the Weather API.

    days : int
        Number of forecast days requested.

    Returns
    -------
    dict
        Compact weather information suitable for an MCP response.
    """

    # -----------------------------
    # Extract location information
    # -----------------------------
    location_data = result.get("location", {})

    location = {
        "name": location_data.get("name"),
        "region": location_data.get("region"),
        "country": location_data.get("country"),
        "localtime": location_data.get("localtime"),
    }

    # -----------------------------
    # Extract current weather
    # -----------------------------
    current_data = result.get("current", {})

    current = {
        "temperature_c": current_data.get("temp_c"),
        "condition": current_data.get("condition", {}).get("text"),
        "humidity": current_data.get("humidity"),
        "is_raining": bool(current_data.get("will_it_rain", 0)),
        "precip_mm": current_data.get("precip_mm"),
    }

    # -----------------------------
    # Extract daily forecasts
    # IMPORTANT:
    # We intentionally ignore the
    # "hour" array to save tokens.
    # -----------------------------
    forecast_days = result.get("forecast", {}).get("forecastday", [])

    forecast = []

    for forecast_day in forecast_days[:days]:
        day_data = forecast_day.get("day", {})

        forecast.append(
            {
                "date": forecast_day.get("date"),
                "condition": day_data.get("condition", {}).get("text"),
                "min_temp_c": day_data.get("mintemp_c"),
                "max_temp_c": day_data.get("maxtemp_c"),
                "avg_temp_c": day_data.get("avgtemp_c"),
                "will_rain": bool(day_data.get("daily_will_it_rain", 0)),
                "chance_of_rain": day_data.get("daily_chance_of_rain", 0),
                "total_precip_mm": day_data.get("totalprecip_mm", 0),
                "humidity": day_data.get("avghumidity"),
            }
        )

    # -----------------------------
    # Return compact result
    # -----------------------------
    return {
        "location": location,
        "current": current,
        "forecast": forecast,
    }




@mcp.tool()
async def weather_forcasting(

    lattitude : float = Field(... , description='Lattitude of the Farmer field'),
    longitude : float = Field(... , description='Longitude of the Farmer field'),
    days : int = Field(default=4 , description='No of days for forcasting including the today'),
    aqi : Literal['yes' , 'no'] = Field(default='no' , description='AQI needed'),
    alert : Literal['yes' , 'no'] = Field(default= 'yes' , description='For alerting'),
    
    ) -> MCPResponse:

    'Fetch live rainfall/temperature forecast for irrigation timing or crop selection decisions.'

    params = {
        'key' : API_KEY,
        'q' : f'{lattitude} , {longitude}', 
        'days' : days,
        'aqi' : aqi,
        'alert' : alert,
    }

    try:

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(BASE_URL , params=params)
            response.raise_for_status()
            result = response.json()

            parsed = helper_function(result, days)

            return MCPResponse(
                success=True,
                result=parsed,
                meta={
                    'source': 'weatherapi.com',
                    'generated_at': datetime.utcnow().isoformat(),
                    'requested_days': days,
                },
            )
            
    except httpx.HTTPStatusError as e:
            return MCPResponse(
                success=False, error=f'Upstream API error: {e.response.status_code}'
            ).model_dump()
    
    except httpx.RequestError as e:
        return MCPResponse(success=False, error=f'Network error: {str(e)}').model_dump()
    
    except Exception as e:
        return MCPResponse(success=False, error=f'Unexpected error: {str(e)}').model_dump()

    
if __name__ == '__main__':
    mcp.run(transport='stdio')