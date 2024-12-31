from typing import Dict, Union, Annotated, Optional

import yfinance as yf
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel
from pydantic import BaseModel, Field, SkipValidation


class TickerPriceInfo(BaseModel):
    ticker: str = Field(..., description="The ticker of the stock.")


class TickerPriceInfoTool(BaseTool):
    name: str = "get-ticker-price-info"
    description: str = """
        This tool can be used to get the price information for a ticker such as:
        - currency
        - day high
        - day low
        - exchange
        - fifty day average
        - last price
        - last volume
        - market cap
        - open
        - previous close
        - quote type
        - regular market previous close
        - shares
        - ten day average volume
        - three month average volume
        - timezone
        - two hundred day average
        - year change
        - year high
        - year low
    """
    args_schema: Annotated[Optional[TypeBaseModel], SkipValidation()] = TickerPriceInfo
    return_direct: bool = True

    def _run(self, ticker: str) -> Union[Dict, str]:
        dat = yf.Ticker(ticker)
        return dat.fast_info.items()
