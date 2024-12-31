from typing import Annotated, Optional

import yfinance as yf
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel
from pydantic import BaseModel, Field, SkipValidation


class TickerHistoricalDataInfo(BaseModel):
    ticker: str = Field(..., description="The ticker of the stock.")
    start: Optional[str] = Field(
        None, description="The start date (YYYY-MM-DD) for the historical data."
    )
    end: Optional[str] = Field(
        None, description="The end date (YYYY-MM-DD) for the historical data."
    )
    period: Optional[str] = Field(
        "1mo",
        description="The period for the historical data. Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max",
    )
    interval: str = Field(
        "1d",
        description="The interval for the historical data. Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo",
    )
    actions: bool = Field(
        True, description="Include actions such as dividends and stock splits."
    )


class TickerHistoricalDataInfoTool(BaseTool):
    name: str = "get-ticker-historical-data-info"
    description: str = """
        This tool can be used to get the historical data for a ticker such as:
        - open
        - high
        - low
        - close
        - volume
        - dividends
        - stock splits
    """
    args_schema: Annotated[Optional[TypeBaseModel], SkipValidation()] = (
        TickerHistoricalDataInfo
    )
    return_direct: bool = True

    def _run(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: str = "1mo",
        interval: str = "1d",
        actions: bool = True,
    ) -> str:
        dat = yf.Ticker(ticker)
        return dat.history(
            start=start, end=end, period=period, interval=interval, actions=actions
        ).to_string()
