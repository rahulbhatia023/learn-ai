from agents.financial_analyst_agent import FinancialAnalystAgent
from common.page import BasePage


class FinancialAnalystAgentPage(BasePage):
    agent = FinancialAnalystAgent
    required_keys = {"OPENAI_API_KEY": "password", "TAVILY_API_KEY": "password"}


FinancialAnalystAgentPage.display()
