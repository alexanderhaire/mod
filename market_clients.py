"""
Multi-Market Clients
====================
Adapters to execute trades on Manifold and Kalshi.
"""

import requests
import time
import os
import uuid

class ManifoldClient:
    def __init__(self, api_key):
        self.key = api_key
        self.base = "https://api.manifold.markets/v0"
        
    def place_bet(self, contract_id, amount, outcome="YES"):
        """Places a bet on Manifold."""
        url = f"{self.base}/bet"
        headers = {"Authorization": f"Key {self.key}"}
        payload = {
            "amount": amount,
            "contractId": contract_id,
            "outcome": outcome
        }
        res = requests.post(url, json=payload, headers=headers)
        if res.status_code == 200:
            return res.json()
        else:
            raise Exception(f"Manifold Error: {res.text}")

class KalshiClient:
    """
    Simplified Kalshi V2 Adapter.
    Requires KALSHI_API_KEY (email) and KALSHI_API_SECRET (password) or proper Keys depending on version.
    For this demo, we assume the user has V2 T&E keys.
    """
    def __init__(self, key, secret):
        self.base = "https://trading-api.kalshi.com/trade-api/v2"
        self.token = None
        self.email = key
        self.password = secret
        self.login()
        
    def login(self):
        url = f"{self.base}/login"
        res = requests.post(url, json={"email": self.email, "password": self.password})
        if res.status_code == 200:
            self.token = res.json().get('token')
        else:
            print(f"Kalshi Login Failed: {res.text}")
            
    def place_order(self, ticker, action, count, side="yes"):
        if not self.token: return None
        url = f"{self.base}/portfolio/orders"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Kalshi Sides: 'yes' or 'no'
        payload = {
            "ticker": ticker,
            "action": action, # 'buy' or 'sell'
            "type": "market", # Use market for simplicity in this bond example
            "count": int(count),
            "side": side,
            "client_order_id": str(uuid.uuid4())
        }
        res = requests.post(url, json=payload, headers=headers)
        return res.json()
