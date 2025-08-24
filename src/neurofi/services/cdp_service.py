# src/neurofi/services/cdp_service.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional, Tuple

import requests
from requests import Response
from cdp.auth.utils.jwt import generate_jwt, JwtOptions


class CdpApiError(Exception):
    """Raised for non-2xx responses from the CDP REST API."""
    def __init__(self, status_code: int, message: str, response_text: str = ""):
        super().__init__(f"[HTTP {status_code}] {message}")
        self.status_code = status_code
        self.response_text = response_text


class CdpService:
    """
    Minimal Coinbase Developer Platform REST client for NeuroFi.

    - Auth: short-lived JWT generated per request using KEY_NAME/KEY_SECRET.
    - Base host: api.cdp.coinbase.com
    - Example:
        svc = CdpService()
        balances = svc.list_evm_token_balances("base-sepolia", "0x...")
    """

    DEFAULT_HOST = "api.cdp.coinbase.com"
    SUPPORTED_NETWORKS = {"base", "base-sepolia", "ethereum"}

    def __init__(
        self,
        key_name: Optional[str] = None,
        key_secret: Optional[str] = None,
        host: str = DEFAULT_HOST,
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ):
        self.key_name = key_name or os.getenv("KEY_NAME")
        self.key_secret = key_secret or os.getenv("KEY_SECRET")
        self.host = host
        self.timeout = timeout
        self.session = session or requests.Session()

        missing = [k for k, v in {"KEY_NAME": self.key_name, "KEY_SECRET": self.key_secret}.items() if not v]
        if missing:
            raise ValueError(f"Missing required credentials: {', '.join(missing)}")

    # ---------- Internal helpers ----------

    def _generate_jwt(self, method: str, path: str, expires_in: int = 120) -> str:
        """
        Generate a short-lived JWT tied to this exact (method, host, path).
        """
        method = method.upper()
        if not path.startswith("/"):
            path = "/" + path

        token = generate_jwt(
            JwtOptions(
                api_key_id=self.key_name,
                api_key_secret=self.key_secret,
                request_method=method,
                request_host=self.host,
                request_path=path,
                expires_in=expires_in,  # CDP default is 120 seconds
            )
        )
        return token

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> Response:
        """
        Make an authenticated request with a fresh JWT for the exact route.
        """
        method = method.upper()
        if not path.startswith("/"):
            path = "/" + path

        jwt_token = self._generate_jwt(method, path)
        url = f"https://{self.host}{path}"
        all_headers = {"Authorization": f"Bearer {jwt_token}"}
        if headers:
            all_headers.update(headers)

        resp = self.session.request(
            method=method,
            url=url,
            headers=all_headers,
            params=params,
            json=json_body,
            timeout=timeout or self.timeout,
        )
        return resp

    @staticmethod
    def _parse_or_raise(resp: Response) -> Dict[str, Any]:
        """
        Parse JSON or raise CdpApiError with as much context as possible.
        """
        content = resp.text or ""
        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except Exception:
                # Should rarely happen for CDP, but keep it defensive
                return {"raw": content}

        # Try to pull structured error message if available
        message = ""
        try:
            payload = resp.json()
            message = payload.get("message") or payload.get("error") or ""
        except Exception:
            pass

        if not message:
            message = content[:500]  # avoid overly long error messages

        raise CdpApiError(resp.status_code, message, response_text=content)

    # ---------- Public endpoints ----------

    def list_evm_token_balances(
        self,
        network: str,
        address: str,
        *,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Lists ERC-20 + native token balances for an EVM address on a given network.

        Returns the raw API JSON with keys like: 'balances', 'nextPageToken'.
        """
        if network not in self.SUPPORTED_NETWORKS:
            raise ValueError(f"Unsupported network '{network}'. Choose one of: {sorted(self.SUPPORTED_NETWORKS)}")

        path = f"/platform/v2/data/evm/token-balances/{network}/{address}"
        params = {}
        if page_size is not None:
            params["pageSize"] = int(page_size)
        if page_token:
            params["pageToken"] = page_token

        resp = self._request("GET", path, params=params)
        return self._parse_or_raise(resp)

    # You can add more helpers here:
    # - get_evm_account(...)
    # - list_transactions(...)
    # - token_metadata(...)
