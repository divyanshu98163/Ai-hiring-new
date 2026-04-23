from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote

import httpx


class SupabaseError(RuntimeError):
    """Raised when a Supabase request fails."""


def _to_rest_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class SupabaseRestClient:
    def __init__(self, access_token: str | None = None) -> None:
        supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
        supabase_anon_key = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")

        if not supabase_url:
            raise SupabaseError("Missing environment variable: NEXT_PUBLIC_SUPABASE_URL")
        if not supabase_anon_key:
            raise SupabaseError(
                "Missing environment variable: NEXT_PUBLIC_SUPABASE_ANON_KEY"
            )

        self.base_url = supabase_url.rstrip("/")
        self.access_token = access_token
        self.base_headers = {
            "apikey": supabase_anon_key,
        }
        if access_token:
            self.base_headers["Authorization"] = f"Bearer {access_token}"

    async def auth_user(self) -> dict[str, Any]:
        response = await self._request("GET", "auth/v1/user")
        return response.json()

    async def update_auth_user(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request("PUT", "auth/v1/user", json=payload)
        return response.json()

    async def select(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: dict[str, Any] | None = None,
        order: tuple[str, bool] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {"select": columns}

        for field, raw_value in (filters or {}).items():
            if isinstance(raw_value, tuple):
                operator, value = raw_value
            else:
                operator, value = "eq", raw_value

            if operator == "in":
                joined = ",".join(_to_rest_value(item) for item in value)
                params[field] = f"in.({joined})"
            else:
                params[field] = f"{operator}.{_to_rest_value(value)}"

        if order:
            field, ascending = order
            params["order"] = f"{field}.{'asc' if ascending else 'desc'}"

        if limit:
            params["limit"] = str(limit)

        response = await self._request(
            "GET",
            f"rest/v1/{table}",
            params=params,
            headers={"Accept": "application/json"},
        )
        return response.json()

    async def maybe_single(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: dict[str, Any] | None = None,
        order: tuple[str, bool] | None = None,
    ) -> dict[str, Any] | None:
        rows = await self.select(
            table,
            columns=columns,
            filters=filters,
            order=order,
            limit=1,
        )
        return rows[0] if rows else None

    async def insert(
        self,
        table: str,
        payload: dict[str, Any] | list[dict[str, Any]],
        *,
        on_conflict: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {"on_conflict": on_conflict} if on_conflict else None
        response = await self._request(
            "POST",
            f"rest/v1/{table}",
            params=params,
            json=payload,
            headers={"Prefer": "return=representation"},
        )
        data = response.json()
        return data if isinstance(data, list) else [data]

    async def upsert(
        self,
        table: str,
        payload: dict[str, Any] | list[dict[str, Any]],
        *,
        on_conflict: str | None = None,
    ) -> list[dict[str, Any]]:
        prefer = "resolution=merge-duplicates,return=representation"
        params = {"on_conflict": on_conflict} if on_conflict else None
        response = await self._request(
            "POST",
            f"rest/v1/{table}",
            params=params,
            json=payload,
            headers={"Prefer": prefer},
        )
        data = response.json()
        return data if isinstance(data, list) else [data]

    async def update(
        self,
        table: str,
        payload: dict[str, Any],
        *,
        filters: dict[str, Any],
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        for field, raw_value in filters.items():
            if isinstance(raw_value, tuple):
                operator, value = raw_value
            else:
                operator, value = "eq", raw_value
            params[field] = f"{operator}.{_to_rest_value(value)}"

        response = await self._request(
            "PATCH",
            f"rest/v1/{table}",
            params=params,
            json=payload,
            headers={"Prefer": "return=representation"},
        )
        data = response.json()
        return data if isinstance(data, list) else [data]

    async def upload_file(
        self,
        *,
        bucket: str,
        path: str,
        content: bytes,
        content_type: str,
        upsert: bool = False,
    ) -> None:
        quoted_path = quote(path, safe="/")
        await self._request(
            "POST",
            f"storage/v1/object/{bucket}/{quoted_path}",
            content=content,
            headers={
                "Content-Type": content_type,
                "x-upsert": "true" if upsert else "false",
            },
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json: Any = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        merged_headers = dict(self.base_headers)
        if headers:
            merged_headers.update(headers)

        timeout = httpx.Timeout(60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method,
                f"{self.base_url}/{path}",
                params=params,
                json=json,
                content=content,
                headers=merged_headers,
            )

        if response.is_success:
            return response

        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = response.text

        message = payload.get("message") if isinstance(payload, dict) else None
        error = payload.get("error") if isinstance(payload, dict) else None
        details = error or message or str(payload)
        raise SupabaseError(f"Supabase request failed: {details}")
