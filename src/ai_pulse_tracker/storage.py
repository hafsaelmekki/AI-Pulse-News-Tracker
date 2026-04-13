from __future__ import annotations

import logging
from typing import Iterable, Sequence
from datetime import datetime

from azure.cosmos import ContainerProxy, CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from .config import Settings
from .models import AnalyzedArticle, UpsertResult

LOGGER = logging.getLogger(__name__)


class CosmosRepository:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = CosmosClient(settings.cosmos_endpoint, settings.cosmos_key)
        self._database = self._client.create_database_if_not_exists(id=settings.database_name)
        self._container = self._database.create_container_if_not_exists(
            id=settings.container_name,
            partition_key=PartitionKey(path="/source"),
            offer_throughput=400,
        )

    @property
    def container(self) -> ContainerProxy:
        return self._container

    def upsert_articles(self, articles: Iterable[AnalyzedArticle]) -> UpsertResult:
        persisted_ids: list[str] = []
        created = updated = 0
        for article in articles:
            document = article.to_cosmos_document()
            partition_key = document["source"]
            exists = self._item_exists(document["id"], partition_key)
            if not exists:
                exists = self._delete_legacy_if_exists(
                    document["id"], self._legacy_partitions(article, partition_key)
                )
            self._container.upsert_item(document)
            persisted_ids.append(document["id"])
            if exists:
                updated += 1
            else:
                created += 1
            LOGGER.debug("Upserted document %s", document["id"])
        LOGGER.info("Persisted %s documents (%s new, %s refreshed)", len(persisted_ids), created, updated)
        return UpsertResult(ids=persisted_ids, created=created, updated=updated)

    def _item_exists(self, item_id: str, partition_key: str) -> bool:
        try:
            self._container.read_item(item=item_id, partition_key=partition_key)
            return True
        except CosmosResourceNotFoundError:
            return False

    def _legacy_partitions(self, article: AnalyzedArticle, normalized_partition: str) -> Sequence[str]:
        candidates: list[str] = []
        legacy = article.source
        if legacy and legacy != normalized_partition:
            candidates.append(legacy)
        return candidates

    def _delete_legacy_if_exists(self, item_id: str, partitions: Sequence[str]) -> bool:
        for partition in partitions:
            try:
                self._container.delete_item(item=item_id, partition_key=partition)
                LOGGER.info("Removed legacy document %s in partition %s", item_id, partition)
                return True
            except CosmosResourceNotFoundError:
                continue
        return False

    def load_all(self) -> list[dict]:
        return list(
            self._container.query_items(
                query="SELECT * FROM c",
                enable_cross_partition_query=True,
            )
        )

    def latest_published_at(self) -> datetime | None:
        items = list(
            self._container.query_items(
                query="SELECT TOP 1 c.date FROM c ORDER BY c.date DESC",
                enable_cross_partition_query=True,
            )
        )
        if not items:
            return None
        return self._parse_iso_datetime(items[0].get("date"))

    def _parse_iso_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            if value.endswith("Z"):
                value = value.replace("Z", "+00:00")
            return datetime.fromisoformat(value)
        except ValueError:
            LOGGER.warning("Invalid date stored in Cosmos DB: %s", value)
            return None
