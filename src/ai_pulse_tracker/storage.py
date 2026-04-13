from __future__ import annotations

import logging
import uuid
from typing import Iterable

from azure.cosmos import ContainerProxy, CosmosClient, PartitionKey

from .config import Settings
from .models import AnalyzedArticle

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

    def upsert_articles(self, articles: Iterable[AnalyzedArticle]) -> list[str]:
        persisted_ids: list[str] = []
        for article in articles:
            document = article.to_cosmos_document()
            document.setdefault("id", str(uuid.uuid4()))
            self._container.upsert_item(document)
            persisted_ids.append(document["id"])
            LOGGER.debug("Upserted document %s", document["id"])
        LOGGER.info("Persisted %s documents", len(persisted_ids))
        return persisted_ids

    def load_all(self) -> list[dict]:
        return list(
            self._container.query_items(
                query="SELECT * FROM c",
                enable_cross_partition_query=True,
            )
        )
