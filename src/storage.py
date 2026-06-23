"""Pluggable artifact storage: local filesystem or Azure Blob Storage.

The same interface backs both, so the model registry is identical whether it
runs against a local folder, Azurite (the Azure Storage emulator), or a real
Azure Storage account. The backend is chosen by configuration.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


class StorageBackend(ABC):
    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]: ...


class LocalStorage(StorageBackend):
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def get_bytes(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return os.path.exists(self._path(key))

    def list(self, prefix: str = "") -> list[str]:
        out = []
        for dirpath, _, filenames in os.walk(self.root):
            for name in filenames:
                rel = os.path.relpath(os.path.join(dirpath, name), self.root).replace("\\", "/")
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)


class AzureBlobStorage(StorageBackend):
    def __init__(self, connection_string: str, container: str):
        from azure.storage.blob import BlobServiceClient

        self.service = BlobServiceClient.from_connection_string(connection_string)
        self.container = container
        try:
            self.service.create_container(container)
        except Exception:
            pass  # already exists

    def _client(self):
        return self.service.get_container_client(self.container)

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client().upload_blob(name=key, data=data, overwrite=True)

    def get_bytes(self, key: str) -> bytes:
        return self._client().download_blob(key).readall()

    def exists(self, key: str) -> bool:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            self._client().get_blob_client(key).get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False

    def list(self, prefix: str = "") -> list[str]:
        return sorted(b.name for b in self._client().list_blobs(name_starts_with=prefix))


def get_storage(settings) -> StorageBackend:
    if settings.storage_backend == "azure_blob" and settings.azure_storage_connection_string:
        return AzureBlobStorage(settings.azure_storage_connection_string, settings.blob_container)
    return LocalStorage(settings.local_registry_dir)
