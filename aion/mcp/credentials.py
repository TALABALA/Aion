"""
AION MCP Credential Management

Secure credential storage and retrieval for MCP servers.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import secrets

import structlog

logger = structlog.get_logger(__name__)


class CredentialManager:
    """
    Secure credential management for MCP servers.

    Provides:
    - Encrypted credential storage
    - Environment variable fallback
    - Secure credential injection
    """

    def __init__(
        self,
        credentials_path: Optional[Path] = None,
        encryption_key: Optional[str] = None,
    ):
        """
        Initialize credential manager.

        Args:
            credentials_path: Path to credentials file
            encryption_key: Encryption key (from env if not provided)
        """
        self.credentials_path = credentials_path or Path("./config/mcp_credentials.json")
        self._encryption_key = encryption_key or os.environ.get("AION_MCP_ENCRYPTION_KEY")

        self._credentials: Dict[str, Dict[str, str]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the credential manager."""
        if self._initialized:
            return

        # Load credentials from file if exists
        if self.credentials_path.exists():
            try:
                self._load_credentials()
            except Exception as e:
                logger.warning("Failed to load credentials file", error=str(e))

        self._initialized = True
        logger.info("Credential manager initialized")

    def _load_credentials(self) -> None:
        """Load credentials from file."""
        with open(self.credentials_path, "r") as f:
            data = json.load(f)

        for cred_id, cred_data in data.get("credentials", {}).items():
            if isinstance(cred_data, dict):
                # Check if encrypted
                if "encrypted" in cred_data and cred_data["encrypted"]:
                    if self._encryption_key:
                        try:
                            decrypted = self._decrypt(cred_data["data"])
                            self._credentials[cred_id] = json.loads(decrypted)
                        except Exception as e:
                            logger.warning(
                                "Failed to decrypt credential",
                                credential_id=cred_id,
                                error=str(e),
                            )
                    else:
                        logger.warning(
                            "Encrypted credential but no key provided",
                            credential_id=cred_id,
                        )
                else:
                    # Plaintext credentials (not recommended for production)
                    self._credentials[cred_id] = cred_data.get("data", cred_data)

    def _save_credentials(self) -> None:
        """Save credentials to file."""
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)

        data = {"credentials": {}}

        for cred_id, cred_data in self._credentials.items():
            if self._encryption_key:
                encrypted = self._encrypt(json.dumps(cred_data))
                data["credentials"][cred_id] = {
                    "encrypted": True,
                    "data": encrypted,
                }
            else:
                data["credentials"][cred_id] = {
                    "encrypted": False,
                    "data": cred_data,
                }

        with open(self.credentials_path, "w") as f:
            json.dump(data, f, indent=2)

    def _encrypt(self, data: str) -> str:
        """Simple encryption using key derivation."""
        if not self._encryption_key:
            return data

        # Simple XOR-based encryption (use proper encryption in production)
        key_bytes = hashlib.sha256(self._encryption_key.encode()).digest()
        data_bytes = data.encode()

        encrypted = bytes(
            b ^ key_bytes[i % len(key_bytes)]
            for i, b in enumerate(data_bytes)
        )

        return base64.b64encode(encrypted).decode()

    def _decrypt(self, data: str) -> str:
        """Simple decryption using key derivation."""
        if not self._encryption_key:
            return data

        key_bytes = hashlib.sha256(self._encryption_key.encode()).digest()
        encrypted_bytes = base64.b64decode(data.encode())

        decrypted = bytes(
            b ^ key_bytes[i % len(key_bytes)]
            for i, b in enumerate(encrypted_bytes)
        )

        return decrypted.decode()

    async def get(self, credential_id: str) -> Optional[Dict[str, str]]:
        """
        Get credentials by ID.

        First checks stored credentials, then falls back to environment variables.

        Args:
            credential_id: Credential identifier

        Returns:
            Dictionary of credential key-value pairs, or None
        """
        # Check stored credentials
        if credential_id in self._credentials:
            return self._credentials[credential_id].copy()

        # Try environment variable fallback
        return self._get_from_env(credential_id)

    def _get_from_env(self, credential_id: str) -> Optional[Dict[str, str]]:
        """Get credentials from environment variables."""
        # Map credential IDs to environment variables
        env_mappings = {
            "postgres_default": {
                "POSTGRES_CONNECTION_STRING": "POSTGRES_CONNECTION_STRING",
                "DATABASE_URL": "DATABASE_URL",
            },
            "github_token": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "GITHUB_TOKEN",
            },
            "slack_token": {
                "SLACK_BOT_TOKEN": "SLACK_BOT_TOKEN",
                "SLACK_TEAM_ID": "SLACK_TEAM_ID",
            },
            "brave_api_key": {
                "BRAVE_API_KEY": "BRAVE_API_KEY",
            },
            "openai_api_key": {
                "OPENAI_API_KEY": "OPENAI_API_KEY",
            },
            "anthropic_api_key": {
                "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
            },
        }

        mapping = env_mappings.get(credential_id, {})
        result = {}

        for env_key, env_var in mapping.items():
            value = os.environ.get(env_var)
            if value:
                result[env_key] = value

        return result if result else None

    async def set(
        self,
        credential_id: str,
        credentials: Dict[str, str],
        persist: bool = True,
    ) -> None:
        """
        Set credentials.

        Args:
            credential_id: Credential identifier
            credentials: Credential key-value pairs
            persist: Whether to save to file
        """
        self._credentials[credential_id] = credentials.copy()

        if persist:
            self._save_credentials()

        logger.info("Credential stored", credential_id=credential_id)

    async def delete(self, credential_id: str, persist: bool = True) -> bool:
        """
        Delete credentials.

        Args:
            credential_id: Credential identifier
            persist: Whether to save to file

        Returns:
            True if credentials were deleted
        """
        if credential_id in self._credentials:
            del self._credentials[credential_id]

            if persist:
                self._save_credentials()

            logger.info("Credential deleted", credential_id=credential_id)
            return True
        return False

    def list_credentials(self) -> list[str]:
        """List all credential IDs."""
        return list(self._credentials.keys())

    def has_credential(self, credential_id: str) -> bool:
        """Check if a credential exists."""
        return (
            credential_id in self._credentials or
            self._get_from_env(credential_id) is not None
        )


class EnvironmentCredentialProvider:
    """
    Credential provider that reads from environment variables.

    Useful for containerized deployments where secrets are
    injected via environment.
    """

    def __init__(self, prefix: str = "MCP_"):
        """
        Initialize environment credential provider.

        Args:
            prefix: Environment variable prefix
        """
        self.prefix = prefix

    def get(self, credential_id: str) -> Optional[Dict[str, str]]:
        """
        Get credentials from environment.

        Looks for environment variables with format:
        {prefix}{CREDENTIAL_ID}_{KEY}

        Args:
            credential_id: Credential identifier

        Returns:
            Dictionary of credential key-value pairs
        """
        result = {}
        env_prefix = f"{self.prefix}{credential_id.upper()}_"

        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                cred_key = key[len(env_prefix):]
                result[cred_key] = value

        return result if result else None

    def list_available(self) -> list[str]:
        """List credential IDs available in environment."""
        credential_ids = set()

        for key in os.environ.keys():
            if key.startswith(self.prefix):
                # Extract credential ID from MCP_{ID}_{KEY}
                parts = key[len(self.prefix):].split("_", 1)
                if len(parts) >= 1:
                    credential_ids.add(parts[0].lower())

        return list(credential_ids)


class VaultCredentialProvider:
    """
    Credential provider that reads from HashiCorp Vault.

    For production environments with centralized secret management.
    """

    def __init__(
        self,
        vault_url: Optional[str] = None,
        vault_token: Optional[str] = None,
        mount_point: str = "secret",
    ):
        """
        Initialize Vault credential provider.

        Args:
            vault_url: Vault server URL
            vault_token: Vault authentication token
            mount_point: KV secrets mount point
        """
        self.vault_url = vault_url or os.environ.get("VAULT_ADDR")
        self.vault_token = vault_token or os.environ.get("VAULT_TOKEN")
        self.mount_point = mount_point

        self._client = None

    async def initialize(self) -> None:
        """Initialize Vault client."""
        try:
            import hvac
            self._client = hvac.Client(
                url=self.vault_url,
                token=self.vault_token,
            )
            logger.info("Vault client initialized")
        except ImportError:
            logger.warning("hvac not installed, Vault integration disabled")

    async def get(self, credential_id: str) -> Optional[Dict[str, str]]:
        """
        Get credentials from Vault.

        Args:
            credential_id: Credential identifier (Vault path)

        Returns:
            Dictionary of credential key-value pairs
        """
        if not self._client:
            return None

        try:
            secret = self._client.secrets.kv.v2.read_secret_version(
                path=f"mcp/{credential_id}",
                mount_point=self.mount_point,
            )
            return secret.get("data", {}).get("data", {})
        except Exception as e:
            logger.warning(
                "Failed to read from Vault",
                credential_id=credential_id,
                error=str(e),
            )
            return None
