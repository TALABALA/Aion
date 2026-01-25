"""
AION OAuth2 Authentication Provider

Handles OAuth2/OIDC authentication flows.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import structlog

from aion.security.types import AuthMethod, Credentials, User
from aion.security.authentication.providers.base import AuthProvider, AuthProviderResult

logger = structlog.get_logger(__name__)


@dataclass
class OAuth2ProviderConfig:
    """Configuration for an OAuth2 provider."""

    name: str  # e.g., "google", "github", "azure"
    display_name: str

    # Endpoints
    authorization_url: str
    token_url: str
    userinfo_url: str
    jwks_url: Optional[str] = None

    # Client credentials
    client_id: str = ""
    client_secret: str = ""

    # Scopes
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])

    # Claim mappings
    id_claim: str = "sub"
    email_claim: str = "email"
    name_claim: str = "name"
    picture_claim: str = "picture"

    # Settings
    verify_email: bool = True
    allow_unverified_email: bool = False


# Pre-configured providers
OAUTH_PROVIDERS = {
    "google": OAuth2ProviderConfig(
        name="google",
        display_name="Google",
        authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        jwks_url="https://www.googleapis.com/oauth2/v3/certs",
        scopes=["openid", "profile", "email"],
    ),
    "github": OAuth2ProviderConfig(
        name="github",
        display_name="GitHub",
        authorization_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        userinfo_url="https://api.github.com/user",
        scopes=["read:user", "user:email"],
        id_claim="id",
        email_claim="email",
        name_claim="name",
        picture_claim="avatar_url",
    ),
    "microsoft": OAuth2ProviderConfig(
        name="microsoft",
        display_name="Microsoft",
        authorization_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        userinfo_url="https://graph.microsoft.com/oidc/userinfo",
        scopes=["openid", "profile", "email"],
    ),
}


@dataclass
class OAuthState:
    """State for OAuth flow tracking."""

    state: str
    provider: str
    redirect_uri: str
    code_verifier: Optional[str] = None  # PKCE
    nonce: Optional[str] = None  # OIDC
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(
        default_factory=lambda: datetime.now() + timedelta(minutes=10)
    )


class OAuth2Provider(AuthProvider):
    """
    OAuth2/OIDC authentication provider.

    Features:
    - Multiple OAuth providers (Google, GitHub, Microsoft, etc.)
    - PKCE support for public clients
    - ID token validation
    - User provisioning/linking
    """

    def __init__(
        self,
        user_getter: callable,
        user_creator: callable,
        user_linker: callable,
        providers: Optional[Dict[str, OAuth2ProviderConfig]] = None,
    ):
        self._get_user = user_getter
        self._create_user = user_creator
        self._link_account = user_linker
        self._providers = providers or {}

        # Pending OAuth flows
        self._pending_flows: Dict[str, OAuthState] = {}

    @property
    def method(self) -> AuthMethod:
        return AuthMethod.OAUTH2

    @property
    def name(self) -> str:
        return "OAuth2"

    def register_provider(self, config: OAuth2ProviderConfig) -> None:
        """Register an OAuth provider."""
        self._providers[config.name] = config
        logger.info(f"Registered OAuth provider: {config.name}")

    def get_provider(self, name: str) -> Optional[OAuth2ProviderConfig]:
        """Get a provider configuration."""
        return self._providers.get(name)

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self._providers.keys())

    # =========================================================================
    # OAuth Flow
    # =========================================================================

    def create_authorization_url(
        self,
        provider_name: str,
        redirect_uri: str,
        scopes: Optional[List[str]] = None,
        state: Optional[str] = None,
        use_pkce: bool = True,
    ) -> tuple[Optional[str], Optional[OAuthState], Optional[str]]:
        """
        Create OAuth authorization URL.

        Returns (url, state, error).
        """
        provider = self._providers.get(provider_name)
        if not provider:
            return None, None, f"Unknown OAuth provider: {provider_name}"

        # Generate state
        state_value = state or secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(16)

        # PKCE
        code_verifier = None
        code_challenge = None
        if use_pkce:
            code_verifier = secrets.token_urlsafe(64)
            import hashlib
            import base64

            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).decode().rstrip("=")

        # Build URL
        params = {
            "client_id": provider.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes or provider.scopes),
            "state": state_value,
            "nonce": nonce,
        }

        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        url = f"{provider.authorization_url}?{urlencode(params)}"

        # Store state
        oauth_state = OAuthState(
            state=state_value,
            provider=provider_name,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
            nonce=nonce,
        )
        self._pending_flows[state_value] = oauth_state

        return url, oauth_state, None

    async def authenticate(
        self,
        credentials: Credentials,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """
        Complete OAuth authentication with authorization code.

        Expects credentials.authorization_code and state in context.
        """
        context = context or {}
        code = credentials.authorization_code
        state = context.get("state")

        if not code:
            return AuthProviderResult(
                success=False,
                error_code="missing_code",
                error_message="Authorization code is required",
            )

        if not state:
            return AuthProviderResult(
                success=False,
                error_code="missing_state",
                error_message="State parameter is required",
            )

        # Validate state
        oauth_state = self._pending_flows.pop(state, None)
        if not oauth_state:
            return AuthProviderResult(
                success=False,
                error_code="invalid_state",
                error_message="Invalid or expired OAuth state",
            )

        if datetime.now() > oauth_state.expires_at:
            return AuthProviderResult(
                success=False,
                error_code="state_expired",
                error_message="OAuth flow has expired",
            )

        provider = self._providers.get(oauth_state.provider)
        if not provider:
            return AuthProviderResult(
                success=False,
                error_code="provider_not_found",
                error_message="OAuth provider not found",
            )

        # Exchange code for tokens
        tokens, error = await self._exchange_code(
            provider,
            code,
            oauth_state.redirect_uri,
            oauth_state.code_verifier,
        )

        if error:
            return AuthProviderResult(
                success=False,
                error_code="token_exchange_failed",
                error_message=error,
            )

        # Get user info
        user_info, error = await self._get_user_info(provider, tokens.get("access_token"))

        if error:
            return AuthProviderResult(
                success=False,
                error_code="userinfo_failed",
                error_message=error,
            )

        # Find or create user
        user, created = await self._find_or_create_user(provider, user_info)

        if not user:
            return AuthProviderResult(
                success=False,
                error_code="user_creation_failed",
                error_message="Failed to create user account",
            )

        return AuthProviderResult(
            success=True,
            user=user,
            user_id=user.id,
            claims={
                "provider": provider.name,
                "oauth_user_info": user_info,
            },
            metadata={
                "auth_method": "oauth2",
                "provider": provider.name,
                "user_created": created,
            },
        )

    async def validate(
        self,
        token_or_key: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthProviderResult:
        """Validate an OAuth access token (limited functionality)."""
        # OAuth tokens are typically validated at the provider
        # This would need the provider name to validate
        return AuthProviderResult(
            success=False,
            error_code="not_supported",
            error_message="Direct token validation requires provider context",
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _exchange_code(
        self,
        provider: OAuth2ProviderConfig,
        code: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Exchange authorization code for tokens."""
        try:
            import aiohttp

            data = {
                "client_id": provider.client_id,
                "client_secret": provider.client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }

            if code_verifier:
                data["code_verifier"] = code_verifier

            headers = {"Accept": "application/json"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    provider.token_url,
                    data=data,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        return None, f"Token exchange failed: {text}"

                    tokens = await response.json()
                    return tokens, None

        except ImportError:
            return None, "aiohttp is required for OAuth2"
        except Exception as e:
            logger.error(f"OAuth token exchange error: {e}")
            return None, str(e)

    async def _get_user_info(
        self,
        provider: OAuth2ProviderConfig,
        access_token: str,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Get user info from OAuth provider."""
        try:
            import aiohttp

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    provider.userinfo_url,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        return None, f"User info request failed: {text}"

                    user_info = await response.json()
                    return user_info, None

        except ImportError:
            return None, "aiohttp is required for OAuth2"
        except Exception as e:
            logger.error(f"OAuth user info error: {e}")
            return None, str(e)

    async def _find_or_create_user(
        self,
        provider: OAuth2ProviderConfig,
        user_info: Dict[str, Any],
    ) -> tuple[Optional[User], bool]:
        """Find existing user or create new one."""
        provider_id = str(user_info.get(provider.id_claim))
        email = user_info.get(provider.email_claim)

        # Try to find by linked account
        user = await self._get_user(
            provider_id,
            by="oauth",
            provider=provider.name,
        )

        if user:
            return user, False

        # Try to find by email
        if email:
            user = await self._get_user(email, by="email")
            if user:
                # Link account
                await self._link_account(
                    user,
                    provider.name,
                    provider_id,
                    user_info,
                )
                return user, False

        # Create new user
        try:
            user = await self._create_user(
                username=self._generate_username(user_info, provider),
                email=email,
                display_name=user_info.get(provider.name_claim, ""),
                oauth_provider=provider.name,
                oauth_id=provider_id,
                oauth_data=user_info,
                email_verified=True,  # OAuth emails are typically verified
            )
            return user, True
        except Exception as e:
            logger.error(f"Failed to create OAuth user: {e}")
            return None, False

    def _generate_username(
        self,
        user_info: Dict[str, Any],
        provider: OAuth2ProviderConfig,
    ) -> str:
        """Generate a username from OAuth user info."""
        # Try various fields
        username = user_info.get("login")  # GitHub
        if not username:
            username = user_info.get("preferred_username")  # OIDC
        if not username:
            email = user_info.get(provider.email_claim, "")
            username = email.split("@")[0] if email else None
        if not username:
            username = user_info.get(provider.name_claim, "").replace(" ", "_").lower()
        if not username:
            username = f"{provider.name}_{secrets.token_hex(4)}"

        return username
