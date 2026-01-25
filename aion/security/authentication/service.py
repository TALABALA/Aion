"""
AION Authentication Service

Central authentication orchestrator supporting multiple providers.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from aion.security.types import (
    AuthMethod,
    AuthToken,
    AuthenticationResult,
    AuthenticationResultStatus,
    Credentials,
    MFAConfig,
    MFAMethod,
    SecurityContext,
    Session,
    TokenType,
    User,
    UserStatus,
)
from aion.security.authentication.tokens import TokenManager
from aion.security.authentication.session import SessionManager, SessionConfig
from aion.security.authentication.providers.base import AuthProvider, AuthProviderResult
from aion.security.authentication.providers.api_key import APIKeyProvider
from aion.security.authentication.providers.jwt_provider import JWTProvider
from aion.security.authentication.providers.password import PasswordProvider, PasswordPolicy
from aion.security.authentication.providers.oauth import OAuth2Provider

logger = structlog.get_logger(__name__)


class AuthenticationService:
    """
    Central authentication service.

    Features:
    - Multiple authentication providers
    - MFA support
    - Session management
    - Token lifecycle management
    - User management
    - Security event emission
    """

    def __init__(
        self,
        jwt_secret: Optional[str] = None,
        jwt_expiry_minutes: int = 60,
        password_policy: Optional[PasswordPolicy] = None,
        session_config: Optional[SessionConfig] = None,
    ):
        # Core managers
        self.tokens = TokenManager(jwt_secret=jwt_secret)
        self.sessions = SessionManager(config=session_config)

        # User storage (in production, use database)
        self._users: Dict[str, User] = {}
        self._users_by_email: Dict[str, str] = {}
        self._users_by_username: Dict[str, str] = {}
        self._oauth_links: Dict[str, str] = {}  # provider:id -> user_id

        # Providers
        self._providers: Dict[AuthMethod, AuthProvider] = {}

        # MFA state (pending verifications)
        self._mfa_challenges: Dict[str, Dict[str, Any]] = {}

        # Password policy
        self._password_policy = password_policy or PasswordPolicy()

        # Event handlers
        self._event_handlers: List[Callable] = []

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize authentication service."""
        if self._initialized:
            return

        logger.info("Initializing Authentication Service")

        await self.tokens.initialize()
        await self.sessions.initialize()

        # Register default providers
        self._register_default_providers()

        # Create default admin if no users exist
        if not self._users:
            await self._create_default_admin()

        self._initialized = True
        logger.info("Authentication Service initialized")

    async def shutdown(self) -> None:
        """Shutdown authentication service."""
        await self.tokens.shutdown()
        await self.sessions.shutdown()
        self._initialized = False

    def _register_default_providers(self) -> None:
        """Register default authentication providers."""
        # API Key provider
        api_key_provider = APIKeyProvider(
            token_manager=self.tokens,
            user_getter=self._get_user_by_id,
        )
        self._providers[AuthMethod.API_KEY] = api_key_provider

        # JWT provider
        jwt_provider = JWTProvider(
            token_manager=self.tokens,
            user_getter=self._get_user_by_id,
        )
        self._providers[AuthMethod.JWT] = jwt_provider

        # Password provider
        password_provider = PasswordProvider(
            user_getter=self._get_user_flexible,
            user_updater=self._update_user,
            password_policy=self._password_policy,
        )
        self._providers[AuthMethod.BASIC] = password_provider

        # OAuth provider
        oauth_provider = OAuth2Provider(
            user_getter=self._get_user_flexible,
            user_creator=self._create_oauth_user,
            user_linker=self._link_oauth_account,
        )
        self._providers[AuthMethod.OAUTH2] = oauth_provider

    async def _create_default_admin(self) -> None:
        """Create default admin user."""
        try:
            admin = await self.create_user(
                username="admin",
                email="admin@aion.local",
                password="admin123!",  # Change in production!
                roles=["admin"],
            )
            logger.warning(
                "Created default admin user. CHANGE THE PASSWORD IMMEDIATELY!",
                username="admin",
            )
        except Exception as e:
            logger.error(f"Failed to create default admin: {e}")

    # =========================================================================
    # Authentication
    # =========================================================================

    async def authenticate(
        self,
        credentials: Credentials,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> AuthenticationResult:
        """
        Authenticate with credentials.

        Returns AuthenticationResult with tokens on success.
        """
        if not credentials.is_valid():
            return AuthenticationResult(
                status=AuthenticationResultStatus.FAILURE,
                error_code="invalid_credentials",
                error_message="Invalid or incomplete credentials",
            )

        # Get provider for method
        provider = self._providers.get(credentials.method)
        if not provider:
            return AuthenticationResult(
                status=AuthenticationResultStatus.FAILURE,
                error_code="unsupported_method",
                error_message=f"Authentication method not supported: {credentials.method.value}",
            )

        # Build context
        context = {
            "ip_address": ip_address,
            "user_agent": user_agent,
            "device_id": device_id,
        }

        # Authenticate
        result = await provider.authenticate(credentials, context)

        if not result.success:
            await self._emit_event(
                "auth_failed",
                {
                    "method": credentials.method.value,
                    "error": result.error_code,
                    "ip_address": ip_address,
                },
            )

            return AuthenticationResult(
                status=self._map_error_status(result),
                error_code=result.error_code,
                error_message=result.error_message,
                retry_after=30 if result.account_locked else None,
            )

        # Handle MFA requirement
        if result.mfa_required:
            return AuthenticationResult(
                status=AuthenticationResultStatus.MFA_REQUIRED,
                mfa_token=result.mfa_token,
                mfa_methods=result.mfa_methods,
            )

        # Handle password expiration
        if result.password_expired:
            return AuthenticationResult(
                status=AuthenticationResultStatus.PASSWORD_EXPIRED,
                error_code="password_expired",
                error_message="Your password has expired",
            )

        # Success - create tokens
        return await self._create_auth_response(
            result.user,
            ip_address,
            user_agent,
            device_id,
            credentials.method,
        )

    async def verify_mfa(
        self,
        mfa_token: str,
        code: str,
        method: MFAMethod,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> AuthenticationResult:
        """Verify MFA code and complete authentication."""
        challenge = self._mfa_challenges.get(mfa_token)
        if not challenge:
            return AuthenticationResult(
                status=AuthenticationResultStatus.FAILURE,
                error_code="invalid_mfa_token",
                error_message="Invalid or expired MFA token",
            )

        if datetime.now() > challenge["expires_at"]:
            del self._mfa_challenges[mfa_token]
            return AuthenticationResult(
                status=AuthenticationResultStatus.FAILURE,
                error_code="mfa_expired",
                error_message="MFA verification has expired",
            )

        user = await self._get_user_by_id(challenge["user_id"])
        if not user:
            return AuthenticationResult(
                status=AuthenticationResultStatus.FAILURE,
                error_code="user_not_found",
                error_message="User not found",
            )

        # Verify code based on method
        verified = await self._verify_mfa_code(user, code, method)

        if not verified:
            challenge["attempts"] = challenge.get("attempts", 0) + 1
            if challenge["attempts"] >= 3:
                del self._mfa_challenges[mfa_token]
                return AuthenticationResult(
                    status=AuthenticationResultStatus.FAILURE,
                    error_code="mfa_max_attempts",
                    error_message="Too many failed MFA attempts",
                )

            return AuthenticationResult(
                status=AuthenticationResultStatus.FAILURE,
                error_code="invalid_mfa_code",
                error_message="Invalid verification code",
            )

        # MFA verified - complete authentication
        del self._mfa_challenges[mfa_token]

        await self._emit_event(
            "mfa_success",
            {"user_id": user.id, "method": method.value},
        )

        return await self._create_auth_response(
            user,
            ip_address,
            user_agent,
            device_id,
            AuthMethod.BASIC,
        )

    async def refresh_tokens(
        self,
        refresh_token: str,
    ) -> AuthenticationResult:
        """Refresh access token using refresh token."""
        provider = self._providers.get(AuthMethod.JWT)
        if not isinstance(provider, JWTProvider):
            return AuthenticationResult(
                status=AuthenticationResultStatus.FAILURE,
                error_code="refresh_not_supported",
                error_message="Token refresh not supported",
            )

        result = await provider.refresh(refresh_token)

        if not result.success:
            return AuthenticationResult(
                status=AuthenticationResultStatus.FAILURE,
                error_code=result.error_code,
                error_message=result.error_message,
            )

        return AuthenticationResult(
            status=AuthenticationResultStatus.SUCCESS,
            access_token=result.claims["access_token"],
            refresh_token=result.claims["refresh_token"],
            expires_in=result.claims.get("expires_in", 3600),
            authenticated_at=datetime.now(),
        )

    async def logout(
        self,
        session_id: Optional[str] = None,
        access_token: Optional[str] = None,
        all_sessions: bool = False,
        user_id: Optional[str] = None,
    ) -> bool:
        """Logout user by terminating sessions and revoking tokens."""
        if session_id:
            await self.sessions.terminate_session(session_id, reason="user_logout")

        if access_token:
            # Revoke the access token
            token_hash = self.tokens._hash_token(access_token)
            token_id = self.tokens._token_hash_index.get(token_hash)
            if token_id:
                await self.tokens.revoke_token(token_id, reason="logout")

        if all_sessions and user_id:
            await self.sessions.terminate_all_sessions(user_id, reason="logout_all")
            await self.tokens.revoke_all_user_tokens(user_id, reason="logout_all")

        return True

    # =========================================================================
    # Token Management
    # =========================================================================

    async def create_api_key(
        self,
        user: User,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
    ) -> Tuple[str, AuthToken]:
        """Create an API key for a user."""
        result = await self.tokens.create_api_key(
            user_id=user.id,
            tenant_id=user.tenant_id,
            name=name,
            scopes=scopes or [],
            expires_in_days=expires_in_days,
            allowed_ips=allowed_ips,
        )

        await self._emit_event(
            "api_key_created",
            {"user_id": user.id, "token_id": result.token_record.token_id, "name": name},
        )

        return result.raw_token, result.token_record

    async def revoke_api_key(
        self,
        token_id: str,
        revoked_by: Optional[str] = None,
    ) -> bool:
        """Revoke an API key."""
        result = await self.tokens.revoke_token(
            token_id,
            reason="user_revoked",
            revoked_by=revoked_by,
        )

        if result:
            await self._emit_event(
                "api_key_revoked",
                {"token_id": token_id, "revoked_by": revoked_by},
            )

        return result

    async def get_user_api_keys(self, user_id: str) -> List[AuthToken]:
        """Get all API keys for a user."""
        return await self.tokens.get_active_api_keys(user_id)

    # =========================================================================
    # User Management
    # =========================================================================

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        display_name: Optional[str] = None,
        require_email_verification: bool = True,
    ) -> User:
        """Create a new user."""
        # Validate uniqueness
        if email.lower() in self._users_by_email:
            raise ValueError(f"Email already exists: {email}")
        if username.lower() in self._users_by_username:
            raise ValueError(f"Username already exists: {username}")

        # Validate password
        is_valid, error = self._password_policy.validate(password)
        if not is_valid:
            raise ValueError(f"Invalid password: {error}")

        # Hash password
        provider = self._providers.get(AuthMethod.BASIC)
        password_hash = provider._hash_password(password) if provider else None

        # Create user
        user = User(
            username=username,
            email=email.lower(),
            display_name=display_name or username,
            password_hash=password_hash,
            password_changed_at=datetime.now(),
            password_expires_at=datetime.now() + timedelta(days=self._password_policy.max_age_days),
            roles=roles or ["user"],
            tenant_id=tenant_id,
            status=UserStatus.PENDING_VERIFICATION if require_email_verification else UserStatus.ACTIVE,
            email_verified=not require_email_verification,
        )

        # Store
        self._users[user.id] = user
        self._users_by_email[email.lower()] = user.id
        self._users_by_username[username.lower()] = user.id

        await self._emit_event(
            "user_created",
            {"user_id": user.id, "username": username, "email": email},
        )

        logger.info("User created", user_id=user.id, username=username)

        return user

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self._users.get(user_id)

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        user_id = self._users_by_email.get(email.lower())
        return self._users.get(user_id) if user_id else None

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        user_id = self._users_by_username.get(username.lower())
        return self._users.get(user_id) if user_id else None

    async def update_user(self, user: User) -> None:
        """Update a user."""
        user.updated_at = datetime.now()
        self._users[user.id] = user

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        user = self._users.pop(user_id, None)
        if not user:
            return False

        self._users_by_email.pop(user.email.lower(), None)
        self._users_by_username.pop(user.username.lower(), None)

        # Revoke all tokens
        await self.tokens.revoke_all_user_tokens(user_id, reason="user_deleted")

        # Terminate all sessions
        await self.sessions.terminate_all_sessions(user_id, reason="user_deleted")

        return True

    async def verify_email(self, user_id: str) -> bool:
        """Mark user's email as verified."""
        user = self._users.get(user_id)
        if not user:
            return False

        user.email_verified = True
        user.email_verified_at = datetime.now()
        if user.status == UserStatus.PENDING_VERIFICATION:
            user.status = UserStatus.ACTIVE

        return True

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> Tuple[bool, str]:
        """Change user's password."""
        user = self._users.get(user_id)
        if not user:
            return False, "User not found"

        provider = self._providers.get(AuthMethod.BASIC)
        if not isinstance(provider, PasswordProvider):
            return False, "Password authentication not available"

        return await provider.change_password(user, current_password, new_password)

    # =========================================================================
    # MFA Management
    # =========================================================================

    async def setup_mfa(
        self,
        user_id: str,
        method: MFAMethod,
    ) -> Dict[str, Any]:
        """Setup MFA for a user."""
        user = self._users.get(user_id)
        if not user:
            raise ValueError("User not found")

        if method == MFAMethod.TOTP:
            # Generate TOTP secret
            secret = secrets.token_hex(20)
            user.mfa.totp_secret = secret
            user.mfa.methods.append(MFAMethod.TOTP)

            # Generate provisioning URI
            import base64

            secret_b32 = base64.b32encode(bytes.fromhex(secret)).decode()
            uri = f"otpauth://totp/AION:{user.email}?secret={secret_b32}&issuer=AION"

            return {
                "secret": secret_b32,
                "uri": uri,
                "method": "totp",
            }

        elif method == MFAMethod.BACKUP_CODES:
            # Generate backup codes
            codes = [secrets.token_hex(4).upper() for _ in range(10)]
            user.mfa.backup_codes_hash = [
                self._hash_backup_code(code) for code in codes
            ]

            return {
                "codes": codes,
                "method": "backup_codes",
            }

        raise ValueError(f"Unsupported MFA method: {method}")

    async def verify_mfa_setup(
        self,
        user_id: str,
        method: MFAMethod,
        code: str,
    ) -> bool:
        """Verify MFA setup with a test code."""
        user = self._users.get(user_id)
        if not user:
            return False

        verified = await self._verify_mfa_code(user, code, method)

        if verified:
            user.mfa.enabled = True
            if method == MFAMethod.TOTP:
                user.mfa.totp_verified = True
                user.mfa.primary_method = MFAMethod.TOTP

        return verified

    async def disable_mfa(self, user_id: str) -> bool:
        """Disable MFA for a user."""
        user = self._users.get(user_id)
        if not user:
            return False

        user.mfa = MFAConfig()
        return True

    # =========================================================================
    # Session Management
    # =========================================================================

    async def create_session(
        self,
        user: User,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> Session:
        """Create a session for a user."""
        return await self.sessions.create_session(
            user_id=user.id,
            tenant_id=user.tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            device_id=device_id,
        )

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return await self.sessions.get_session(session_id)

    async def validate_session(
        self,
        session_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Tuple[Optional[Session], Optional[str]]:
        """Validate a session."""
        return await self.sessions.validate_session(
            session_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    async def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID (for providers)."""
        return self._users.get(user_id)

    async def _get_user_flexible(
        self,
        identifier: str,
        by: str = "id",
        **kwargs,
    ) -> Optional[User]:
        """Flexible user lookup for providers."""
        if by == "id":
            return self._users.get(identifier)
        elif by == "email":
            user_id = self._users_by_email.get(identifier.lower())
            return self._users.get(user_id) if user_id else None
        elif by == "username":
            user_id = self._users_by_username.get(identifier.lower())
            return self._users.get(user_id) if user_id else None
        elif by == "oauth":
            provider = kwargs.get("provider", "")
            key = f"{provider}:{identifier}"
            user_id = self._oauth_links.get(key)
            return self._users.get(user_id) if user_id else None
        return None

    async def _update_user(self, user: User) -> None:
        """Update user (for providers)."""
        await self.update_user(user)

    async def _create_oauth_user(
        self,
        username: str,
        email: str,
        display_name: str = "",
        oauth_provider: str = "",
        oauth_id: str = "",
        oauth_data: Dict[str, Any] = None,
        email_verified: bool = False,
    ) -> User:
        """Create user from OAuth (for providers)."""
        # Ensure unique username
        base_username = username
        counter = 1
        while username.lower() in self._users_by_username:
            username = f"{base_username}_{counter}"
            counter += 1

        user = User(
            username=username,
            email=email.lower() if email else "",
            display_name=display_name or username,
            roles=["user"],
            status=UserStatus.ACTIVE,
            email_verified=email_verified,
            email_verified_at=datetime.now() if email_verified else None,
            linked_accounts=[{
                "provider": oauth_provider,
                "id": oauth_id,
                "data": oauth_data,
                "linked_at": datetime.now().isoformat(),
            }],
        )

        self._users[user.id] = user
        if email:
            self._users_by_email[email.lower()] = user.id
        self._users_by_username[username.lower()] = user.id
        self._oauth_links[f"{oauth_provider}:{oauth_id}"] = user.id

        return user

    async def _link_oauth_account(
        self,
        user: User,
        provider: str,
        provider_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Link OAuth account to existing user."""
        user.linked_accounts.append({
            "provider": provider,
            "id": provider_id,
            "data": data,
            "linked_at": datetime.now().isoformat(),
        })
        self._oauth_links[f"{provider}:{provider_id}"] = user.id

    async def _create_auth_response(
        self,
        user: User,
        ip_address: Optional[str],
        user_agent: Optional[str],
        device_id: Optional[str],
        auth_method: AuthMethod,
    ) -> AuthenticationResult:
        """Create authentication response with tokens."""
        # Create access token
        access_result = await self.tokens.create_access_token(user)

        # Create refresh token
        refresh_result = await self.tokens.create_refresh_token(user)

        # Create session
        session = await self.sessions.create_session(
            user_id=user.id,
            tenant_id=user.tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            device_id=device_id,
        )

        # Update user last login
        user.last_login_at = datetime.now()
        user.last_active_at = datetime.now()

        # Create security context
        context = SecurityContext(
            user_id=user.id,
            user=user,
            tenant_id=user.tenant_id,
            auth_method=auth_method,
            session=session,
            roles=user.roles,
            authenticated_at=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        await self._emit_event(
            "auth_success",
            {"user_id": user.id, "method": auth_method.value, "ip_address": ip_address},
        )

        return AuthenticationResult(
            status=AuthenticationResultStatus.SUCCESS,
            context=context,
            access_token=access_result.raw_token,
            refresh_token=refresh_result.raw_token,
            expires_in=3600,
            authenticated_at=datetime.now(),
            auth_method=auth_method,
        )

    async def _verify_mfa_code(
        self,
        user: User,
        code: str,
        method: MFAMethod,
    ) -> bool:
        """Verify MFA code."""
        if method == MFAMethod.TOTP:
            return self._verify_totp(user.mfa.totp_secret, code)
        elif method == MFAMethod.BACKUP_CODES:
            return self._verify_backup_code(user, code)
        return False

    def _verify_totp(self, secret: str, code: str) -> bool:
        """Verify TOTP code."""
        try:
            import pyotp

            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)
        except ImportError:
            # Fallback implementation
            import hmac
            import struct
            import time

            counter = int(time.time()) // 30
            key = bytes.fromhex(secret)

            for i in range(-1, 2):  # Allow 1 period before/after
                c = counter + i
                msg = struct.pack(">Q", c)
                h = hmac.new(key, msg, "sha1").digest()
                offset = h[-1] & 0x0F
                truncated = struct.unpack(">I", h[offset : offset + 4])[0]
                otp = (truncated & 0x7FFFFFFF) % 1000000
                if f"{otp:06d}" == code:
                    return True
            return False
        except Exception:
            return False

    def _verify_backup_code(self, user: User, code: str) -> bool:
        """Verify and consume a backup code."""
        code_hash = self._hash_backup_code(code)

        if code_hash in user.mfa.backup_codes_hash:
            user.mfa.backup_codes_hash.remove(code_hash)
            user.mfa.backup_codes_used += 1
            return True
        return False

    def _hash_backup_code(self, code: str) -> str:
        """Hash a backup code."""
        import hashlib

        return hashlib.sha256(code.upper().encode()).hexdigest()

    def _map_error_status(self, result: AuthProviderResult) -> AuthenticationResultStatus:
        """Map provider result to authentication status."""
        if result.account_locked:
            return AuthenticationResultStatus.ACCOUNT_LOCKED
        if result.error_code == "account_suspended":
            return AuthenticationResultStatus.ACCOUNT_SUSPENDED
        return AuthenticationResultStatus.FAILURE

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit authentication event."""
        for handler in self._event_handlers:
            try:
                handler(event_type, data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def add_event_handler(self, handler: Callable) -> None:
        """Add an authentication event handler."""
        self._event_handlers.append(handler)

    def get_stats(self) -> Dict[str, Any]:
        """Get authentication service statistics."""
        return {
            "users": len(self._users),
            "tokens": self.tokens.get_stats(),
            "sessions": self.sessions.get_stats(),
            "providers": list(self._providers.keys()),
        }
