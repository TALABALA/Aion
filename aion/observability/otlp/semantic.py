"""
OpenTelemetry Semantic Conventions.

Implements standard semantic conventions for:
- Resource attributes
- Span attributes
- Metric attributes
- Log attributes
- Protocol-specific semantics (HTTP, Database, Messaging, RPC)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# =============================================================================
# Resource Semantic Conventions
# =============================================================================

class ResourceAttributes:
    """Standard resource attribute names."""

    # Service
    SERVICE_NAME = "service.name"
    SERVICE_NAMESPACE = "service.namespace"
    SERVICE_INSTANCE_ID = "service.instance.id"
    SERVICE_VERSION = "service.version"

    # Telemetry SDK
    TELEMETRY_SDK_NAME = "telemetry.sdk.name"
    TELEMETRY_SDK_LANGUAGE = "telemetry.sdk.language"
    TELEMETRY_SDK_VERSION = "telemetry.sdk.version"

    # Host
    HOST_ID = "host.id"
    HOST_NAME = "host.name"
    HOST_TYPE = "host.type"
    HOST_ARCH = "host.arch"
    HOST_IMAGE_NAME = "host.image.name"
    HOST_IMAGE_ID = "host.image.id"
    HOST_IMAGE_VERSION = "host.image.version"

    # OS
    OS_TYPE = "os.type"
    OS_DESCRIPTION = "os.description"
    OS_NAME = "os.name"
    OS_VERSION = "os.version"

    # Process
    PROCESS_PID = "process.pid"
    PROCESS_PARENT_PID = "process.parent_pid"
    PROCESS_EXECUTABLE_NAME = "process.executable.name"
    PROCESS_EXECUTABLE_PATH = "process.executable.path"
    PROCESS_COMMAND = "process.command"
    PROCESS_COMMAND_LINE = "process.command_line"
    PROCESS_COMMAND_ARGS = "process.command_args"
    PROCESS_OWNER = "process.owner"
    PROCESS_RUNTIME_NAME = "process.runtime.name"
    PROCESS_RUNTIME_VERSION = "process.runtime.version"
    PROCESS_RUNTIME_DESCRIPTION = "process.runtime.description"

    # Container
    CONTAINER_NAME = "container.name"
    CONTAINER_ID = "container.id"
    CONTAINER_RUNTIME = "container.runtime"
    CONTAINER_IMAGE_NAME = "container.image.name"
    CONTAINER_IMAGE_TAG = "container.image.tag"

    # Kubernetes
    K8S_CLUSTER_NAME = "k8s.cluster.name"
    K8S_NODE_NAME = "k8s.node.name"
    K8S_NODE_UID = "k8s.node.uid"
    K8S_NAMESPACE_NAME = "k8s.namespace.name"
    K8S_POD_UID = "k8s.pod.uid"
    K8S_POD_NAME = "k8s.pod.name"
    K8S_CONTAINER_NAME = "k8s.container.name"
    K8S_REPLICASET_UID = "k8s.replicaset.uid"
    K8S_REPLICASET_NAME = "k8s.replicaset.name"
    K8S_DEPLOYMENT_UID = "k8s.deployment.uid"
    K8S_DEPLOYMENT_NAME = "k8s.deployment.name"
    K8S_STATEFULSET_UID = "k8s.statefulset.uid"
    K8S_STATEFULSET_NAME = "k8s.statefulset.name"
    K8S_DAEMONSET_UID = "k8s.daemonset.uid"
    K8S_DAEMONSET_NAME = "k8s.daemonset.name"
    K8S_JOB_UID = "k8s.job.uid"
    K8S_JOB_NAME = "k8s.job.name"
    K8S_CRONJOB_UID = "k8s.cronjob.uid"
    K8S_CRONJOB_NAME = "k8s.cronjob.name"

    # Cloud
    CLOUD_PROVIDER = "cloud.provider"
    CLOUD_ACCOUNT_ID = "cloud.account.id"
    CLOUD_REGION = "cloud.region"
    CLOUD_AVAILABILITY_ZONE = "cloud.availability_zone"
    CLOUD_PLATFORM = "cloud.platform"

    # Deployment
    DEPLOYMENT_ENVIRONMENT = "deployment.environment"


# =============================================================================
# Span Semantic Conventions
# =============================================================================

class SpanAttributes:
    """Standard span attribute names."""

    # General
    OTEL_STATUS_CODE = "otel.status_code"
    OTEL_STATUS_DESCRIPTION = "otel.status_description"

    # Exception
    EXCEPTION_TYPE = "exception.type"
    EXCEPTION_MESSAGE = "exception.message"
    EXCEPTION_STACKTRACE = "exception.stacktrace"
    EXCEPTION_ESCAPED = "exception.escaped"

    # Network
    NET_TRANSPORT = "net.transport"
    NET_PEER_IP = "net.peer.ip"
    NET_PEER_PORT = "net.peer.port"
    NET_PEER_NAME = "net.peer.name"
    NET_HOST_IP = "net.host.ip"
    NET_HOST_PORT = "net.host.port"
    NET_HOST_NAME = "net.host.name"
    NET_SOCK_PEER_ADDR = "net.sock.peer.addr"
    NET_SOCK_PEER_PORT = "net.sock.peer.port"
    NET_SOCK_HOST_ADDR = "net.sock.host.addr"
    NET_SOCK_HOST_PORT = "net.sock.host.port"

    # Peer
    PEER_SERVICE = "peer.service"

    # Thread
    THREAD_ID = "thread.id"
    THREAD_NAME = "thread.name"

    # Code
    CODE_FUNCTION = "code.function"
    CODE_NAMESPACE = "code.namespace"
    CODE_FILEPATH = "code.filepath"
    CODE_LINENO = "code.lineno"

    # Enduser
    ENDUSER_ID = "enduser.id"
    ENDUSER_ROLE = "enduser.role"
    ENDUSER_SCOPE = "enduser.scope"


# =============================================================================
# HTTP Semantic Conventions
# =============================================================================

class HTTPSemantics:
    """HTTP semantic conventions."""

    # Request
    HTTP_METHOD = "http.method"
    HTTP_URL = "http.url"
    HTTP_TARGET = "http.target"
    HTTP_HOST = "http.host"
    HTTP_SCHEME = "http.scheme"
    HTTP_ROUTE = "http.route"
    HTTP_USER_AGENT = "http.user_agent"
    HTTP_REQUEST_CONTENT_LENGTH = "http.request_content_length"
    HTTP_REQUEST_CONTENT_LENGTH_UNCOMPRESSED = "http.request_content_length_uncompressed"

    # Response
    HTTP_STATUS_CODE = "http.status_code"
    HTTP_FLAVOR = "http.flavor"
    HTTP_RESPONSE_CONTENT_LENGTH = "http.response_content_length"
    HTTP_RESPONSE_CONTENT_LENGTH_UNCOMPRESSED = "http.response_content_length_uncompressed"

    # Server
    HTTP_SERVER_NAME = "http.server_name"
    HTTP_CLIENT_IP = "http.client_ip"

    # Retry
    HTTP_RETRY_COUNT = "http.retry_count"

    @staticmethod
    def status_code_type(code: int) -> str:
        """Get status code type (1xx, 2xx, etc.)."""
        if 100 <= code < 200:
            return "1xx"
        elif 200 <= code < 300:
            return "2xx"
        elif 300 <= code < 400:
            return "3xx"
        elif 400 <= code < 500:
            return "4xx"
        elif 500 <= code < 600:
            return "5xx"
        return "unknown"


# =============================================================================
# Database Semantic Conventions
# =============================================================================

class DatabaseSemantics:
    """Database semantic conventions."""

    # Connection
    DB_SYSTEM = "db.system"
    DB_CONNECTION_STRING = "db.connection_string"
    DB_USER = "db.user"
    DB_NAME = "db.name"

    # Statement
    DB_STATEMENT = "db.statement"
    DB_OPERATION = "db.operation"

    # SQL
    DB_SQL_TABLE = "db.sql.table"

    # Redis
    DB_REDIS_DATABASE_INDEX = "db.redis.database_index"

    # MongoDB
    DB_MONGODB_COLLECTION = "db.mongodb.collection"

    # Cassandra
    DB_CASSANDRA_KEYSPACE = "db.cassandra.keyspace"
    DB_CASSANDRA_PAGE_SIZE = "db.cassandra.page_size"
    DB_CASSANDRA_CONSISTENCY_LEVEL = "db.cassandra.consistency_level"
    DB_CASSANDRA_TABLE = "db.cassandra.table"
    DB_CASSANDRA_IDEMPOTENCE = "db.cassandra.idempotence"
    DB_CASSANDRA_SPECULATIVE_EXECUTION_COUNT = "db.cassandra.speculative_execution_count"
    DB_CASSANDRA_COORDINATOR_ID = "db.cassandra.coordinator.id"
    DB_CASSANDRA_COORDINATOR_DC = "db.cassandra.coordinator.dc"

    # Elasticsearch
    DB_ELASTICSEARCH_CLUSTER_NAME = "db.elasticsearch.cluster.name"
    DB_ELASTICSEARCH_NODE_NAME = "db.elasticsearch.node.name"

    # Common values for db.system
    SYSTEM_POSTGRESQL = "postgresql"
    SYSTEM_MYSQL = "mysql"
    SYSTEM_MARIADB = "mariadb"
    SYSTEM_MSSQL = "mssql"
    SYSTEM_ORACLE = "oracle"
    SYSTEM_DB2 = "db2"
    SYSTEM_SQLITE = "sqlite"
    SYSTEM_MONGODB = "mongodb"
    SYSTEM_REDIS = "redis"
    SYSTEM_MEMCACHED = "memcached"
    SYSTEM_CASSANDRA = "cassandra"
    SYSTEM_ELASTICSEARCH = "elasticsearch"


# =============================================================================
# Messaging Semantic Conventions
# =============================================================================

class MessagingSemantics:
    """Messaging semantic conventions."""

    # System
    MESSAGING_SYSTEM = "messaging.system"
    MESSAGING_DESTINATION = "messaging.destination"
    MESSAGING_DESTINATION_KIND = "messaging.destination_kind"
    MESSAGING_TEMP_DESTINATION = "messaging.temp_destination"
    MESSAGING_PROTOCOL = "messaging.protocol"
    MESSAGING_PROTOCOL_VERSION = "messaging.protocol_version"
    MESSAGING_URL = "messaging.url"

    # Message
    MESSAGING_MESSAGE_ID = "messaging.message_id"
    MESSAGING_CONVERSATION_ID = "messaging.conversation_id"
    MESSAGING_MESSAGE_PAYLOAD_SIZE_BYTES = "messaging.message_payload_size_bytes"
    MESSAGING_MESSAGE_PAYLOAD_COMPRESSED_SIZE_BYTES = "messaging.message_payload_compressed_size_bytes"

    # Consumer
    MESSAGING_CONSUMER_ID = "messaging.consumer_id"

    # Operation
    MESSAGING_OPERATION = "messaging.operation"

    # Kafka
    MESSAGING_KAFKA_MESSAGE_KEY = "messaging.kafka.message_key"
    MESSAGING_KAFKA_CONSUMER_GROUP = "messaging.kafka.consumer_group"
    MESSAGING_KAFKA_CLIENT_ID = "messaging.kafka.client_id"
    MESSAGING_KAFKA_PARTITION = "messaging.kafka.partition"
    MESSAGING_KAFKA_TOMBSTONE = "messaging.kafka.tombstone"

    # RabbitMQ
    MESSAGING_RABBITMQ_ROUTING_KEY = "messaging.rabbitmq.routing_key"

    # Common values
    SYSTEM_KAFKA = "kafka"
    SYSTEM_RABBITMQ = "rabbitmq"
    SYSTEM_ACTIVEMQ = "activemq"
    SYSTEM_AZURE_SERVICEBUS = "azure_servicebus"
    SYSTEM_AZURE_EVENTHUBS = "azure_eventhubs"
    SYSTEM_AWS_SQS = "aws_sqs"
    SYSTEM_AWS_SNS = "aws_sns"
    SYSTEM_GCP_PUBSUB = "gcp_pubsub"


# =============================================================================
# RPC Semantic Conventions
# =============================================================================

class RPCSemantics:
    """RPC semantic conventions."""

    # System
    RPC_SYSTEM = "rpc.system"
    RPC_SERVICE = "rpc.service"
    RPC_METHOD = "rpc.method"

    # gRPC
    RPC_GRPC_STATUS_CODE = "rpc.grpc.status_code"

    # JSON-RPC
    RPC_JSONRPC_VERSION = "rpc.jsonrpc.version"
    RPC_JSONRPC_REQUEST_ID = "rpc.jsonrpc.request_id"
    RPC_JSONRPC_ERROR_CODE = "rpc.jsonrpc.error_code"
    RPC_JSONRPC_ERROR_MESSAGE = "rpc.jsonrpc.error_message"

    # Common values
    SYSTEM_GRPC = "grpc"
    SYSTEM_JAVA_RMI = "java_rmi"
    SYSTEM_DOTNET_WCF = "dotnet_wcf"
    SYSTEM_APACHE_DUBBO = "apache_dubbo"


# =============================================================================
# Metric Semantic Conventions
# =============================================================================

class MetricAttributes:
    """Standard metric attribute names."""

    # HTTP
    HTTP_METHOD = "http.method"
    HTTP_STATUS_CODE = "http.status_code"
    HTTP_FLAVOR = "http.flavor"
    HTTP_SCHEME = "http.scheme"
    HTTP_HOST = "http.host"
    HTTP_TARGET = "http.target"
    HTTP_ROUTE = "http.route"

    # Network
    NET_HOST_NAME = "net.host.name"
    NET_HOST_PORT = "net.host.port"
    NET_PEER_NAME = "net.peer.name"
    NET_PEER_PORT = "net.peer.port"

    # RPC
    RPC_SYSTEM = "rpc.system"
    RPC_SERVICE = "rpc.service"
    RPC_METHOD = "rpc.method"
    RPC_GRPC_STATUS_CODE = "rpc.grpc.status_code"

    # Database
    DB_SYSTEM = "db.system"
    DB_NAME = "db.name"
    DB_OPERATION = "db.operation"

    # Messaging
    MESSAGING_SYSTEM = "messaging.system"
    MESSAGING_DESTINATION = "messaging.destination"
    MESSAGING_DESTINATION_KIND = "messaging.destination_kind"


# =============================================================================
# Log Semantic Conventions
# =============================================================================

class LogAttributes:
    """Standard log attribute names."""

    # Log record
    LOG_RECORD_UID = "log.record.uid"
    LOG_FILE_NAME = "log.file.name"
    LOG_FILE_PATH = "log.file.path"
    LOG_FILE_NAME_RESOLVED = "log.file.name_resolved"
    LOG_FILE_PATH_RESOLVED = "log.file.path_resolved"
    LOG_IOSTREAM = "log.iostream"

    # Event
    EVENT_NAME = "event.name"
    EVENT_DOMAIN = "event.domain"


# =============================================================================
# Semantic Conventions Helper
# =============================================================================

class SemanticConventions:
    """Helper class for working with semantic conventions."""

    resource = ResourceAttributes
    span = SpanAttributes
    metric = MetricAttributes
    log = LogAttributes
    http = HTTPSemantics
    db = DatabaseSemantics
    messaging = MessagingSemantics
    rpc = RPCSemantics

    @staticmethod
    def create_http_server_span_attributes(
        method: str,
        url: str,
        status_code: int,
        route: str = None,
        user_agent: str = None,
        client_ip: str = None
    ) -> Dict[str, Any]:
        """Create standard HTTP server span attributes."""
        attrs = {
            HTTPSemantics.HTTP_METHOD: method,
            HTTPSemantics.HTTP_URL: url,
            HTTPSemantics.HTTP_STATUS_CODE: status_code,
        }

        if route:
            attrs[HTTPSemantics.HTTP_ROUTE] = route
        if user_agent:
            attrs[HTTPSemantics.HTTP_USER_AGENT] = user_agent
        if client_ip:
            attrs[HTTPSemantics.HTTP_CLIENT_IP] = client_ip

        return attrs

    @staticmethod
    def create_db_span_attributes(
        system: str,
        name: str = None,
        statement: str = None,
        operation: str = None,
        user: str = None
    ) -> Dict[str, Any]:
        """Create standard database span attributes."""
        attrs = {DatabaseSemantics.DB_SYSTEM: system}

        if name:
            attrs[DatabaseSemantics.DB_NAME] = name
        if statement:
            attrs[DatabaseSemantics.DB_STATEMENT] = statement
        if operation:
            attrs[DatabaseSemantics.DB_OPERATION] = operation
        if user:
            attrs[DatabaseSemantics.DB_USER] = user

        return attrs

    @staticmethod
    def create_messaging_span_attributes(
        system: str,
        destination: str,
        operation: str,
        message_id: str = None
    ) -> Dict[str, Any]:
        """Create standard messaging span attributes."""
        attrs = {
            MessagingSemantics.MESSAGING_SYSTEM: system,
            MessagingSemantics.MESSAGING_DESTINATION: destination,
            MessagingSemantics.MESSAGING_OPERATION: operation,
        }

        if message_id:
            attrs[MessagingSemantics.MESSAGING_MESSAGE_ID] = message_id

        return attrs
