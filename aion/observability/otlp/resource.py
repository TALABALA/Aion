"""
Resource Detection for OpenTelemetry.

Automatically detects resource attributes from the environment.
"""

import os
import platform
import socket
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .semantic import ResourceAttributes

logger = logging.getLogger(__name__)


@dataclass
class Resource:
    """Represents a resource with attributes."""
    attributes: Dict[str, Any] = field(default_factory=dict)
    schema_url: str = ""

    def merge(self, other: 'Resource') -> 'Resource':
        """Merge with another resource (other takes precedence)."""
        merged = dict(self.attributes)
        merged.update(other.attributes)
        return Resource(attributes=merged, schema_url=other.schema_url or self.schema_url)


class ResourceDetector(ABC):
    """Base class for resource detectors."""

    @abstractmethod
    def detect(self) -> Resource:
        """Detect resource attributes."""
        pass


class ServiceResourceDetector(ResourceDetector):
    """Detect service resource attributes."""

    def __init__(self, service_name: str = None, service_version: str = None,
                 service_namespace: str = None, service_instance_id: str = None):
        self.service_name = service_name
        self.service_version = service_version
        self.service_namespace = service_namespace
        self.service_instance_id = service_instance_id

    def detect(self) -> Resource:
        attrs = {}
        attrs[ResourceAttributes.SERVICE_NAME] = (
            self.service_name or
            os.environ.get("OTEL_SERVICE_NAME") or
            os.environ.get("SERVICE_NAME") or
            "unknown_service"
        )

        if self.service_version or os.environ.get("SERVICE_VERSION"):
            attrs[ResourceAttributes.SERVICE_VERSION] = (
                self.service_version or os.environ.get("SERVICE_VERSION")
            )

        if self.service_namespace or os.environ.get("SERVICE_NAMESPACE"):
            attrs[ResourceAttributes.SERVICE_NAMESPACE] = (
                self.service_namespace or os.environ.get("SERVICE_NAMESPACE")
            )

        if self.service_instance_id or os.environ.get("SERVICE_INSTANCE_ID"):
            attrs[ResourceAttributes.SERVICE_INSTANCE_ID] = (
                self.service_instance_id or os.environ.get("SERVICE_INSTANCE_ID")
            )

        return Resource(attributes=attrs)


class HostResourceDetector(ResourceDetector):
    """Detect host resource attributes."""

    def detect(self) -> Resource:
        attrs = {}
        try:
            attrs[ResourceAttributes.HOST_NAME] = socket.gethostname()
            attrs[ResourceAttributes.HOST_ARCH] = platform.machine()

            # Try to get host ID
            if os.path.exists("/etc/machine-id"):
                with open("/etc/machine-id") as f:
                    attrs[ResourceAttributes.HOST_ID] = f.read().strip()
        except Exception as e:
            logger.debug(f"Host detection error: {e}")

        return Resource(attributes=attrs)


class ProcessResourceDetector(ResourceDetector):
    """Detect process resource attributes."""

    def detect(self) -> Resource:
        import sys
        attrs = {
            ResourceAttributes.PROCESS_PID: os.getpid(),
            ResourceAttributes.PROCESS_EXECUTABLE_NAME: os.path.basename(sys.executable),
            ResourceAttributes.PROCESS_EXECUTABLE_PATH: sys.executable,
            ResourceAttributes.PROCESS_RUNTIME_NAME: platform.python_implementation(),
            ResourceAttributes.PROCESS_RUNTIME_VERSION: platform.python_version(),
        }

        if hasattr(os, 'getppid'):
            attrs[ResourceAttributes.PROCESS_PARENT_PID] = os.getppid()

        try:
            import getpass
            attrs[ResourceAttributes.PROCESS_OWNER] = getpass.getuser()
        except Exception:
            pass

        return Resource(attributes=attrs)


class ContainerResourceDetector(ResourceDetector):
    """Detect container resource attributes."""

    def detect(self) -> Resource:
        attrs = {}

        # Check for Docker
        if os.path.exists("/.dockerenv"):
            attrs[ResourceAttributes.CONTAINER_RUNTIME] = "docker"

            # Try to get container ID from cgroup
            try:
                with open("/proc/self/cgroup") as f:
                    for line in f:
                        if "docker" in line or "kubepods" in line:
                            parts = line.strip().split("/")
                            if parts:
                                attrs[ResourceAttributes.CONTAINER_ID] = parts[-1][:12]
                                break
            except Exception:
                pass

        # Check for container name from env
        if os.environ.get("CONTAINER_NAME"):
            attrs[ResourceAttributes.CONTAINER_NAME] = os.environ["CONTAINER_NAME"]

        return Resource(attributes=attrs)


class K8sResourceDetector(ResourceDetector):
    """Detect Kubernetes resource attributes."""

    def detect(self) -> Resource:
        attrs = {}

        # Standard K8s environment variables
        env_mappings = {
            "KUBERNETES_SERVICE_HOST": None,  # Just for detection
            "POD_NAME": ResourceAttributes.K8S_POD_NAME,
            "POD_NAMESPACE": ResourceAttributes.K8S_NAMESPACE_NAME,
            "NODE_NAME": ResourceAttributes.K8S_NODE_NAME,
        }

        # Check if running in K8s
        if not os.environ.get("KUBERNETES_SERVICE_HOST"):
            return Resource(attributes=attrs)

        for env_var, attr_name in env_mappings.items():
            if attr_name and os.environ.get(env_var):
                attrs[attr_name] = os.environ[env_var]

        # Try downward API paths
        downward_paths = {
            "/etc/podinfo/labels": None,
            "/etc/podinfo/annotations": None,
        }

        return Resource(attributes=attrs)


class CloudResourceDetector(ResourceDetector):
    """Detect cloud provider resource attributes."""

    def detect(self) -> Resource:
        attrs = {}

        # AWS
        if os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"):
            attrs[ResourceAttributes.CLOUD_PROVIDER] = "aws"
            attrs[ResourceAttributes.CLOUD_REGION] = (
                os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            )
            if os.environ.get("AWS_ACCOUNT_ID"):
                attrs[ResourceAttributes.CLOUD_ACCOUNT_ID] = os.environ["AWS_ACCOUNT_ID"]

        # GCP
        elif os.environ.get("GOOGLE_CLOUD_PROJECT"):
            attrs[ResourceAttributes.CLOUD_PROVIDER] = "gcp"
            attrs[ResourceAttributes.CLOUD_ACCOUNT_ID] = os.environ["GOOGLE_CLOUD_PROJECT"]

        # Azure
        elif os.environ.get("AZURE_SUBSCRIPTION_ID"):
            attrs[ResourceAttributes.CLOUD_PROVIDER] = "azure"
            attrs[ResourceAttributes.CLOUD_ACCOUNT_ID] = os.environ["AZURE_SUBSCRIPTION_ID"]

        return Resource(attributes=attrs)


class CompositeResourceDetector(ResourceDetector):
    """Combines multiple resource detectors."""

    def __init__(self, detectors: List[ResourceDetector] = None):
        self.detectors = detectors or [
            ServiceResourceDetector(),
            HostResourceDetector(),
            ProcessResourceDetector(),
            ContainerResourceDetector(),
            K8sResourceDetector(),
            CloudResourceDetector(),
        ]

    def detect(self) -> Resource:
        result = Resource()
        for detector in self.detectors:
            try:
                detected = detector.detect()
                result = result.merge(detected)
            except Exception as e:
                logger.debug(f"Resource detection error: {e}")
        return result


def get_resource(service_name: str = None, **kwargs) -> Resource:
    """Get resource with automatic detection."""
    detectors = [
        ServiceResourceDetector(service_name=service_name, **kwargs),
        HostResourceDetector(),
        ProcessResourceDetector(),
        ContainerResourceDetector(),
        K8sResourceDetector(),
        CloudResourceDetector(),
    ]

    composite = CompositeResourceDetector(detectors)
    return composite.detect()
