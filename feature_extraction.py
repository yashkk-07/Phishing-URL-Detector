
from urllib.parse import urlparse
import ipaddress


def extract_host_features(url):
    parsed = urlparse(str(url))
    host = (parsed.hostname or "").lower()
    parts = host.split(".") if host else []

    # Check whether the hostname is an IP address.
    try:
        ipaddress.ip_address(host)
        is_ip = 1
    except ValueError:
        is_ip = 0

    # Extract structured URL features.
    return {
        "HostLength": len(host),
        "HostIsIP": is_ip,
        "SubdomainCount": max(0, len(parts) - 2),
        "HasHyphenInHost": int("-" in host),
        "HostDigitCount": sum(c.isdigit() for c in host),
        "PathLength": len(parsed.path),
        "QueryLength": len(parsed.query),
        "URLLength": len(str(url)),
    }
