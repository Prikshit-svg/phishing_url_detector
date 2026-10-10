"""Deterministic lexical URL feature extraction and heuristic analysis.

No network requests are made. DNS, certificate, and geographic facts cannot be
verified from a URL string alone; helper outputs are explicitly heuristic.
"""
from __future__ import annotations

import ipaddress
import math
import re
from urllib.parse import urlsplit

SUSP = ['login','secure','account','update','verify','banking','confirm','password',
        'signin','paypal','ebay','amazon','support','service','free','lucky','bonus','click','webscr']
BRANDS = ['paypal','amazon','ebay','apple','microsoft','netflix','facebook',
          'google','bank','instagram','whatsapp','outlook','office365']
SHORT = ['bit.ly','tinyurl','goo.gl','t.co','ow.ly','is.gd','cutt.ly']
CT = ['.com','.org','.net','.edu','.gov']
ST = ['.xyz','.tk','.ml','.ga','.cf','.gq','.top','.click','.link','.online','.site','.work']


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    n = len(value)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def _normalise(url) -> str:
    value = str(url or '').strip().lower()
    value = re.sub(r'^https?://', '', value)
    value = re.sub(r'^www\.', '', value)
    return value


def _host_and_path(url: str):
    """Parse host/path even when a user supplies a scheme-less URL."""
    raw = str(url or '').strip()
    candidate = raw if re.match(r'^[a-z][a-z0-9+.-]*://', raw, re.I) else '//' + raw
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or '').lower().rstrip('.')
        path = parsed.path or ''
        # Preserve port separately in the host string for legacy feature semantics.
        netloc = (parsed.netloc or '').lower().split('@')[-1]
        return host or netloc, path, parsed
    except ValueError:
        return _normalise(raw).split('/', 1)[0], '', None


def extract_features(url) -> dict:
    raw = str(url or '').strip()
    norm = _normalise(raw)
    host, path, parsed = _host_and_path(raw)
    # Lexical statistics intentionally cover the complete normalized URL.
    domain = host or norm.split('/', 1)[0]
    f = {}
    f['url_length'] = len(norm)
    f['domain_length'] = len(domain)
    f['path_length'] = len(path)
    f['num_dots'] = norm.count('.')
    f['num_hyphens'] = norm.count('-')
    f['num_underscores'] = norm.count('_')
    f['num_slashes'] = norm.count('/')
    f['num_question_marks'] = norm.count('?')
    f['num_equal_signs'] = norm.count('=')
    f['num_at_signs'] = norm.count('@')
    f['num_ampersands'] = norm.count('&')
    f['num_percent'] = norm.count('%')
    f['num_digits'] = sum(c.isdigit() for c in norm)
    f['num_subdomains'] = max(0, domain.count('.') - 1)
    f['url_entropy'] = round(_entropy(norm), 4)
    f['domain_entropy'] = round(_entropy(domain), 4)
    f['digit_ratio'] = round(f['num_digits'] / max(len(norm), 1), 4)
    try:
        ipaddress.ip_address(domain.strip('[]'))
        is_ip = True
    except ValueError:
        is_ip = False
    f['has_ip'] = int(is_ip)
    netloc = parsed.netloc if parsed else domain
    f['has_port'] = int(bool(re.search(r':\d+$', netloc.split('@')[-1])))
    f['is_shortened'] = int(any(s == domain or domain.endswith('.' + s) for s in SHORT))
    f['has_suspicious_word'] = int(any(w in norm for w in SUSP))
    f['has_double_slash'] = int('//' in path)
    f['has_prefix_suffix'] = int('-' in domain)
    f['path_depth'] = len([segment for segment in path.split('/') if segment])
    tld = '.' + domain.split('.')[-1] if '.' in domain else ''
    f['is_common_tld'] = int(tld in CT)
    f['is_suspicious_tld'] = int(tld in ST)
    # Brand terms in the registrable host are suspicious unless the host is the
    # brand's canonical brand.com domain; this remains a heuristic, not proof.
    f['brand_impersonation'] = int(any(b in norm and not (domain == b + '.com' or domain.endswith('.' + b + '.com')) for b in BRANDS))
    f['num_suspicious_words'] = sum(1 for w in SUSP if w in norm)
    f['digits_in_domain'] = sum(c.isdigit() for c in domain)
    f['long_domain'] = int(len(domain) > 30)
    return f


def feature_names():
    return list(extract_features('https://example.com/path').keys())


def analyze_ssl(url):
    """Report the scheme only; do not imply certificate validation."""
    value = str(url or '').strip().lower()
    https = value.startswith('https://')
    return {'scheme': 'https' if https else ('http' if value.startswith('http://') else 'unspecified'),
            'verdict': 'HTTPS scheme present (certificate not checked)' if https else 'HTTPS not specified (certificate not checked)',
            'risk': 'Neutral' if https else 'Elevated'}


def analyze_dns(url):
    host, _, _ = _host_and_path(url)
    labels = host.split('.') if host else []
    tld = '.' + labels[-1] if labels else ''
    try:
        ipaddress.ip_address(host.strip('[]'))
        is_ip = True
    except ValueError:
        is_ip = False
    return {'tld': tld or '(none)', 'num_subdomains': max(0, len(labels) - 2),
            'is_ip_literal': is_ip, 'risk': 'Elevated' if (is_ip or tld in ST) else 'Neutral',
            'note': 'String-based heuristic only; DNS was not queried.'}


_TLD_REGION = {'.uk':'United Kingdom (TLD)', '.us':'United States (TLD)', '.ru':'Russia (TLD)',
               '.cn':'China (TLD)', '.de':'Germany (TLD)', '.in':'India (TLD)',
               '.tk':'Tokelau (TLD)', '.ml':'Mali (TLD)', '.ga':'Gabon (TLD)'}

def analyze_geo(url):
    host, _, _ = _host_and_path(url)
    labels = host.split('.') if host else []
    tld = '.' + labels[-1] if labels else ''
    return {'inferred_region': _TLD_REGION.get(tld, 'Unknown / generic TLD'),
            'risk': 'Neutral',
            'note': 'TLD association only; this does not identify server location.'}


def analyze_tokens(url):
    u = _normalise(url)
    found = [w for w in SUSP if w in u]
    brands = [b for b in BRANDS if b in u]
    return {'suspicious_tokens': found, 'brand_terms': brands, 'has_at_symbol': '@' in u,
            'has_hex_encoding': '%' in u,
            'risk': 'Elevated' if (len(found) >= 2 or brands) else ('Medium' if found else 'Low')}


def threat_score(url):
    f = extract_features(url)
    score, reasons = 0, []
    def add(points, message):
        nonlocal score
        score += points
        reasons.append((points, message))
    if f['has_ip']: add(20, 'IP address used as host')
    if f['is_suspicious_tld']: add(15, 'TLD on heuristic watchlist')
    if f['brand_impersonation']: add(20, 'Possible brand impersonation')
    if f['is_shortened']: add(10, 'Known URL shortener domain')
    if f['num_at_signs']: add(10, 'Contains @ symbol')
    if f['num_suspicious_words'] >= 2: add(10, 'Multiple suspicious keywords')
    if f['digits_in_domain'] >= 2: add(5, 'Digits in hostname')
    if f['long_domain']: add(5, 'Unusually long hostname')
    score = min(score, 100)
    level = 'High' if score >= 60 else 'Medium' if score >= 30 else 'Low'
    return {'score': score, 'level': level, 'reasons': reasons,
            'note': 'Heuristic indicator score, not a calibrated probability.'}


def full_analysis(url):
    return {'features': extract_features(url), 'ssl': analyze_ssl(url), 'dns': analyze_dns(url),
            'geo': analyze_geo(url), 'tokens': analyze_tokens(url), 'threat': threat_score(url)}
