#!/usr/bin/env python3
"""
Bibliothèque de génération de logs OpenStack synthétiques.
Utilise les patterns extraits du dataset réel.
"""
import json
import random
import uuid as uuidlib
from datetime import datetime, timedelta

# IDENTITÉS PERSISTANTES (tirées du dataset réel)
TENANTS = [
    "aae94b70fe574388bd98c5eb42daedc3",
    "a2dc1d02b1434044a78b35c2a0515e2d",
    "29fe7b5cd6cb4065b92b628f279fdb82",
    "721d8e28a75e45739f1a52b8c882cf5e",
]
USERS = [
    "1604e020bd18426aa0906259d272ad6c",
    "d83ce04ec578457e8fcf808882f16d7c",
    "d0e50f40da114520bbc08fec8920297e",
    "c58459880af645bf9daca418ea94fe97",
    "655d81b99ca84c27b885afa136b5c807",
    "45ce9e73bd5844eeb3dc91716814be9f",
    "60e255ef7c9a45eda3d3ef930bc423e4",
    "58a5c37b0d16482e91a1013eac00b78b",
]
CLIENT_IP = "192.168.91.128"
DB_HOST = "192.168.91.250"
HOSTNAME = "projet"
ROLE_MEMBER = "0eedb64c0b7343e4b3b357908247eda7"

# PIDS PERSISTANTS PAR SERVICE (réalistes)
SERVICE_PIDS = {
    "neutron-openvswitch-agent": ["7"],
    "neutron-server": ["26", "27", "28"],
    "neutron-rpc-server": ["7"],
    "neutron-l3-agent": ["3820"],
    "neutron-dhcp-agent": ["3997"],
    "neutron-metadata-agent": ["3850"],
    "neutron-periodic-workers": ["7"],
    "nova-api": ["22", "23", "24", "25", "26"],
    "nova-compute": ["7"],
    "nova-conductor": ["6"],
    "nova-scheduler": ["7"],
    "nova-metadata": ["6"],
    "cinder-api": ["23", "24", "25", "26"],
    "cinder-backup": ["7"],
    "cinder-volume": ["6"],
    "cinder-scheduler": ["6"],
    "keystone": ["25", "26", "27", "28"],
    "placement-api": ["24", "25", "26", "27"],
    "glance-api": ["21", "22", "23"],
    "heat-engine": ["7"],
    "heat-api": ["27", "28"],
    "heat-api-cfn": ["27", "28"],
    "libvirt": ["4612"],
}

UWSGI_PIDS = {
    "horizon-uwsgi": ["28", "29", "30", "31"],
    "cinder-api-uwsgi": ["23", "24", "25", "26"],
    "keystone-uwsgi": ["25", "26", "27", "28"],
    "neutron-server-uwsgi": ["24", "25", "26", "27"],
    "placement-api-uwsgi": ["24", "25", "26", "27"],
    "nova-api-uwsgi": ["22", "23", "24", "25"],
    "glance-api-uwsgi": ["20", "21", "22", "23"],
    "heat-api-uwsgi": ["25", "26", "27", "28"],
    "heat-api-cfn-uwsgi": ["22", "23", "24", "25"],
    "nova-metadata-uwsgi": ["22", "23", "24", "25"],
}

# FORMATAGE TIMESTAMPS
def fmt_iso(dt):
    """Format @timestamp: 2026-05-01T08:58:15.377000000+00:00"""
    nanos = dt.microsecond * 1000
    return dt.strftime('%Y-%m-%dT%H:%M:%S.') + f"{nanos:09d}+00:00"

def fmt_human(dt):
    """Format Timestamp: 2026-05-01 08:58:15.377"""
    millis = dt.microsecond // 1000
    return dt.strftime('%Y-%m-%d %H:%M:%S.') + f"{millis:03d}"

def fmt_uwsgi_ts(dt):
    """Format Timestamp uWSGI: Fri May  1 08:58:44 2026"""
    return dt.strftime('%a %b %e %H:%M:%S %Y').replace('  ', ' ').replace('  ', ' ').strip()

def fmt_mariadb_ts(dt):
    """Format mariadb timestamp: 2026-05-01 08:57:52 +0000"""
    return dt.strftime('%Y-%m-%d %H:%M:%S +0000')

def gen_uuid():
    return str(uuidlib.uuid4())

def gen_hex(n=12):
    return ''.join(random.choices('0123456789abcdef', k=n))

# GÉNÉRATEURS PAR TYPE DE LOG

def make_python_log(dt, prog, level, python_module, payload, *,
                    request_id=None, user_id="-", tenant_id="-",
                    domain_id="-", system_scope="-",
                    user_domain="-", project_domain="-",
                    pid=None, logger=None):
    """Format A - Logs Python OpenStack"""
    if pid is None:
        pid = random.choice(SERVICE_PIDS.get(prog, ["7"]))
    if logger is None:
        # Inférence du logger depuis prog
        for service in ['nova', 'neutron', 'cinder', 'keystone', 'glance',
                        'heat', 'placement', 'horizon']:
            if service in prog:
                logger = f"openstack.{service}"
                break
        else:
            logger = "openstack.nova"
    if request_id is None:
        request_id = gen_uuid()
    
    obj = {
        "Timestamp": fmt_human(dt),
        "Pid": pid,
        "log_level": level,
        "python_module": python_module,
        "request_id": request_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "domain_id": domain_id,
        "system_scope": system_scope,
        "user_domain": user_domain,
        "project_domain": project_domain,
        "Payload": payload,
        "Hostname": HOSTNAME,
        "Logger": logger,
        "programname": prog,
        "@timestamp": fmt_iso(dt),
    }
    return obj


def make_python_log_simple(dt, prog, level, python_module, payload, *,
                           pid=None, logger=None):
    """Variante sans request_id/user_id (lorsque ces champs absents)."""
    if pid is None:
        pid = random.choice(SERVICE_PIDS.get(prog, ["7"]))
    if logger is None:
        for service in ['nova', 'neutron', 'cinder', 'keystone', 'glance',
                        'heat', 'placement', 'horizon']:
            if service in prog:
                logger = f"openstack.{service}"
                break
        else:
            logger = "openstack.nova"
    
    obj = {
        "Timestamp": fmt_human(dt),
        "Pid": pid,
        "log_level": level,
        "python_module": python_module,
        "Payload": payload,
        "Hostname": HOSTNAME,
        "Logger": logger,
        "programname": prog,
        "@timestamp": fmt_iso(dt),
    }
    return obj


def make_uwsgi_log(dt, prog, method, path, status, gen_bytes, duration_ms, *,
                   http_version="1.1", req_counter=None, vars_count=None,
                   total_bytes=None, headers_count=None, headers_bytes=None,
                   pid=None, user="", logger=None):
    """Format B - uWSGI access log"""
    if pid is None:
        pid = random.choice(UWSGI_PIDS.get(prog, ["24"]))
    if req_counter is None:
        req_counter = f"{random.randint(1, 50)}/{random.randint(50, 500)}"
    if vars_count is None:
        vars_count = random.choice([22, 28])
    if total_bytes is None:
        total_bytes = random.randint(218, 350)
    if headers_count is None:
        headers_count = random.randint(3, 7)
    if headers_bytes is None:
        headers_bytes = random.randint(98, 400)
    if logger is None:
        for service in ['nova', 'neutron', 'cinder', 'keystone', 'glance',
                        'heat', 'placement', 'horizon']:
            if service in prog:
                logger = f"openstack.{service}"
                break
        else:
            logger = "openstack.nova"
    
    payload = (f"{method} {path} => generated {gen_bytes} bytes in {duration_ms} msecs "
               f"(HTTP/{http_version} {status}) {headers_count} headers in {headers_bytes} bytes "
               f"(1 switches on core 0)")
    
    obj = {
        "Pid": pid,
        "App": "0",
        "Req": req_counter,
        "Address": CLIENT_IP,
        "User": user,
        "Vars": str(vars_count),
        "Bytes": str(total_bytes),
        "Timestamp": fmt_uwsgi_ts(dt),
        "Payload": payload,
        "Hostname": HOSTNAME,
        "Logger": logger,
        "programname": prog,
        "@timestamp": fmt_iso(dt),
    }
    return obj


def make_uwsgi_log_minimal(dt, prog, payload, logger=None):
    """uwsgi log sans les champs Pid/App/Req (cas ~10-20% du dataset)"""
    if logger is None:
        for service in ['nova', 'neutron', 'cinder', 'keystone', 'glance',
                        'heat', 'placement', 'horizon']:
            if service in prog:
                logger = f"openstack.{service}"
                break
        else:
            logger = "openstack.nova"
    return {
        "Payload": payload,
        "Hostname": HOSTNAME,
        "Logger": logger,
        "programname": prog,
        "@timestamp": fmt_iso(dt),
    }


def make_neutron_ovs_rpc_loop(dt, iteration, prog="neutron-openvswitch-agent",
                               added=0, updated=0, removed=0, elapsed=None):
    """Boucle rpc_loop typique - 2 logs par itération (started + completed)"""
    if elapsed is None:
        elapsed = round(random.uniform(0.003, 0.012), 3)
    
    pid = "7"
    rid = gen_uuid()
    
    started = {
        "Timestamp": fmt_human(dt),
        "Pid": pid,
        "log_level": "INFO",
        "python_module": "neutron.plugins.ml2.drivers.openvswitch.agent.ovs_neutron_agent",
        "request_id": rid,
        "user_id": "-",
        "tenant_id": "-",
        "domain_id": "-",
        "system_scope": "-",
        "user_domain": "-",
        "project_domain": "-",
        "Payload": f" Agent rpc_loop - iteration:{iteration} started",
        "Hostname": HOSTNAME,
        "Logger": "openstack.neutron",
        "programname": prog,
        "@timestamp": fmt_iso(dt),
    }
    
    dt2 = dt + timedelta(milliseconds=int(elapsed*1000) + random.randint(0, 3))
    completed = {
        "Timestamp": fmt_human(dt2),
        "Pid": pid,
        "log_level": "INFO",
        "python_module": "neutron.plugins.ml2.drivers.openvswitch.agent.ovs_neutron_agent",
        "request_id": rid,
        "user_id": "-",
        "tenant_id": "-",
        "domain_id": "-",
        "system_scope": "-",
        "user_domain": "-",
        "project_domain": "-",
        "Payload": (f" Agent rpc_loop - iteration:{iteration} completed. "
                    f"Processed ports statistics: {{'regular': {{'added': {added}, "
                    f"'updated': {updated}, 'removed': {removed}}}}}. Elapsed:{elapsed}"),
        "Hostname": HOSTNAME,
        "Logger": "openstack.neutron",
        "programname": prog,
        "@timestamp": fmt_iso(dt2),
    }
    
    return [started, completed]


def make_request_log(dt, prog, method, path, status, length, *,
                     microversion="-", elapsed_seconds=None):
    """Logs request typique de nova-api / placement-api / cinder-api"""
    if elapsed_seconds is None:
        elapsed_seconds = round(random.uniform(0.001, 0.01), 6)
    pid = random.choice(SERVICE_PIDS.get(prog, ["22"]))
    
    if 'nova' in prog:
        python_module = "nova.api.openstack.requestlog"
        payload = (f" {CLIENT_IP} \"{method} {path}\" status: {status} "
                   f"len: {length} microversion: {microversion} time: {elapsed_seconds:.6f}")
        logger = "openstack.nova"
    elif 'placement' in prog:
        python_module = "placement.requestlog"
        payload = (f" {CLIENT_IP} \"{method} {path}\" status: {status} "
                   f"len: {length} microversion: {microversion}")
        logger = "openstack.placement"
    elif 'cinder' in prog:
        python_module = "cinder.api.openstack.wsgi"
        payload = f" {CLIENT_IP} \"{method} {path}\" status: {status} len: {length}"
        logger = "openstack.cinder"
    else:
        python_module = "wsgi"
        payload = f" {CLIENT_IP} \"{method} {path}\" status: {status} len: {length}"
        logger = f"openstack.{prog.split('-')[0]}"
    
    return {
        "Timestamp": fmt_human(dt),
        "Pid": pid,
        "log_level": "INFO",
        "python_module": python_module,
        "Payload": payload,
        "Hostname": HOSTNAME,
        "Logger": logger,
        "programname": prog,
        "@timestamp": fmt_iso(dt),
    }


# TEMPLATES DE WORKFLOWS UTILISATEUR

def workflow_auth_token(dt, user_id=None, tenant_id=None):
    """POST /v3/auth/tokens - retourne liste de logs"""
    if user_id is None:
        user_id = random.choice(USERS)
    if tenant_id is None:
        tenant_id = random.choice(TENANTS)
    
    logs = []
    duration = random.randint(150, 400)
    
    # Le POST sur keystone-uwsgi
    logs.append(make_uwsgi_log(
        dt, "keystone-uwsgi", "POST", "/v3/auth/tokens", 201,
        gen_bytes=random.choice([3960, 3994, 4077, 4080, 4138, 4141, 4146]),
        duration_ms=duration,
        headers_count=6, headers_bytes=random.randint(385, 410),
        total_bytes=random.choice([218, 219])
    ))
    
    # Warning Truncating password (~95% des cas)
    if random.random() < 0.95:
        logs.append(make_python_log(
            dt + timedelta(milliseconds=random.randint(5, 50)),
            "keystone", "WARNING", "keystone.common.password_hashing",
            " Truncating password to algorithm specific maximum length 72 characters.",
            user_id="-", tenant_id="-"
        ))
    
    return logs, user_id, tenant_id


def workflow_horizon_login(dt, user_id=None):
    """Login Horizon"""
    if user_id is None:
        user_id = random.choice(USERS)
    logs = []
    # POST /auth/login/
    logs.append(make_uwsgi_log(
        dt, "horizon-uwsgi", "POST", "/auth/login/", 302,
        gen_bytes=0, duration_ms=random.randint(800, 2500),
        headers_count=7, headers_bytes=random.randint(300, 600),
        total_bytes=random.randint(280, 450)
    ))
    return logs, user_id


def workflow_create_vm(dt, user_id=None, tenant_id=None):
    """Création VM - workflow complet inter-services"""
    if user_id is None:
        user_id = random.choice(USERS)
    if tenant_id is None:
        tenant_id = random.choice(TENANTS)
    
    logs = []
    cur = dt
    server_uuid = gen_uuid()
    port_uuid = gen_uuid()
    allocation_uuid = gen_uuid()
    
    # 1. Auth token
    auth_logs, _, _ = workflow_auth_token(cur, user_id, tenant_id)
    logs.extend(auth_logs)
    cur += timedelta(milliseconds=random.randint(200, 500))
    
    # 2. POST /v2.1/servers (latence ~1.0-1.5s en normal)
    duration = random.randint(900, 1500)
    logs.append(make_uwsgi_log(
        cur, "nova-api-uwsgi", "POST", "/v2.1/servers", 202,
        gen_bytes=380, duration_ms=duration,
        headers_count=10, headers_bytes=451,
        total_bytes=random.randint(280, 350)
    ))
    # warning neutronclient deprecated
    for _ in range(random.randint(1, 3)):
        logs.append(make_python_log(
            cur + timedelta(milliseconds=random.randint(50, 200)),
            "nova-api", "WARNING", "neutronclient.v2_0.client",
            " The python binding code in neutronclient is deprecated in favor of OpenstackSDK, please use that as this will be removed in a future release."
        ))
    cur += timedelta(milliseconds=duration + random.randint(50, 200))
    
    # 3. POST /v2.0/ports (neutron)
    logs.append(make_uwsgi_log(
        cur, "neutron-server-uwsgi", "POST", "/v2.0/ports", 201,
        gen_bytes=random.randint(700, 800), duration_ms=random.randint(300, 500),
        headers_count=4, headers_bytes=162
    ))
    cur += timedelta(milliseconds=random.randint(100, 300))
    
    # 4. PUT /v2.0/ports/UUID
    logs.append(make_uwsgi_log(
        cur, "neutron-server-uwsgi", "PUT", f"/v2.0/ports/{port_uuid}", 200,
        gen_bytes=random.randint(900, 1000), duration_ms=random.randint(400, 600),
        headers_count=4, headers_bytes=157
    ))
    cur += timedelta(milliseconds=random.randint(100, 300))
    
    # 5. PUT /allocations/UUID (placement)
    logs.append(make_uwsgi_log(
        cur, "placement-api-uwsgi", "PUT", f"/allocations/{allocation_uuid}", 204,
        gen_bytes=0, duration_ms=random.randint(50, 150),
        headers_count=5, headers_bytes=199, total_bytes=random.randint(280, 320)
    ))
    cur += timedelta(milliseconds=random.randint(100, 300))
    
    # 6. POST /v2.1/os-server-external-events
    logs.append(make_uwsgi_log(
        cur, "nova-api-uwsgi", "POST", "/v2.1/os-server-external-events", 200,
        gen_bytes=179, duration_ms=random.randint(40, 100),
        headers_count=9, headers_bytes=357
    ))
    
    # nova-compute warnings (deprecated neutronclient) en parallèle
    for _ in range(random.randint(2, 4)):
        logs.append(make_python_log(
            cur + timedelta(milliseconds=random.randint(-2000, 100)),
            "nova-compute", "WARNING", "neutronclient.v2_0.client",
            " The python binding code in neutronclient is deprecated in favor of OpenstackSDK, please use that as this will be removed in a future release."
        ))
    
    return logs, server_uuid


def workflow_delete_vm(dt, server_uuid=None, user_id=None, tenant_id=None):
    """Suppression VM"""
    if server_uuid is None:
        server_uuid = gen_uuid()
    if user_id is None:
        user_id = random.choice(USERS)
    if tenant_id is None:
        tenant_id = random.choice(TENANTS)
    
    logs = []
    cur = dt
    auth_logs, _, _ = workflow_auth_token(cur, user_id, tenant_id)
    logs.extend(auth_logs)
    cur += timedelta(milliseconds=random.randint(150, 350))
    
    # GET server
    logs.append(make_uwsgi_log(
        cur, "nova-api-uwsgi", "GET", f"/v2.1/servers/{server_uuid}", 200,
        gen_bytes=random.randint(1500, 2500), duration_ms=random.randint(50, 200),
        headers_count=10, headers_bytes=400
    ))
    cur += timedelta(milliseconds=random.randint(100, 300))
    
    # DELETE server
    logs.append(make_uwsgi_log(
        cur, "nova-api-uwsgi", "DELETE", f"/v2.1/servers/{server_uuid}", 204,
        gen_bytes=0, duration_ms=random.randint(300, 700),
        headers_count=5, headers_bytes=200
    ))
    
    return logs


def workflow_create_volume(dt, user_id=None, tenant_id=None):
    """Création volume Cinder"""
    if user_id is None:
        user_id = random.choice(USERS)
    if tenant_id is None:
        tenant_id = random.choice(TENANTS)
    
    logs = []
    cur = dt
    volume_uuid = gen_uuid()
    
    auth_logs, _, _ = workflow_auth_token(cur, user_id, tenant_id)
    logs.extend(auth_logs)
    cur += timedelta(milliseconds=random.randint(150, 400))
    
    # POST /v3/volumes
    logs.append(make_uwsgi_log(
        cur, "cinder-api-uwsgi", "POST", "/v3/volumes", 202,
        gen_bytes=random.randint(800, 1100), duration_ms=random.randint(200, 600),
        headers_count=6, headers_bytes=300
    ))
    cur += timedelta(milliseconds=random.randint(50, 150))
    
    # cinder request log
    logs.append(make_request_log(
        cur, "cinder-api", "POST", "/v3/volumes",
        202, random.randint(800, 1100)
    ))
    
    return logs, volume_uuid


def workflow_create_network(dt, user_id=None, tenant_id=None):
    """Création réseau"""
    if user_id is None:
        user_id = random.choice(USERS)
    if tenant_id is None:
        tenant_id = random.choice(TENANTS)
    
    logs = []
    cur = dt
    network_uuid = gen_uuid()
    
    auth_logs, _, _ = workflow_auth_token(cur, user_id, tenant_id)
    logs.extend(auth_logs)
    cur += timedelta(milliseconds=random.randint(150, 350))
    
    # POST /v2.0/networks
    logs.append(make_uwsgi_log(
        cur, "neutron-server-uwsgi", "POST", "/v2.0/networks", 201,
        gen_bytes=random.randint(500, 600), duration_ms=random.randint(300, 600),
        headers_count=4, headers_bytes=162
    ))
    cur += timedelta(milliseconds=random.randint(200, 600))
    
    # POST /v2.0/subnets
    logs.append(make_uwsgi_log(
        cur, "neutron-server-uwsgi", "POST", "/v2.0/subnets", 201,
        gen_bytes=random.randint(600, 700), duration_ms=random.randint(250, 500),
        headers_count=4, headers_bytes=162
    ))
    
    return logs, network_uuid


# GÉNÉRATEURS DE BRUIT DE FOND

def gen_healthcheck(dt, prog):
    """GET /healthcheck - retourne 200, ~1-50ms"""
    duration = random.choice([0, 1, 1, 2, 2, 2, 3, 3, 5, 8, 12, 18, 25, 40, 60])
    headers_bytes = 98 if 'neutron' in prog or 'cinder' in prog or 'glance' in prog or 'heat' in prog else 164
    headers_count = 3 if headers_bytes == 98 else 4
    return make_uwsgi_log(
        dt, prog, "GET", "/healthcheck", 200,
        gen_bytes=random.choice([0, 2]),
        duration_ms=duration,
        headers_count=headers_count, headers_bytes=headers_bytes,
        total_bytes=random.choice([218, 239, 240, 322, 324])
    )


def gen_root_get(dt, prog):
    """GET / - placement-api, nova-api, nova-metadata"""
    duration = random.choice([0, 1, 1, 1, 2, 2, 3, 5, 8, 12, 25])
    if 'placement' in prog:
        gen = 136
        headers_bytes = 224
        status = 200
    elif 'nova-metadata' in prog:
        gen = 98
        headers_bytes = 99
        status = 200
    elif 'nova-api' in prog:
        gen = random.choice([363, 379])
        headers_bytes = 91
        status = 200
    else:
        gen = 100
        headers_bytes = 100
        status = 200
    return make_uwsgi_log(
        dt, prog, "GET", "/", status,
        gen_bytes=gen, duration_ms=duration,
        headers_count=random.choice([3, 6]), headers_bytes=headers_bytes
    )


def gen_horizon_options(dt):
    """OPTIONS / => 302 (Horizon poll typique en arrière-plan)"""
    duration = random.choice([0, 1, 1, 2, 2, 3, 5, 10, 50, 100])
    return make_uwsgi_log(
        dt, "horizon-uwsgi", "OPTIONS", "/", 302,
        gen_bytes=0, http_version="1.0",
        duration_ms=duration,
        headers_count=7, headers_bytes=225,
        total_bytes=random.choice([221, 222])
    )


def gen_horizon_get_root(dt):
    """GET / Horizon"""
    return make_uwsgi_log(
        dt, "horizon-uwsgi", "GET", "/", random.choice([200, 302]),
        gen_bytes=random.randint(2000, 25000),
        duration_ms=random.randint(50, 400),
        headers_count=7, headers_bytes=random.randint(300, 600)
    )


def gen_placement_request_log(dt, path="/", status=200, length=136):
    """request log de placement-api"""
    return make_request_log(dt, "placement-api", "GET", path, status, length, microversion="1.0")


def gen_nova_request_log(dt, path="/", status=200, length=363):
    """request log de nova-api"""
    return make_request_log(dt, "nova-api", "GET", path, status, length,
                            microversion="-",
                            elapsed_seconds=round(random.uniform(0.001, 0.01), 6))


def gen_neutron_periodic_check(dt):
    """Logs périodiques de neutron-periodic-workers (rare)"""
    return make_python_log(
        dt, "neutron-periodic-workers", "WARNING",
        "neutron.plugins.ml2.plugin.DhcpAgentSchedulerDbMixin",
        " Function 'neutron.plugins.ml2.plugin.DhcpAgentSchedulerDbMixin.remove_networks_from_down_agents' run outlasted interval by 0.31 sec"
    )


def gen_keystone_get_token_validate(dt):
    """GET /v3/auth/tokens (validation token)"""
    return make_uwsgi_log(
        dt, "keystone-uwsgi", "GET", "/v3/auth/tokens", 200,
        gen_bytes=random.choice([3960, 3994, 4138]),
        duration_ms=random.randint(20, 80),
        headers_count=5, headers_bytes=320
    )


def gen_keystone_random_browse(dt):
    """Endpoints divers de keystone (browse Horizon)"""
    choices = [
        ("GET", f"/v3/users/{random.choice(USERS)}/projects", 200, random.randint(800, 1500)),
        ("GET", "/v3", 200, 247),
        ("GET", "/v3/domains/default", 200, 200),
        ("GET", "/", 300, 578),
    ]
    method, path, status, length = random.choice(choices)
    return make_uwsgi_log(
        dt, "keystone-uwsgi", method, path, status,
        gen_bytes=length, duration_ms=random.randint(5, 50),
        headers_count=random.randint(4, 6), headers_bytes=random.randint(150, 350)
    )


def gen_neutron_browse(dt):
    """Browse neutron API (security-groups, ports, etc.)"""
    choices = [
        ("GET", "/v2.0/ports?device_owner=network:dhcp", 200),
        ("GET", "/v2.0/ports", 200),
        ("GET", "/v2.0/security-groups", 200),
        ("GET", "/v2.0/security-groups?...", 200),
        ("GET", "/v2.0/extensions", 200),
        ("GET", "/v2.0/networks", 200),
    ]
    method, path, status = random.choice(choices)
    return make_uwsgi_log(
        dt, "neutron-server-uwsgi", method, path, status,
        gen_bytes=random.randint(500, 5000),
        duration_ms=random.randint(20, 150),
        headers_count=4, headers_bytes=162
    )


def gen_cinder_browse(dt):
    choices = [
        ("GET", "/v3/limits", 200),
        ("GET", f"/v3/volumes/{gen_uuid()}", 200),
        ("GET", "/v3/volumes", 200),
        ("GET", "/v3/volumes/detail", 200),
    ]
    method, path, status = random.choice(choices)
    return make_uwsgi_log(
        dt, "cinder-api-uwsgi", method, path, status,
        gen_bytes=random.randint(300, 3000),
        duration_ms=random.randint(20, 200),
        headers_count=4, headers_bytes=162
    )


def gen_placement_resource_query(dt):
    """GET /resource_providers/UUID/allocations etc."""
    return make_uwsgi_log(
        dt, "placement-api-uwsgi", "GET",
        f"/resource_providers/{gen_uuid()}/allocations", 200,
        gen_bytes=random.randint(200, 800),
        duration_ms=random.randint(5, 40),
        headers_count=6, headers_bytes=224
    )


