#!/usr/bin/env python3
"""
Génération du TEST SET avec 7 anomalies labellisées.

Output :
  - data/test_anomalies/test_day.json  (NDJSON brut)
  - data/test_anomalies/labels.csv     (window_id → is_anomaly + type)
"""
import sys
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Import genlib qui est dans scripts/
sys.path.insert(0, str(Path(__file__).parent))
from genlib import (
    USERS, TENANTS, CLIENT_IP, HOSTNAME,
    fmt_iso, fmt_human, fmt_uwsgi_ts, gen_uuid,
    make_python_log, make_python_log_simple, make_uwsgi_log,
    make_neutron_ovs_rpc_loop, make_request_log,
    workflow_auth_token, workflow_horizon_login,
    workflow_create_vm, workflow_create_volume, workflow_create_network,
    workflow_delete_vm,
    gen_healthcheck, gen_root_get, gen_horizon_options, gen_horizon_get_root,
    gen_placement_request_log, gen_nova_request_log,
    gen_neutron_periodic_check, gen_keystone_random_browse,
    gen_neutron_browse, gen_cinder_browse, gen_placement_resource_query,
    UWSGI_PIDS, SERVICE_PIDS,
)

# Different seed from training (training used 20260427-20260503)
random.seed(20260508)

DAY = datetime(2026, 5, 8)  # vendredi
START = DAY.replace(hour=8, minute=45)
END = DAY.replace(hour=17, minute=20)

logs = []
anomaly_intervals = []  # liste de (start_dt, end_dt, anomaly_type, severity)


def add(dt, obj):
    logs.append((dt, obj))


def add_workflow(workflow_logs):
    for log in workflow_logs:
        ts = log.get('@timestamp', '').split('.')[0]
        try:
            dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S')
            logs.append((dt, log))
        except:
            pass


def hc_interval(dt):
    h = dt.hour
    m = dt.minute
    if (h == 12 and m >= 30) or (h == 13 and m < 30):
        return random.uniform(7.0, 9.0)
    if 9 <= h < 12 or 14 <= h < 16:
        return random.uniform(4.5, 5.5)
    if h == 8 or h >= 16:
        return random.uniform(5.0, 6.5)
    return random.uniform(5.0, 6.0)


# 1. BASELINE SAINE - Healthchecks
print("Generating healthchecks...", file=sys.stderr)

HEALTHCHECK_SERVICES_HC = [
    "cinder-api-uwsgi", "keystone-uwsgi", "neutron-server-uwsgi",
    "glance-api-uwsgi", "heat-api-uwsgi", "heat-api-cfn-uwsgi",
]
ROOT_GET_SERVICES = [
    "placement-api-uwsgi", "nova-api-uwsgi", "nova-metadata-uwsgi",
]

# ANOMALIE 4 : silence Neutron-OVS entre 13:15 et 13:27
NEUTRON_SILENCE_START = DAY.replace(hour=13, minute=15)
NEUTRON_SILENCE_END = DAY.replace(hour=13, minute=27)

hc_t = START
while hc_t < END:
    interval = hc_interval(hc_t)
    for svc in HEALTHCHECK_SERVICES_HC:
        offset_ms = random.randint(0, int(interval * 1000) - 100)
        t = hc_t + timedelta(milliseconds=offset_ms)
        if t < END:
            add(t, gen_healthcheck(t, svc))
    for svc in ROOT_GET_SERVICES:
        offset_ms = random.randint(0, int(interval * 1000) - 100)
        t = hc_t + timedelta(milliseconds=offset_ms)
        if t < END:
            add(t, gen_root_get(t, svc))
            if svc == "placement-api-uwsgi":
                add(t + timedelta(milliseconds=1), gen_placement_request_log(t))
            elif svc == "nova-api-uwsgi":
                add(t + timedelta(milliseconds=1),
                    gen_nova_request_log(t, status=200, length=random.choice([363, 379])))
            elif svc == "nova-metadata-uwsgi":
                if random.random() < 0.4:
                    add(t + timedelta(milliseconds=1),
                        make_request_log(t, "nova-metadata", "GET", "/", 200, 98))
    hc_t += timedelta(seconds=interval)

# 2. NEUTRON-OVS rpc_loop (sauf pendant l'anomalie 4)
print("Generating neutron rpc_loop (with anomaly 4: silence)...", file=sys.stderr)

iter_num = random.randint(15000, 20000)
rpc_t = START
while rpc_t < END:
    # ANOMALIE 4 : pas de logs neutron-ovs pendant 13:15-13:27
    if NEUTRON_SILENCE_START <= rpc_t < NEUTRON_SILENCE_END:
        rpc_t += timedelta(seconds=random.uniform(4.5, 6.5))
        iter_num += 1
        continue

    pair = make_neutron_ovs_rpc_loop(rpc_t, iter_num)
    for p in pair:
        ts = p['@timestamp'].split('.')[0]
        dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S')
        logs.append((dt, p))
    iter_num += 1
    rpc_t += timedelta(seconds=random.uniform(4.5, 6.5))

# Enregistrer l'anomalie 4
anomaly_intervals.append((
    NEUTRON_SILENCE_START, NEUTRON_SILENCE_END,
    "service_silence_neutron_ovs",
    "high"
))

# 3. SESSIONS HORIZON
print("Generating horizon sessions...", file=sys.stderr)

sessions_horizon = [
    (DAY.replace(hour=8, minute=58), DAY.replace(hour=12, minute=20)),
    (DAY.replace(hour=13, minute=45), DAY.replace(hour=15, minute=55)),
    (DAY.replace(hour=16, minute=15), DAY.replace(hour=17, minute=10)),
]

for start_s, end_s in sessions_horizon:
    if start_s < START:
        start_s = START
    if end_s > END:
        end_s = END
    t = start_s
    while t < end_s:
        h = t.hour
        if (h == 12 and t.minute >= 30) or (h == 13 and t.minute < 30):
            poll_proba = 0.20
            interval = random.uniform(4.0, 8.0)
        elif 9 <= h < 12 or 14 <= h < 16:
            poll_proba = 0.95
            interval = random.uniform(1.8, 2.5)
        else:
            poll_proba = 0.7
            interval = random.uniform(2.0, 3.5)
        if random.random() < poll_proba:
            add(t, gen_horizon_options(t))
        if random.random() < 0.04:
            add(t + timedelta(milliseconds=random.randint(50, 300)),
                gen_horizon_get_root(t))
        t += timedelta(seconds=interval)

# 4. ACTIVITÉ UTILISATEUR LÉGITIME
print("Generating user activity...", file=sys.stderr)

main_user = USERS[2]
main_tenant = TENANTS[1]

activity_windows = [
    (DAY.replace(hour=9, minute=0), DAY.replace(hour=12, minute=15)),
    (DAY.replace(hour=13, minute=50), DAY.replace(hour=15, minute=50)),
    (DAY.replace(hour=16, minute=20), DAY.replace(hour=17, minute=10)),
]

# Logins
for win_idx in [0, 2]:
    start_w, _ = activity_windows[win_idx]
    login_t = start_w + timedelta(minutes=random.randint(2, 10))
    horizon_logs, _ = workflow_horizon_login(login_t, main_user)
    add_workflow(horizon_logs)
    auth_logs, _, _ = workflow_auth_token(login_t + timedelta(seconds=1),
                                            main_user, main_tenant)
    add_workflow(auth_logs)

# Auth tokens
total_tokens = random.randint(75, 110)
for _ in range(total_tokens):
    win = random.choice(activity_windows)
    t = win[0] + timedelta(seconds=random.uniform(0, (win[1]-win[0]).total_seconds()))
    auth_logs, _, _ = workflow_auth_token(t, random.choice(USERS), random.choice(TENANTS))
    add_workflow(auth_logs)

# 1 VM + 1 volume + 1 réseau
t_vm = activity_windows[0][0] + timedelta(minutes=random.randint(15, 90))
vm_logs, vm_uuid = workflow_create_vm(t_vm, main_user, main_tenant)
add_workflow(vm_logs)

t_vol = activity_windows[1][0] + timedelta(minutes=random.randint(5, 30))
vol_logs, vol_uuid = workflow_create_volume(t_vol, main_user, main_tenant)
add_workflow(vol_logs)

t_net = activity_windows[0][0] + timedelta(minutes=random.randint(30, 120))
net_logs, _ = workflow_create_network(t_net, main_user, main_tenant)
add_workflow(net_logs)

# Browse
for _ in range(random.randint(25, 40)):
    win = random.choice(activity_windows)
    t = win[0] + timedelta(seconds=random.uniform(0, (win[1]-win[0]).total_seconds()))
    add(t, gen_keystone_random_browse(t))

for _ in range(random.randint(15, 25)):
    win = random.choice(activity_windows)
    t = win[0] + timedelta(seconds=random.uniform(0, (win[1]-win[0]).total_seconds()))
    add(t, gen_neutron_browse(t))

for _ in range(random.randint(15, 25)):
    win = random.choice(activity_windows)
    t = win[0] + timedelta(seconds=random.uniform(0, (win[1]-win[0]).total_seconds()))
    add(t, gen_cinder_browse(t))

for _ in range(random.randint(15, 25)):
    win = random.choice(activity_windows)
    t = win[0] + timedelta(seconds=random.uniform(0, (win[1]-win[0]).total_seconds()))
    add(t, gen_placement_resource_query(t))

# Background services (INFO normaux)
for _ in range(random.randint(8, 15)):
    t = START + timedelta(seconds=random.randint(0, int((END-START).total_seconds())))
    add(t, make_python_log(
        t, "cinder-volume", "INFO", "cinder.volume.manager",
        " Updating volume status",
        user_id="-", tenant_id="-"
    ))

for _ in range(random.randint(5, 12)):
    t = START + timedelta(seconds=random.randint(0, int((END-START).total_seconds())))
    add(t, make_python_log(
        t, "nova-conductor", "INFO", "nova.compute.task_manager",
        " Periodic task running normally",
        user_id="-", tenant_id="-"
    ))

for _ in range(random.randint(3, 8)):
    t = START + timedelta(seconds=random.randint(0, int((END-START).total_seconds())))
    add(t, make_python_log_simple(
        t, "cinder-backup", "INFO", "cinder.backup.manager",
        "[-] Backup service is running normally"
    ))

# Quelques periodic warnings normaux
for _ in range(random.randint(2, 5)):
    t = START + timedelta(seconds=random.randint(0, int((END-START).total_seconds())))
    add(t, gen_neutron_periodic_check(t))

# Quelques 404 normaux
for _ in range(random.randint(2, 5)):
    win = random.choice(activity_windows)
    t = win[0] + timedelta(seconds=random.uniform(0, (win[1]-win[0]).total_seconds()))
    svc = random.choice(["nova-api-uwsgi", "cinder-api-uwsgi", "neutron-server-uwsgi"])
    if svc == "nova-api-uwsgi":
        path = f"/v2.1/servers/{gen_uuid()}"
    elif svc == "cinder-api-uwsgi":
        path = f"/v3/volumes/{gen_uuid()}"
    else:
        path = f"/v2.0/ports/{gen_uuid()}"
    add(t, make_uwsgi_log(t, svc, "GET", path, 404,
                           gen_bytes=random.randint(80, 150),
                           duration_ms=random.randint(20, 80),
                           headers_count=4, headers_bytes=162))

# 5. ANOMALIE 1 : Burst ERROR cinder-volume (09:30 → 09:34)
print("Injecting anomaly 1: cinder-volume ERROR burst...", file=sys.stderr)

ANOM1_START = DAY.replace(hour=9, minute=30, second=15)
ANOM1_END = DAY.replace(hour=9, minute=34, second=10)
n_errors = random.randint(45, 55)

for i in range(n_errors):
    progress = i / n_errors
    t_offset = (ANOM1_END - ANOM1_START).total_seconds() * progress
    t = ANOM1_START + timedelta(seconds=t_offset + random.uniform(-2, 2))
    if t < ANOM1_START or t > ANOM1_END:
        continue

    # ERROR cinder-volume avec différents messages plausibles
    error_messages = [
        "[-] Failed to attach volume to instance: VolumeAttachmentNotFound",
        "[-] Volume creation failed: Backend storage unreachable",
        "[-] Connection lost to storage backend, retrying...",
        "[-] Failed to delete volume: VolumeIsBusy",
    ]
    msg = random.choice(error_messages)
    add(t, make_python_log(
        t, "cinder-volume", "ERROR", "cinder.volume.manager", msg,
        user_id="-", tenant_id="-"
    ))

anomaly_intervals.append((ANOM1_START, ANOM1_END, "error_burst_cinder", "high"))

# 6. ANOMALIE 2 : Pic de latence Keystone (10:45 → 10:53)
print("Injecting anomaly 2: keystone latency spike...", file=sys.stderr)

ANOM2_START = DAY.replace(hour=10, minute=45)
ANOM2_END = DAY.replace(hour=10, minute=53)

# On génère des requêtes keystone normales mais avec des latences anormalement élevées
n_slow_requests = random.randint(40, 60)
for i in range(n_slow_requests):
    progress = i / n_slow_requests
    t_offset = (ANOM2_END - ANOM2_START).total_seconds() * progress
    t = ANOM2_START + timedelta(seconds=t_offset + random.uniform(-3, 3))
    if t < ANOM2_START or t > ANOM2_END:
        continue

    # Latence anormale : 1500ms à 8000ms (vs ~3ms normal, 60ms p95)
    duration = random.randint(1500, 8000)

    # Différents endpoints keystone
    endpoint = random.choice([
        ("POST", "/v3/auth/tokens", 201, random.choice([3960, 4138])),
        ("GET", "/v3/auth/tokens", 200, random.choice([3994, 4080])),
        ("GET", "/v3", 200, 247),
    ])
    method, path, status, gen_bytes = endpoint

    add(t, make_uwsgi_log(
        t, "keystone-uwsgi", method, path, status,
        gen_bytes=gen_bytes, duration_ms=duration,
        headers_count=5, headers_bytes=320
    ))

anomaly_intervals.append((ANOM2_START, ANOM2_END, "latency_spike_keystone", "medium"))

# 7. ANOMALIE 3 : Burst auth failures 401 (11:30 → 11:33)
print("Injecting anomaly 3: auth failures burst...", file=sys.stderr)

ANOM3_START = DAY.replace(hour=11, minute=30, second=20)
ANOM3_END = DAY.replace(hour=11, minute=33, second=15)
n_401 = random.randint(35, 50)

for i in range(n_401):
    progress = i / n_401
    t_offset = (ANOM3_END - ANOM3_START).total_seconds() * progress
    t = ANOM3_START + timedelta(seconds=t_offset + random.uniform(-1.5, 1.5))
    if t < ANOM3_START or t > ANOM3_END:
        continue

    # POST /v3/auth/tokens => 401
    add(t, make_uwsgi_log(
        t, "keystone-uwsgi", "POST", "/v3/auth/tokens", 401,
        gen_bytes=random.randint(120, 200),
        duration_ms=random.randint(120, 380),
        headers_count=5, headers_bytes=random.randint(180, 280)
    ))

    # WARNING associé
    add(t + timedelta(milliseconds=random.randint(5, 30)),
        make_python_log(
            t + timedelta(milliseconds=random.randint(5, 30)),
            "keystone", "WARNING",
            "keystone.common.wsgi",
            f" Authorization failed. The request you have made requires authentication. from {CLIENT_IP}: keystone.exception.Unauthorized: The request you have made requires authentication.",
            user_id="-", tenant_id="-"
        ))

anomaly_intervals.append((ANOM3_START, ANOM3_END, "auth_failures_burst", "high"))

# (ANOMALIE 4 déjà gérée plus haut : silence neutron-ovs)

# 8. ANOMALIE 5 : Nouveaux templates d'erreur (14:00 → 14:05)
print("Injecting anomaly 5: new error templates...", file=sys.stderr)

ANOM5_START = DAY.replace(hour=14, minute=0)
ANOM5_END = DAY.replace(hour=14, minute=5)

# Messages JAMAIS vus dans le training (vraiment nouveaux)
new_error_templates = [
    ("nova-compute", "ERROR", "nova.compute.manager",
     "[-] Hypervisor connection refused: libvirtd unreachable, attempting reconnection #{counter}"),
    ("neutron-server", "CRITICAL", "neutron.plugins.ml2.plugin",
     "[-] Mechanism driver 'openvswitch' failed in create_port_postcommit: PortBindingFailed"),
    ("glance-api", "ERROR", "glance.api.v2.images",
     "[-] Image upload corrupted: checksum mismatch on chunk #{counter}, expected {hash} got {hash}"),
    ("heat-engine", "ERROR", "heat.engine.stack",
     "[-] Stack convergence timeout: resource type=OS::Nova::Server stuck in CREATE_IN_PROGRESS"),
]

n_new_errors = random.randint(15, 25)
for i in range(n_new_errors):
    progress = i / n_new_errors
    t_offset = (ANOM5_END - ANOM5_START).total_seconds() * progress
    t = ANOM5_START + timedelta(seconds=t_offset + random.uniform(-2, 2))
    if t < ANOM5_START or t > ANOM5_END:
        continue

    prog, level, mod, template = random.choice(new_error_templates)
    msg = template.replace("{counter}", str(random.randint(1, 50))).replace(
        "{hash}", "0x" + gen_uuid()[:8])

    add(t, make_python_log(
        t, prog, level, mod, msg,
        user_id="-", tenant_id="-"
    ))

anomaly_intervals.append((ANOM5_START, ANOM5_END, "new_error_templates", "medium"))

# 9. ANOMALIE 6 : Volume anormal (DDoS-like) (15:30 → 15:36)
print("Injecting anomaly 6: traffic spike (DDoS-like)...", file=sys.stderr)

ANOM6_START = DAY.replace(hour=15, minute=30)
ANOM6_END = DAY.replace(hour=15, minute=36)

# Burst massif de requêtes legitimes (volume + 300%)
# Simulation : un script qui spam des requêtes auth ou healthcheck
n_burst = random.randint(2500, 3500)
for i in range(n_burst):
    progress = i / n_burst
    t_offset = (ANOM6_END - ANOM6_START).total_seconds() * progress
    t = ANOM6_START + timedelta(seconds=t_offset + random.uniform(-0.5, 0.5))
    if t < ANOM6_START or t > ANOM6_END:
        continue

    # 70% requêtes auth tokens (succès 201), 30% healthchecks
    if random.random() < 0.7:
        add(t, make_uwsgi_log(
            t, "keystone-uwsgi", "POST", "/v3/auth/tokens", 201,
            gen_bytes=random.choice([3960, 4138]),
            duration_ms=random.randint(150, 400),
            headers_count=6, headers_bytes=random.randint(385, 410)
        ))
    else:
        add(t, gen_healthcheck(t, "keystone-uwsgi"))

anomaly_intervals.append((ANOM6_START, ANOM6_END, "traffic_spike_ddos", "high"))

# 10. ANOMALIE 7 : Drift latence lent (16:00 → 17:00)
print("Injecting anomaly 7: slow latency drift...", file=sys.stderr)

ANOM7_START = DAY.replace(hour=16, minute=0)
ANOM7_END = DAY.replace(hour=17, minute=0)

# Sur 60 minutes, on injecte des requêtes nova-api avec latence qui dérive
# t=0 : 50ms, t=60 : 250ms (linéaire)
n_drift_requests = 360  # 6/min sur 60 min
for i in range(n_drift_requests):
    progress = i / n_drift_requests
    t_offset = (ANOM7_END - ANOM7_START).total_seconds() * progress
    t = ANOM7_START + timedelta(seconds=t_offset + random.uniform(-2, 2))
    if t < ANOM7_START or t > ANOM7_END:
        continue

    # Latence dérive : base = 50 + 200 * progress, + noise
    base_latency = 50 + 200 * progress
    duration = max(10, int(base_latency + random.gauss(0, 30)))

    # Sur des endpoints nova-api légitimes
    add(t, make_uwsgi_log(
        t, "nova-api-uwsgi", "GET", "/", 200,
        gen_bytes=random.choice([363, 379]),
        duration_ms=duration,
        headers_count=3, headers_bytes=91
    ))

anomaly_intervals.append((ANOM7_START, ANOM7_END, "latency_drift_nova", "low"))

# 11. TRI ET ÉCRITURE
print(f"\nTotal events generated: {len(logs)}", file=sys.stderr)
logs.sort(key=lambda x: x[0])

# Output dirs
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "test_anomalies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Logs NDJSON
out_logs = OUT_DIR / "test_day.json"
with open(out_logs, "w") as f:
    for dt, obj in logs:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(f"Wrote NDJSON: {out_logs} ({out_logs.stat().st_size/1024/1024:.1f} MB, {len(logs)} lines)",
      file=sys.stderr)

# 2. Labels par fenêtre 5min (window_id 0 = première fenêtre, on calcule les fenêtres comme aggregate.py)
# Fenêtre 5min step 1min, alignée sur la minute pleine
WINDOW_SEC = 300
STEP_SEC = 60
t_min = logs[0][0]
t_max = logs[-1][0]

# Fenêtres alignées sur la minute pleine
origin = t_min.replace(second=0, microsecond=0)
windows = []
cur_start = origin
while cur_start + timedelta(seconds=WINDOW_SEC) <= t_max + timedelta(seconds=STEP_SEC):
    cur_end = cur_start + timedelta(seconds=WINDOW_SEC)
    windows.append((cur_start, cur_end))
    cur_start += timedelta(seconds=STEP_SEC)

print(f"Generated {len(windows)} windows for labeling", file=sys.stderr)

# 3. Labellisation : une fenêtre est anomalie si elle CONTIENT au moins partiellement un intervalle d'anomalie
labels = []
for w_id, (w_start, w_end) in enumerate(windows):
    types_in_window = []
    severities_in_window = []
    for a_start, a_end, a_type, a_sev in anomaly_intervals:
        # Intersection non-vide entre [w_start, w_end] et [a_start, a_end]
        if max(w_start, a_start) < min(w_end, a_end):
            types_in_window.append(a_type)
            severities_in_window.append(a_sev)
    is_anom = len(types_in_window) > 0
    # Garder le type de la 1ère anomalie (en pratique, rare qu'il y en ait plusieurs)
    a_type = types_in_window[0] if types_in_window else ""
    a_sev = severities_in_window[0] if severities_in_window else ""
    labels.append({
        "window_id": w_id,
        "window_start": w_start.isoformat(),
        "window_end": w_end.isoformat(),
        "is_anomaly": int(is_anom),
        "anomaly_type": a_type,
        "anomaly_severity": a_sev,
    })

# Sauver labels CSV
import csv
out_labels = OUT_DIR / "labels.csv"
with open(out_labels, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["window_id", "window_start", "window_end",
                                      "is_anomaly", "anomaly_type", "anomaly_severity"])
    w.writeheader()
    w.writerows(labels)

n_anom_windows = sum(1 for l in labels if l["is_anomaly"])
print(f"Wrote labels: {out_labels} ({n_anom_windows} anomaly windows / {len(labels)} total = {100*n_anom_windows/len(labels):.1f}%)",
      file=sys.stderr)

# Résumé par type d'anomalie
print("\n=== Anomaly summary ===", file=sys.stderr)
from collections import Counter
type_counts = Counter(l["anomaly_type"] for l in labels if l["anomaly_type"])
for typ, count in type_counts.most_common():
    print(f"  {typ}: {count} windows", file=sys.stderr)

print(f"\nDone. Test set ready at {OUT_DIR}", file=sys.stderr)
