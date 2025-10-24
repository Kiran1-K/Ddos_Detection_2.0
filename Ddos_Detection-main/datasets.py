import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Parameters ---
train_rows = 300
test_normal_rows = 100
test_ddos_rows = 50

# --- Helper functions ---
def random_ip(subnet="192.168.1."):
    return subnet + str(np.random.randint(1, 255))

def generate_normal_row(ts):
    protocol = np.random.choice([6, 17, 1])  # TCP, UDP, ICMP
    packet_size = np.random.randint(200, 600)
    duration = round(np.random.uniform(0.3, 1.5), 2)
    flags = np.random.choice(["SYN", "ACK", "-", "FIN"])
    pkt_per_sec = np.random.randint(50, 200)
    syn_count = 1 if flags=="SYN" else 0
    ack_count = 1 if flags=="ACK" else 0
    unique_ips = np.random.randint(1,5)
    return [ts, random_ip(), "10.0.0.1", protocol, packet_size, duration, flags, pkt_per_sec, syn_count, ack_count, unique_ips]

def generate_ddos_row(ts):
    protocol = np.random.choice([6,17])  # TCP or UDP
    packet_size = np.random.randint(40, 80)  # Small packets
    duration = round(np.random.uniform(0.05, 0.2), 2)
    flags = "SYN" if protocol==6 else "-"
    pkt_per_sec = np.random.randint(1000, 2000)  # High packet rate
    syn_count = 1 if flags=="SYN" else 0
    ack_count = 0
    unique_ips = np.random.randint(50, 150)
    return [ts, random_ip("10.1.1."), "10.0.0.1", protocol, packet_size, duration, flags, pkt_per_sec, syn_count, ack_count, unique_ips]

# --- Generate training data ---
start_time = datetime(2025,10,24,10,0,0)
train_data = []
for i in range(train_rows):
    ts = start_time + timedelta(seconds=i)
    train_data.append(generate_normal_row(ts))

train_df = pd.DataFrame(train_data, columns=["Timestamp","Src_IP","Dst_IP","Protocol","Packet_Size","Duration","Flags",
                                             "Packets_Per_Second","SYN_Count","ACK_Count","Unique_IPs_Per_Second"])
train_df.to_csv("train_normal.csv", index=False)

# --- Generate testing data ---
test_data = []
for i in range(test_normal_rows):
    ts = start_time + timedelta(seconds=i)
    test_data.append(generate_normal_row(ts))

for i in range(test_ddos_rows):
    ts = start_time + timedelta(seconds=test_normal_rows+i)
    test_data.append(generate_ddos_row(ts))

test_df = pd.DataFrame(test_data, columns=["Timestamp","Src_IP","Dst_IP","Protocol","Packet_Size","Duration","Flags",
                                           "Packets_Per_Second","SYN_Count","ACK_Count","Unique_IPs_Per_Second"])
test_df.to_csv("test_normal_ddos.csv", index=False)

print("Datasets generated: train_normal.csv (normal) & test_normal_ddos.csv (normal + DDoS)")