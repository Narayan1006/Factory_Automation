from influxdb_client import InfluxDBClient
import yaml

with open("config/twin_config.yaml") as f:
    cfg = yaml.safe_load(f)

client = InfluxDBClient(
    url=cfg["influxdb"]["url"],
    token=cfg["influxdb"]["token"],
    org=cfg["influxdb"]["org"],
)
query_api = client.query_api()

query = f"""
from(bucket: "{cfg['influxdb']['bucket']}")
  |> range(start: -30d)
  |> group(columns: ["_measurement", "_field"])
  |> count()
"""

try:
    tables = query_api.query(query)
    print("InfluxDB Telemetry Verification:")
    count_found = 0
    for table in tables:
        for record in table.records:
            count_found += record.get_value()
            print(f"Measurement: {record.get_measurement()}, Field: {record.get_field()} -> Count: {record.get_value()}")
    if count_found == 0:
        print("No records found in -30d range. Checking all time...")
        query_all = f"""
        from(bucket: "{cfg['influxdb']['bucket']}")
          |> range(start: 0)
          |> group(columns: ["_measurement", "_field"])
          |> count()
        """
        tables_all = query_api.query(query_all)
        for table in tables_all:
            for record in table.records:
                print(f"Measurement: {record.get_measurement()}, Field: {record.get_field()} -> Count: {record.get_value()}")
except Exception as e:
    print(f"Query error: {e}")
