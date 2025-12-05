import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

def influxdb_client():
    # environment variables
    load_dotenv("../.env")

    INFLUXDB_UI_PORT   = os.getenv("INFLUXDB_UI_PORT", "8086")
    INFLUXDB_URL = f"http://influxdb:{INFLUXDB_UI_PORT}"
    INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "local-dev-token")
    INFLUX_ORG   = os.getenv("INFLUX_ORG", "quant")
    BUCKET       = os.getenv("INFLUX_BUCKET", "TBD")
    
    influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    return influx_client
