from confluent_kafka import Consumer
from influxdb_client import InfluxDBClient, Point
import json

# -----------------------------
# Kafka configuration
# -----------------------------
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "network-events"
GROUP_ID = "influx-storage"

# -----------------------------
# InfluxDB configuration
# -----------------------------
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "msana-token-123456789"
INFLUX_ORG = "msana"
INFLUX_BUCKET = "network-events"

# Kafka consumer
consumer = Consumer({
    "bootstrap.servers": KAFKA_BROKER,
    "group.id": GROUP_ID,
    "auto.offset.reset": "earliest"
})
consumer.subscribe([KAFKA_TOPIC])

# InfluxDB client
client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)
write_api = client.write_api()

print("Kafka → InfluxDB consumer started...")
print("Waiting for messages...")

count = 0

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print("Kafka error:", msg.error())
            continue

        try:
            data = json.loads(msg.value().decode("utf-8"))
            point = Point("network_events")

            # Store dataset fields
            for key, value in data.items():
                if value is None:
                    continue
                try:
                    point = point.field(key, float(value))
                except (ValueError, TypeError):
                    point = point.field(key, str(value))

            write_api.write(
                bucket=INFLUX_BUCKET,
                org=INFLUX_ORG,
                record=point
            )
            count += 1
            print(f"Stored record {count}")

        except json.JSONDecodeError:
            print("Skipped non-JSON Kafka message")

except KeyboardInterrupt:
    print("\nStopping consumer...")

finally:
    consumer.close()
    client.close()
