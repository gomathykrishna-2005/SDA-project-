from confluent_kafka import Producer
import pandas as pd
import json

CSV_FILE = "Data/cic_0.005km_78.csv"
TOPIC = "network-events"
BROKER = "localhost:9092"

producer = Producer({"bootstrap.servers": BROKER})

df = pd.read_csv(CSV_FILE)
print("Dataset shape:", df.shape)

for index, row in df.head(100).iterrows():
    message = row.to_dict()
    producer.produce(
        TOPIC,
        key=str(index),
        value=json.dumps(message)
    )
    producer.poll(0)

producer.flush()
print("100 records sent to Kafka successfully.")
