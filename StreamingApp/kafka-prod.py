import time
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError

prod = KafkaProducer(bootstrap_servers=['localhost:9092'])

count = 0
with open("./data/dataset.csv", "rb") as f:
    for line in f:
        count += 1
        try:
            record_metadata = prod.send("test", line).get(timeout=20)
            print(record_metadata.topic)
            print(record_metadata.partition)
            print(record_metadata.offset)
        except KafkaError:
            logging.exception()
        time.sleep(1);
