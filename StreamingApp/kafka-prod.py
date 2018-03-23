import time
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError

prod = KafkaProducer(bootstrap_servers=['localhost:9092'])

with open("./data/temp-data.txt", "rb") as f:
    for line in f:
        try:
            record_metadata = prod.send("test", line).get(timeout=20)
            print(record_metadata.topic)
            print(record_metadata.partition)
            print(record_metadata.offset)
        except KafkaError:
            logging.exception()
        time.sleep(3)
