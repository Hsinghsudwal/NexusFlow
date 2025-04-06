# core/metadata_manager.py
import sqlite3
import boto3
import uuid

class MetadataManager:
    def __init__(self, config):
        self.mode = config['mode']
        if self.mode == 'local':
            self.conn = sqlite3.connect(config['metadata_store']['path'])
            self._init_sqlite()
        elif self.mode in ['cloud', 'localstack']:
            self.dynamodb = boto3.resource('dynamodb', endpoint_url=config.get('dynamodb_endpoint'))
            self.table = self.dynamodb.Table(config['metadata_store']['table'])

    def _init_sqlite(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS runs
                     (id TEXT, pipeline TEXT, status TEXT, metrics TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        self.conn.commit()

    def log_run(self, pipeline_name, status, metrics="{}"):
        run_id = str(uuid.uuid4())
        if self.mode == 'local':
            c = self.conn.cursor()
            c.execute("INSERT INTO runs (id, pipeline, status, metrics) VALUES (?, ?, ?, ?)",
                      (run_id, pipeline_name, status, metrics))
            self.conn.commit()
        else:
            self.table.put_item(Item={
                'id': run_id,
                'pipeline': pipeline_name,
                'status': status,
                'metrics': metrics
            })
