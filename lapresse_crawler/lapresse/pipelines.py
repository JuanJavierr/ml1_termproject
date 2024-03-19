# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json
from pathlib import Path


class LapressePipeline:
    def open_spider(self, spider):
        self.output_folder = Path("output/")

        self.output_folder.mkdir()

    def process_item(self, item, spider):
        with open(self.output_folder / f"{item['date']}.json", mode="+a") as outfile:
            outfile.write(json.dumps(item))
        return item
