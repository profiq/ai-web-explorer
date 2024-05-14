import logging
from . import promptrepo
import json
import sys

logging.basicConfig(level=logging.INFO)


def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    logging.info("Creating dataset for fine-tuning")
    prompt_base = promptrepo.get_prompt("page_title")
    explored_pages = set()
    records = []

    with open("data/titles.jsonl") as fd:
        while line := fd.readline():
            page = json.loads(line)
            page_key = (page["title"], page["url"])

            if page_key in explored_pages:
                continue

            explored_pages.add(page_key)
            prompt = prompt_base.prompt_with_data(html=page["html"])
            response = json.dumps({"title": page["title"]})

            record = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            }

            records.append(record)

    train_size = int(len(records) * 0.75)

    with open(train_file, "w") as fd:
        for record in records[:train_size]:
            fd.write(json.dumps(record) + "\n")

    with open(test_file, "w") as fd:
        for record in records[train_size:]:
            fd.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
