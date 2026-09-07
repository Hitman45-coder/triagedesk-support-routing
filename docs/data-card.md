# BANKING77 data card

## Intended use

Intent-classification development for the TriageDesk portfolio demo. The data is not representative of every support domain, language, customer population, or production banking workflow.

## Source and license

BANKING77 is published by PolyAI under CC BY 4.0. The source repository and citation are recorded in [DATA_LICENSE.md](../DATA_LICENSE.md). The current raw snapshot was fetched from immutable commit `57ec275d8078af65b7731c2a98be812d844a6d6b` on 2026-09-07; the manifest records file hashes and URLs.

## Structure

The publisher provides 10,003 training examples, 3,080 test examples, and 77 English banking intents. Each row contains `text` and `category`.

## Quality and known issues

The validator checks columns, empty values, label membership, and normalized duplicate overlap. The snapshot contains seven normalized train/test duplicate texts. They are recorded rather than silently ignored; the training command excludes those training rows from development fitting while leaving the official test file unchanged.

The examples are short and expert-annotated. They do not contain the operational policy needed to answer a banking question, so TriageDesk uses them for routing only.

## Privacy

The dataset is a public benchmark. The UI warns users not to submit real account or card information. User messages are not added to the training or retrieval corpus by the API.
