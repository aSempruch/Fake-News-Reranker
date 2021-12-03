#!/bin/bash
# set environment variable galago=<galago bin file>

$galago batch-search data/queries.json --index=data/index --requested=10 > data/galago_output.txt
