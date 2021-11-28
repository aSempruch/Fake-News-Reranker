#!/bin/bash
# set environment variable galago=<galago bin file>

$galago build --fileType=trectext --inputPath="data/collection.trec" --indexPath="data/index" --stemmedPostings=true --stemmer+krovetz