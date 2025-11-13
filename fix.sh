#!/bin/bash
set -e

echo "ruff format ."
uvx ruff format .

echo "ruff check --fix"
uvx ruff check --fix

echo "ty check"
uvx ty check
