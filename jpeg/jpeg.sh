#!/usr/bin/env bash
 cat $1.txt |
  python3.11 jpeg/encode.py $2 |
  tee $1.jpg |
  python3.11 jpeg/decode.py > "$12.txt"