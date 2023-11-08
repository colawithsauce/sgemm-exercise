#!/bin/bash

/usr/bin/ctags -o \
  tags --languages=C,C++ --kinds-all=* --fields=* --extras=* \
  -R \
  ./hip/ \
  /opt/rocm/include/
