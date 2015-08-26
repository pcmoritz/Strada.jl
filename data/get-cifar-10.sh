#!/usr/bin/env sh
ARCHIVE=cifar-10-binary.tar.gz
wget -c http://www.cs.toronto.edu/~kriz/$ARCHIVE
echo Unpacking archive...
tar xf $ARCHIVE
