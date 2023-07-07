#!/bin/bash

train_filename=""
for file in "data"/train*
do
  cur_file=`ls -d "${file}"`
  train_filename=${train_filename}${cur_file}":"
done
train_filename=${train_filename%:}
echo ${train_filename}

test_filename=""
for file in "data"/test*
do
  cur_file=`ls -d "${file}"`
  test_filename=${test_filename}${cur_file}":"
done
test_filename=${test_filename%:}
echo ${test_filename}

train_cmd=`xxxxxx/python -u main.py --train_file_name="${train_filename}" --test_file_name="${test_filename}" > ./log.txt`

if [[ $? -ne 0 ]]; then
  echo "$(date +'%Y-%m-%d %H:%M'): train & test failed!" >> ./log.txt && exit 1
else
  echo "$(date +'%Y-%m-%d %H:%M'): train & test success!" >> ./log.txt && exit 0
fi
