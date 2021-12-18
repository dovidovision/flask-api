id=`ps -ef | grep api.py | grep python | cut -d \t -f 2 | sed 's/^ *//' | cut -d ' ' -f 1`
`kill -9 $id`
