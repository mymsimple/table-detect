#ps aux|grep main.train|grep -v grep|awk '{print $1}'|xargs kill -9
ps -ef|grep main.train|awk '{print $2}'|xargs kill -9