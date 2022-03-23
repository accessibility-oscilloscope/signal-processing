# Signal Processing

## Using

### Setup
```
...
export SIGCOMP_FIFO=/run/sigcomp-in
export SIGPROC_FIFO=/run/sigproc-in
...
mkfifo $SIGCOMP_FIFO
mkfifo $SIGPROC_FIFO
```


### Active Component
```
python3 -u main.py -i $SIGPROC_FIFO -o $SIGCOMP_FIFO &
```

## Testing
```
...
export SIGCOMP_FIFO=/run/sigcomp-in
export SIGPROC_FIFO=/run/sigproc-in
...
mkfifo $SIGCOMP_FIFO
mkfifo $SIGPROC_FIFO
```
### Terminal 1
`python3 -u main.py -i $SIGPROC_FIFO -o $SIGCOMP_FIFO -v`

### Terminal 2
`cat $SIGCOMP_FIFO | hexdump -C`

### Terminal 3
`cat /dev/urandom | head -c 480 > $SIGPROC_FIFO`

