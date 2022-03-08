import os

r, w = os.pipe()
path = "/tmp/fifo"

fifo = os.open(path, os.O_WRONLY)
print("fifo open")

try:
    f = open(r"test.signal", "rb")
    while True:
        binary_content = f.read(-1)
        if not binary_content:
            break
        # print(list(binary_content))
        # print("Length: " + str(len(list(binary_content))))
        os.write(fifo, binary_content)
    os.close(fifo)
    f.close()
except IOError:
    print("error")




