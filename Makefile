CFLAGS=-mavx2 -mfma -mtune=native -g -Ofast
all: math;

math: math.c
	$(CC) $(CFLAGS) $^ -o $@ -lm
