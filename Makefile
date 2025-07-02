
CC = gcc
CFLAGS = -Iinclude -Iprotobuf -Imodels -Ioperators

SRC = \
    src/static_onnx_runner.c \
    protobuf/onnx.pb-c.c \
    operators/conv2d.c \
    operators/relu.c \
    operators/matmul.c \
    operators/softmax.c

OBJ = $(SRC:.c=.o)
OUT = build/run_model

all: $(OUT)

$(OUT): $(OBJ)
	$(CC) -o $@ $^ -lm

clean:
	rm -f $(OBJ) $(OUT)
