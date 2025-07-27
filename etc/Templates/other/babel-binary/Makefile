CC          := gcc
WARNINGS    := -Wall
OUTFILE     := babel

default: clib rustlib asmlib
	gcc $(WARNINGS) -o $(OUTFILE) -L${PWD} main.c -Wl,-rpath,${PWD} -lclib -lrustlib -lasmlib

clib:
	gcc $(WARNINGS) -shared -fPIC -o libclib.so clib.c

rustlib:
	rustc --crate-type=cdylib rustlib.rs

asmlib:
	nasm -f elf64 asmlib.asm -o libasmlib.o
	gcc $(WARNINGS) -shared -fPIC -o libasmlib.so libasmlib.o

clean:
	rm -f *.o *.so $(OUTFILE)
