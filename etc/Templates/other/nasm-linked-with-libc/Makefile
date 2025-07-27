NAME := nasm

default: assemble link

assemble:
	nasm -f elf64 -o $(NAME).o $(NAME).asm

link:
	ld -o $(NAME) -dynamic-linker /lib/ld-linux-x86-64.so.2 /lib/crt1.o /lib/crti.o $(NAME).o -lc /lib/crtn.o

clean:
	rm -fv $(NAME).o $(NAME)
