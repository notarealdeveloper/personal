CC          := gcc
WARNINGS    := -Wall
OUTFILE     := daemon
INFILES     := $(OUTFILE).c

default:
	$(CC) $(WARNINGS) -o $(OUTFILE) $(INFILES)

clean:
	rm -f *.o 

clean-all:
	rm -f *.o *.so $(OUTFILE)

