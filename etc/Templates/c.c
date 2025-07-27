#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>

#include <getopt.h>         /* For option processing */
#include <signal.h>         /* For catching signals */
#include <sys/io.h>         /* For inb, outb, etc. */
#include <fcntl.h>          /* For open, O_RDONLY, etc. */
#include <sys/mman.h>       /* For mmap, munmap, etc. */
#include <setjmp.h>         /* For setjmp and longjmp */
#include <sys/inotify.h>    /* For inotify */

#define LOG(msg, ...) printf("[*] " msg "\n", ##__VA_ARGS__)
#define DIE(msg, ...) ({ fprintf(stderr, "ERROR: " msg "\n", ##__VA_ARGS__); exit(1); })


typedef struct Config {
    char *flag;
} Config;

static const char short_opts[] = "htf:l";
static struct option long_opts[] = {
    {"help",    no_argument,         0, 'h'},
    {"test",    no_argument,         0, 't'},
    {"list",    no_argument,         0, 'l'},
    {"flag",    required_argument,   0, 'f'},
    {0, 0, 0, 0}
};

static int test() {
    LOG("Running tests");
    return 0;
}

static void list_options(struct option options[]) {
    int i = 0;
    const char *s;
    struct option *o;
    while ((o = &options[i])->name != 0) {
        s = o->name;
        printf("--%s ", s);
        i++;
    }

    printf("\n");
    return;
}

static void print_help() {
   printf("Usage: template [OPTIONS]\n\n"
          "OPTIONS:\n"
          "  -h, --help\t\tShow this help\n"
          "  -t, --test\t\tRun some tests\n"
          "  -l, --list\t\tList all options\n"
          "  -f, --flag X\t\tFlag to set\n");
}

void parse_options(int argc, char **argv, Config *config) {
    config->flag = NULL;

    int index;
    int opt = 0;
    while (opt != -1) {
        opt = getopt_long(argc, argv, short_opts, long_opts, &index);

        switch (opt) {
        case 'h':
            print_help();
            exit(0);
            break;
        case 't':
            test();
            exit(0);
            break;
        case 'l':
            list_options(long_opts);
            exit(0);
            break;
        case 'f':
            config->flag = strdup(optarg);
            break;
        default:
            /* Errors handled automatically */
            break;
        }
    }
}

int main(int argc, char **argv) {

    Config config;

    parse_options(argc, argv, &config);

    LOG("Doing main stuff");

    return 0;
}
