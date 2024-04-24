#include<stdio.h>
#include<stdlib.h>

int main(void) {

    FILE *fp = fopen("/proc/cpuinfo", "rb");
    char *line = NULL;
    size_t size = 0;
    while (getline(&line, &size, fp) != -1) {
        printf("%s\n", line);
    }
    free(line);
    fclose(fp);

    return 0;
}